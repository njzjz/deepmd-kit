#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os
import shutil
import time
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
    Optional,
    TextIO,
)

import array_api_compat
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from packaging.version import (
    Version,
)

from deepmd.common import (
    symlink_prefix_files,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.loss.ener import (
    EnergyHessianLoss,
    EnergyLoss,
)
from deepmd.dpmodel.model.transform_output import (
    communicate_extended_output,
)
from deepmd.dpmodel.utils import (
    compute_total_numb_batch,
    resolve_model_prob_from_epochs,
)
from deepmd.dpmodel.utils.learning_rate import (
    LearningRateExp,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.jax.env import (
    flax_version,
    jax,
    jnp,
    nnx,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.model import (
    get_model_for_wrapper,
)
from deepmd.jax.model.multitask import (
    ModelWrapper,
)
from deepmd.jax.utils.finetune import (
    merge_finetune_model_data,
)
from deepmd.jax.utils.multi_task import (
    get_case_embd_config,
)
from deepmd.jax.utils.serialization import (
    select_model_branch,
    serialize_from_file,
)
from deepmd.loggers.training import (
    format_training_message,
    format_training_message_per_task,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.finetune import (
    FinetuneRuleItem,
)
from deepmd.utils.model_stat import (
    make_stat_input,
)

log = logging.getLogger(__name__)
_ACTIVE_JAX_MESH_CONTEXT: Any | None = None

DefModel = BaseModel | ModelWrapper
DefLoss = EnergyLoss | EnergyHessianLoss


def _merge_init_frz_model_data(
    target_node: Any,
    source_node: Any,
    *,
    path: tuple[Any, ...] = (),
    missing: list[tuple[Any, ...]] | None = None,
    unexpected: list[tuple[Any, ...]] | None = None,
) -> Any:
    """Merge overlapping frozen-model tensor leaves into a target model tree."""
    if missing is None:
        missing = []
    if unexpected is None:
        unexpected = []

    if isinstance(target_node, dict):
        merged = deepcopy(target_node)
        if not isinstance(source_node, dict):
            missing.append(path)
            return merged
        for key in source_node:
            if key not in target_node:
                unexpected.append(path + (key,))
        for key, value in target_node.items():
            if key not in source_node:
                missing.append(path + (key,))
                continue
            merged[key] = _merge_init_frz_model_data(
                value,
                source_node[key],
                path=path + (key,),
                missing=missing,
                unexpected=unexpected,
            )
        return merged

    if isinstance(target_node, list):
        merged = deepcopy(target_node)
        if not isinstance(source_node, list):
            missing.append(path)
            return merged
        if len(target_node) != len(source_node):
            raise ValueError(
                f"Shape mismatch at {path}: target list len {len(target_node)}, "
                f"source list len {len(source_node)}"
            )
        for idx, value in enumerate(target_node):
            merged[idx] = _merge_init_frz_model_data(
                value,
                source_node[idx],
                path=path + (idx,),
                missing=missing,
                unexpected=unexpected,
            )
        return merged

    if isinstance(target_node, tuple):
        if not isinstance(source_node, tuple):
            missing.append(path)
            return deepcopy(target_node)
        if len(target_node) != len(source_node):
            raise ValueError(
                f"Shape mismatch at {path}: target tuple len {len(target_node)}, "
                f"source tuple len {len(source_node)}"
            )
        return tuple(
            _merge_init_frz_model_data(
                tv,
                sv,
                path=path + (idx,),
                missing=missing,
                unexpected=unexpected,
            )
            for idx, (tv, sv) in enumerate(zip(target_node, source_node))
        )

    if isinstance(target_node, np.ndarray):
        if not isinstance(source_node, np.ndarray):
            missing.append(path)
            return np.array(target_node, copy=True)
        if target_node.shape == source_node.shape:
            return np.array(source_node, copy=True)
        if target_node.size == source_node.size:
            return np.array(source_node, copy=True).reshape(target_node.shape)
        if target_node.shape != source_node.shape:
            raise ValueError(
                f"Shape mismatch at {path}: target {target_node.shape}, "
                f"source {source_node.shape}"
            )
        return np.array(source_node, copy=True)

    return deepcopy(target_node)


def _clear_jax_mesh_for_host_ops() -> None:
    _set_nnx_eager_sharding(False)
    _set_jax_mesh(Mesh(np.empty((), dtype=object), ()))


def _set_nnx_eager_sharding(enabled: bool) -> None:
    use_eager_sharding = getattr(nnx, "use_eager_sharding", None)
    if use_eager_sharding is not None:
        use_eager_sharding(enabled)


def _set_jax_mesh(mesh: Mesh) -> None:
    """Set the global mesh across JAX versions used by CI and users."""
    global _ACTIVE_JAX_MESH_CONTEXT

    if _ACTIVE_JAX_MESH_CONTEXT is not None:
        _ACTIVE_JAX_MESH_CONTEXT.__exit__(None, None, None)
        _ACTIVE_JAX_MESH_CONTEXT = None

    set_mesh = getattr(jax, "set_mesh", None)
    if set_mesh is not None:
        set_mesh(mesh)
        return

    enter = getattr(mesh, "__enter__", None)
    if enter is None or getattr(mesh, "__exit__", None) is None:
        raise AttributeError("This JAX version cannot set a global mesh.")

    enter()
    _ACTIVE_JAX_MESH_CONTEXT = mesh


def _merge_batches_for_bias(batch_list: list[np.ndarray], key: str) -> np.ndarray | float:
    arrays = [np.asarray(item) for item in batch_list]
    if key.startswith("find_"):
        return float(np.max(arrays))
    if key in {"natoms", "natoms_vec"}:
        return np.stack(arrays, axis=0)
    if arrays and arrays[0].ndim == 0:
        return np.asarray(arrays)
    return np.concatenate(arrays, axis=0)


def _pack_data_for_bias_adjust(
    train_data: DeepmdDataSystem,
    nbatches: int,
) -> list[dict[str, np.ndarray | None]]:
    all_stat = make_stat_input(train_data, nbatches, merge_sys=False)
    all_stat["atype"] = all_stat.pop("type")
    if "natoms_vec" in all_stat:
        all_stat["natoms"] = all_stat["natoms_vec"]
    sampled = [
        {kk: _merge_batches_for_bias(vv[ii], kk) for kk, vv in all_stat.items()}
        for ii in range(train_data.get_nsystems())
    ]
    for ii, single_data in enumerate(sampled):
        for key, value in list(single_data.items()):
            single_data[key] = to_numpy_array(value)
        if not train_data.data_systems[ii].pbc:
            single_data["box"] = None
    return sampled


def model_change_out_bias(
    model: BaseModel,
    sampled: list[dict[str, np.ndarray | None]],
    bias_adjust_mode: str = "change-by-statistic",
) -> BaseModel:
    old_bias = deepcopy(to_numpy_array(model.get_out_bias()))
    try:
        model.change_out_bias(
            sampled,
            bias_adjust_mode=bias_adjust_mode,
        )
    except np.linalg.LinAlgError:
        if bias_adjust_mode != "change-by-statistic":
            raise
        log.warning(
            "change-by-statistic failed during JAX finetune bias adjustment; "
            "falling back to set-by-statistic."
        )
        model.change_out_bias(
            sampled,
            bias_adjust_mode="set-by-statistic",
        )
    new_bias = deepcopy(to_numpy_array(model.get_out_bias()))
    log.info(
        "Change output bias of %s from %s to %s.",
        model.get_type_map(),
        old_bias.reshape(-1),
        new_bias.reshape(-1),
    )
    return model


def _build_loss(
    loss_param: dict[str, Any],
    starter_learning_rate: float,
) -> tuple[DefLoss, bool]:
    loss_cfg = deepcopy(loss_param)
    loss_cfg["starter_learning_rate"] = starter_learning_rate
    loss_type = loss_cfg.get("type", "ener")
    if loss_type != "ener":
        raise RuntimeError("unknown loss type " + loss_type)
    if loss_cfg.get("start_pref_h", 0.0) > 0.0:
        return EnergyHessianLoss.get_loss(loss_cfg), True
    return EnergyLoss.get_loss(loss_cfg), False


def _compute_single_data_stat(model: BaseModel, train_data: DeepmdDataSystem) -> None:
    descriptor_stat, fitting_stat = _build_single_data_stat(train_data)
    model.atomic_model.descriptor.compute_input_stats(descriptor_stat)
    model.atomic_model.fitting.compute_output_stats(
        fitting_stat, mixed_type=train_data.mixed_type
    )


def _build_single_data_stat(
    train_data: DeepmdDataSystem,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_stat_nbatch = 10
    all_stat = make_stat_input(train_data, data_stat_nbatch, merge_sys=False)
    all_stat["atype"] = all_stat.pop("type")
    all_stat_sys = [
        {
            kk: jnp.asarray(np.concatenate(vv[ii], axis=0))
            for kk, vv in all_stat.items()
            if not kk.startswith("find_")
        }
        for ii in range(train_data.get_nsystems())
    ]
    for ii, single_data in enumerate(all_stat_sys):
        if not train_data.data_systems[ii].pbc:
            single_data["box"] = None
    return all_stat_sys, all_stat


def _apply_weighted_shared_fitting_input_stats(
    component: Any,
    sampled_stats_by_branch: list[list[dict[str, Any]]],
    branch_weights: list[float],
    protection: float,
) -> None:
    if getattr(component, "numb_fparam", 0) > 0:
        weighted_sum = np.zeros(component.numb_fparam, dtype=np.float64)
        weighted_sum_sq = np.zeros(component.numb_fparam, dtype=np.float64)
        weighted_count = 0.0
        for sampled_stats, branch_weight in zip(
            sampled_stats_by_branch, branch_weights, strict=True
        ):
            cat_data = np.concatenate(
                [np.asarray(frame["fparam"]) for frame in sampled_stats], axis=0
            )
            cat_data = np.reshape(cat_data, [-1, component.numb_fparam]).astype(np.float64)
            weighted_sum += branch_weight * np.sum(cat_data, axis=0)
            weighted_sum_sq += branch_weight * np.sum(cat_data * cat_data, axis=0)
            weighted_count += branch_weight * cat_data.shape[0]
        fparam_avg = weighted_sum / weighted_count
        fparam_std = np.sqrt(
            np.maximum(weighted_sum_sq / weighted_count - fparam_avg**2, 0.0)
        )
        fparam_std = np.where(
            fparam_std < protection,
            np.array(protection, dtype=fparam_std.dtype),
            fparam_std,
        )
        xp = array_api_compat.array_namespace(component.fparam_avg)
        component.fparam_avg = xp.asarray(
            fparam_avg,
            dtype=component.fparam_avg.dtype,
            device=array_api_compat.device(component.fparam_avg),
        )
        component.fparam_inv_std = xp.asarray(
            1.0 / fparam_std,
            dtype=component.fparam_inv_std.dtype,
            device=array_api_compat.device(component.fparam_inv_std),
        )

    if getattr(component, "numb_aparam", 0) > 0:
        weighted_sum = np.zeros(component.numb_aparam, dtype=np.float64)
        weighted_sum_sq = np.zeros(component.numb_aparam, dtype=np.float64)
        weighted_count = 0.0
        for sampled_stats, branch_weight in zip(
            sampled_stats_by_branch, branch_weights, strict=True
        ):
            cat_data = np.concatenate(
                [np.asarray(frame["aparam"]) for frame in sampled_stats], axis=0
            )
            cat_data = np.reshape(cat_data, [-1, component.numb_aparam]).astype(np.float64)
            weighted_sum += branch_weight * np.sum(cat_data, axis=0)
            weighted_sum_sq += branch_weight * np.sum(cat_data * cat_data, axis=0)
            weighted_count += branch_weight * cat_data.shape[0]
        aparam_avg = weighted_sum / weighted_count
        aparam_std = np.sqrt(
            np.maximum(weighted_sum_sq / weighted_count - aparam_avg**2, 0.0)
        )
        aparam_std = np.where(
            aparam_std < protection,
            np.array(protection, dtype=aparam_std.dtype),
            aparam_std,
        )
        xp = array_api_compat.array_namespace(component.aparam_avg)
        component.aparam_avg = xp.asarray(
            aparam_avg,
            dtype=component.aparam_avg.dtype,
            device=array_api_compat.device(component.aparam_avg),
        )
        component.aparam_inv_std = xp.asarray(
            1.0 / aparam_std,
            dtype=component.aparam_inv_std.dtype,
            device=array_api_compat.device(component.aparam_inv_std),
        )


def _compute_multitask_data_stat(
    model: ModelWrapper,
    train_data: dict[str, DeepmdDataSystem],
    model_key_prob_map: dict[str, float],
    data_stat_protect_map: dict[str, float],
) -> None:
    stat_cache = {
        model_key: _build_single_data_stat(train_data[model_key])
        for model_key in model.keys()
    }

    descriptor_groups: dict[int, dict[str, Any]] = {}
    fitting_input_groups: dict[tuple[int, int | None, int | None], dict[str, Any]] = {}
    for model_key in model.keys():
        branch_model = model[model_key]
        branch_fitting = branch_model.atomic_model.fitting
        descriptor_id = id(branch_model.atomic_model.descriptor)
        descriptor_groups.setdefault(
            descriptor_id,
            {
                "component": branch_model.atomic_model.descriptor,
                "stats": [],
            },
        )["stats"].append(stat_cache[model_key][0])
        fitting_group_id = (
            id(branch_fitting.nets),
            id(branch_fitting.fparam_avg)
            if getattr(branch_fitting, "numb_fparam", 0) > 0
            else None,
            id(branch_fitting.aparam_avg)
            if getattr(branch_fitting, "numb_aparam", 0) > 0
            else None,
        )
        fitting_input_groups.setdefault(
            fitting_group_id,
            {
                "component": branch_fitting,
                "mixed_type": train_data[model_key].mixed_type,
                "weights": [],
                "samples": [],
                "data_stat_protect": data_stat_protect_map[model_key],
            },
        )
        if (
            fitting_input_groups[fitting_group_id]["mixed_type"]
            != train_data[model_key].mixed_type
        ):
            raise ValueError(
                "All branches sharing a fitting_net must use the same mixed_type setting."
            )
        if not np.isclose(
            fitting_input_groups[fitting_group_id]["data_stat_protect"],
            data_stat_protect_map[model_key],
        ):
            raise ValueError(
                "Model key 'data_stat_protect' must be the same in each branch when multitask!"
            )
        fitting_input_groups[fitting_group_id]["weights"].append(
            model_key_prob_map[model_key]
        )
        fitting_input_groups[fitting_group_id]["samples"].append(stat_cache[model_key][0])

    for descriptor_group in descriptor_groups.values():
        merged_descriptor_stat = []
        for single_stat in descriptor_group["stats"]:
            merged_descriptor_stat.extend(single_stat)
        descriptor_group["component"].compute_input_stats(merged_descriptor_stat)

    for fitting_group in fitting_input_groups.values():
        _apply_weighted_shared_fitting_input_stats(
            fitting_group["component"],
            fitting_group["samples"],
            fitting_group["weights"],
            protection=fitting_group["data_stat_protect"],
        )

    for model_key in model.keys():
        branch_fitting = model[model_key].atomic_model.fitting
        branch_fitting.compute_output_stats(
            stat_cache[model_key][1],
            mixed_type=train_data[model_key].mixed_type,
        )


def _resolve_model_prob_multi(
    model_keys: list[str],
    training_params: dict[str, Any],
    train_data: dict[str, DeepmdDataSystem],
) -> tuple[np.ndarray, int]:
    num_steps = training_params.get("numb_steps")
    num_epoch_dict = training_params.get("num_epoch_dict", {})
    if num_epoch_dict:
        if num_steps is not None:
            raise ValueError(
                "training.numb_steps and training.num_epoch_dict are mutually exclusive."
            )
        per_task_total = []
        for model_key in model_keys:
            sampler_weights = np.asarray(train_data[model_key].sys_probs, dtype=np.float64)
            per_task_total.append(
                compute_total_numb_batch(train_data[model_key].nbatches, sampler_weights)
            )
        model_prob, resolved_num_steps, _ = resolve_model_prob_from_epochs(
            model_keys,
            num_epoch_dict,
            np.asarray(per_task_total, dtype=np.float64),
        )
        return model_prob, resolved_num_steps

    if num_steps is None:
        raise ValueError(
            "Either training.numb_steps (multi-task only) or training.num_epoch_dict must be set."
        )
    model_prob_config = training_params.get("model_prob", {})
    if model_prob_config:
        missing = [k for k in model_keys if k not in model_prob_config]
        if missing:
            raise ValueError(
                f"training.model_prob must specify all tasks; missing: {missing}"
            )
        model_prob = np.asarray(
            [float(model_prob_config[k]) for k in model_keys], dtype=np.float64
        )
    else:
        model_prob = np.asarray(
            [float(train_data[k].get_nsystems()) for k in model_keys], dtype=np.float64
        )
    if np.any(model_prob < 0.0) or not np.all(np.isfinite(model_prob)):
        raise ValueError("training.model_prob must be non-negative and finite.")
    prob_sum = float(np.sum(model_prob))
    if prob_sum <= 0.0:
        raise ValueError("training.model_prob must sum to a positive value.")
    return model_prob / prob_sum, int(num_steps)


class DPTrainer:
    def __init__(
        self,
        jdata: dict,
        init_model: Optional[str] = None,
        restart: Optional[str] = None,
        init_frz_model: Optional[str] = None,
        finetune_model: Optional[str] = None,
        force_load: bool = False,
        shared_links: dict[str, Any] | None = None,
        finetune_links: dict[str, FinetuneRuleItem] | None = None,
        finetune_model_data: dict[str, Any] | None = None,
    ) -> None:
        self.init_model = init_model
        self.restart = restart
        self.init_frz_model = init_frz_model
        self.finetune_model = finetune_model
        self.force_load = force_load
        self.shared_links = shared_links or {}
        self.finetune_links = finetune_links or {}
        self.finetune_model_data = finetune_model_data
        self.model_def_script = deepcopy(jdata["model"])
        if "model_dict" in self.model_def_script and self.shared_links:
            self.model_def_script["shared_links"] = deepcopy(self.shared_links)
        self.training_param = jdata["training"]
        self.num_steps = self.training_param.get("numb_steps")
        self.start_step = 0
        self.multi_task = "model_dict" in jdata["model"]
        self.model_keys = (
            list(jdata["model"]["model_dict"])
            if self.multi_task
            else ["Default"]
        )
        if self.multi_task:
            _, self.case_embd_index = get_case_embd_config(jdata["model"])
        else:
            self.case_embd_index = {}

        learning_rate_param = deepcopy(jdata["learning_rate"])
        self.learning_rate_param = learning_rate_param
        self.lr = LearningRateExp(
            **learning_rate_param,
            num_steps=self.num_steps or 1,
        )

        if self.multi_task:
            self.loss: dict[str, DefLoss] = {}
            self.branch_has_hessian: dict[str, bool] = {}
            for model_key in self.model_keys:
                loss_param = jdata["loss_dict"][model_key]
                self.loss[model_key], self.branch_has_hessian[model_key] = _build_loss(
                    loss_param,
                    learning_rate_param["start_lr"],
                )
        else:
            self.loss, has_hessian = _build_loss(
                jdata.get("loss", {}),
                learning_rate_param["start_lr"],
            )
            self.branch_has_hessian = {"Default": has_hessian}

        self.model: DefModel = get_model_for_wrapper(
            jdata["model"],
            shared_links=self.shared_links,
        )
        self._apply_hessian_flags(self.model)

        if self.init_model is not None:
            model_dict = serialize_from_file(self.init_model)
            self._load_model_data(model_dict)
        elif self.restart is not None:
            model_dict = serialize_from_file(self.restart)
            self._load_model_data(model_dict)
            self.model_def_script = deepcopy(model_dict["model_def_script"])
            self.start_step = model_dict["@variables"].get("current_step", 0)
        if self.init_frz_model is not None:
            frozen_model_data = serialize_from_file(self.init_frz_model)
            self._load_frozen_model_data(frozen_model_data)

        tr_data = self.training_param
        self.disp_file = tr_data.get("disp_file", "lcurve.out")
        self.disp_freq = tr_data.get("disp_freq", 1000)
        self.save_freq = tr_data.get("save_freq", 1000)
        self.save_ckpt = tr_data.get("save_ckpt", "model.ckpt")
        self.max_ckpt_keep = tr_data.get("max_ckpt_keep", 5)
        self.display_in_training = tr_data.get("disp_training", True)
        self.timing_in_training = tr_data.get("time_training", True)
        self.profiling = tr_data.get("profiling", False)
        self.profiling_file = tr_data.get("profiling_file", "timeline.json")
        self.enable_profiler = tr_data.get("enable_profiler", False)
        self.tensorboard = tr_data.get("tensorboard", False)
        self.tensorboard_log_dir = tr_data.get("tensorboard_log_dir", "log")
        self.tensorboard_freq = tr_data.get("tensorboard_freq", 1)
        self.mixed_prec = tr_data.get("mixed_precision", None)
        self.change_bias_after_training = tr_data.get(
            "change_bias_after_training", False
        )
        if self.multi_task:
            self.data_bias_nsample = {
                model_key: jdata["model"]["model_dict"][model_key].get(
                    "data_bias_nsample", 10
                )
                for model_key in self.model_keys
            }
        else:
            self.data_bias_nsample = self.model_def_script.get("data_bias_nsample", 10)
        self.model_prob = None
        self.ckpt_meta = None
        self.model_type = None

    def _apply_hessian_flags(self, model: DefModel) -> None:
        if self.multi_task:
            assert isinstance(model, ModelWrapper)
            for model_key in self.model_keys:
                if self.branch_has_hessian[model_key]:
                    model[model_key].enable_hessian()
        else:
            if isinstance(model, ModelWrapper):
                raise TypeError("single-task JAX trainer expected a single-task model.")
            if self.branch_has_hessian["Default"]:
                model.enable_hessian()

    def _load_model_data(self, model_data: dict[str, Any]) -> None:
        serialized_model = model_data["model"]
        if self.force_load:
            missing: list[tuple[Any, ...]] = []
            unexpected: list[tuple[Any, ...]] = []
            serialized_model = _merge_init_frz_model_data(
                self.model.serialize(),
                serialized_model,
                missing=missing,
                unexpected=unexpected,
            )
            if missing or unexpected:
                log.warning(
                    "Checkpoint loaded in force_load mode. Missing keys reinitialized: %s; Unexpected keys ignored: %s",
                    [".".join(map(str, item)) for item in missing[:20]],
                    [".".join(map(str, item)) for item in unexpected[:20]],
                )
        effective_shared_links = (
            self.shared_links
            or model_data.get("model_def_script", {}).get("shared_links", {})
        )
        if self.multi_task:
            if "model_dict" not in serialized_model:
                raise ValueError(
                    "init_model/restart for JAX multitask target requires a multitask checkpoint."
                )
            self.model = ModelWrapper.deserialize(
                serialized_model,
                shared_links=effective_shared_links,
                case_embd_index=self.case_embd_index,
            )
        else:
            if "model_dict" in serialized_model:
                raise ValueError(
                    "init_model/restart for single-task JAX target does not accept a multitask checkpoint."
                )
            self.model = BaseModel.deserialize(serialized_model)
        self._apply_hessian_flags(self.model)

    def _validate_shared_finetune_rules(self) -> None:
        if not self.multi_task or not self.shared_links or not self.finetune_links:
            return
        for shared_key, link_info in self.shared_links.items():
            for link in link_info.get("links", []):
                shared_level = int(link.get("shared_level", 0))
                if link["shared_type"] == "fitting_net" and shared_level != 0:
                    raise NotImplementedError(
                        "JAX multitask finetune fitting_net sharing only supports shared_level=0, "
                        f"but got '{shared_key}' at level {shared_level}."
                    )

    def _load_frozen_model_data(self, model_data: dict[str, Any]) -> None:
        missing: list[tuple[Any, ...]] = []
        unexpected: list[tuple[Any, ...]] = []
        merged_model_data = _merge_init_frz_model_data(
            self.model.serialize(),
            model_data["model"],
            missing=missing,
            unexpected=unexpected,
        )
        if self.multi_task:
            self.model = ModelWrapper.deserialize(
                merged_model_data,
                shared_links=self.shared_links,
                case_embd_index=self.case_embd_index,
            )
        else:
            self.model = BaseModel.deserialize(merged_model_data)
        self._apply_hessian_flags(self.model)
        if missing or unexpected:
            log.warning(
                "Frozen model loaded non-strictly. Missing keys: %s, Unexpected keys: %s",
                [".".join(map(str, item)) for item in missing[:20]],
                [".".join(map(str, item)) for item in unexpected[:20]],
            )

    @staticmethod
    def _shared_type_to_serialized_paths(
        shared_type: str,
        shared_level: int,
        source_model_data: dict[str, Any],
    ) -> list[tuple[Any, ...]]:
        def descriptor_paths(
            path_prefix: tuple[Any, ...],
            descriptor_data: dict[str, Any],
        ) -> list[tuple[Any, ...]]:
            descriptor_type = descriptor_data.get("type")
            if shared_level == 0:
                return [path_prefix]
            if shared_level == 1 and descriptor_type in {
                "dpa1",
                "dpa2",
                "dpa3",
                "se_e3_tebd",
            }:
                return [path_prefix + ("type_embedding",)]
            raise NotImplementedError(
                "JAX multitask finetune does not support shared override for "
                f"{shared_type}:{shared_level} with descriptor type {descriptor_type}."
            )

        if shared_type == "descriptor":
            return descriptor_paths(("descriptor",), source_model_data["descriptor"])
        if shared_type.startswith("descriptor_hybrid_"):
            idx = int(shared_type.rsplit("_", 1)[1])
            return descriptor_paths(
                ("descriptor", "list", idx),
                source_model_data["descriptor"]["list"][idx],
            )
        if shared_type == "fitting_net":
            if shared_level != 0:
                raise NotImplementedError(
                    "JAX multitask finetune fitting_net sharing only supports shared_level=0, "
                    f"but got {shared_level}."
                )
            fitting_paths = [("fitting", "nets")]
            fitting_vars = source_model_data.get("fitting", {}).get("@variables", {})
            for key in fitting_vars:
                if key not in {"bias_atom_e", "case_embd"}:
                    fitting_paths.append(("fitting", "@variables", key))
            return fitting_paths
        return []

    def _collect_shared_source_overrides(
        self,
        source_multi: bool,
    ) -> dict[str, dict[tuple[Any, ...], dict[str, Any]]]:
        overrides: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {
            model_key: {} for model_key in self.model_keys
        }
        if not self.shared_links or not self.finetune_links or self.finetune_model_data is None:
            return overrides

        for _, link_info in self.shared_links.items():
            shareable_links = [
                link
                for link in link_info.get("links", [])
                if (
                    link["shared_type"].startswith("descriptor")
                    or link["shared_type"] == "fitting_net"
                )
            ]
            if not shareable_links:
                continue
            base_link = shareable_links[0]
            canonical_source_key = self.finetune_links[base_link["model_key"]].get_model_branch()
            canonical_source_model_data = (
                self.finetune_model_data["model"]["model_dict"][canonical_source_key]
                if source_multi
                else self.finetune_model_data["model"]
            )
            for link in shareable_links[1:]:
                current_source_key = self.finetune_links[link["model_key"]].get_model_branch()
                if current_source_key == canonical_source_key:
                    continue
                paths = self._shared_type_to_serialized_paths(
                    link["shared_type"],
                    int(link.get("shared_level", 0)),
                    canonical_source_model_data,
                )
                for path in paths:
                    overrides[link["model_key"]][path] = canonical_source_model_data
        return overrides

    @property
    def data_requirements(self) -> list[DataRequirementItem] | dict[str, list[DataRequirementItem]]:
        if self.multi_task:
            return {
                model_key: self.loss[model_key].label_requirement
                for model_key in self.model_keys
            }
        return self.loss.label_requirement

    def _apply_single_finetune(
        self,
        target_model: BaseModel,
        pretrained_model_data: dict[str, Any],
        finetune_rule: FinetuneRuleItem,
        *,
        source_overrides: dict[tuple[Any, ...], dict[str, Any]] | None = None,
    ) -> BaseModel:
        if self.force_load:
            missing: list[tuple[Any, ...]] = []
            unexpected: list[tuple[Any, ...]] = []
            pretrained_model_data = _merge_init_frz_model_data(
                target_model.serialize(),
                pretrained_model_data,
                missing=missing,
                unexpected=unexpected,
            )
            if missing or unexpected:
                log.warning(
                    "Finetune checkpoint loaded in force_load mode. Missing keys reinitialized: %s; Unexpected keys ignored: %s",
                    [".".join(map(str, item)) for item in missing[:20]],
                    [".".join(map(str, item)) for item in unexpected[:20]],
                )
        pretrained_model = BaseModel.deserialize(pretrained_model_data)
        if finetune_rule.get_update_type():
            pretrained_model.change_type_map(
                target_model.get_type_map(),
                model_with_new_type_stat=target_model.atomic_model,
            )
        merged_model_data = merge_finetune_model_data(
            target_model.serialize(),
            pretrained_model.serialize(),
            finetune_rule,
            source_overrides=source_overrides,
        )
        return BaseModel.deserialize(merged_model_data)

    def _finetune_single(self, train_data: DeepmdDataSystem) -> None:
        if isinstance(self.model, ModelWrapper):
            raise TypeError("single-task JAX finetune expected a single-task model.")
        if self.finetune_model_data is None:
            self.finetune_model_data = serialize_from_file(self.finetune_model)
        finetune_rule = self.finetune_links["Default"]
        pretrained_data = self.finetune_model_data
        if "model_dict" in pretrained_data.get("model_def_script", {}):
            pretrained_data = select_model_branch(
                pretrained_data,
                finetune_rule.get_model_branch(),
            )
        self.model = self._apply_single_finetune(
            self.model,
            pretrained_data["model"],
            finetune_rule,
        )
        self._apply_hessian_flags(self.model)
        self.model = model_change_out_bias(
            self.model,
            _pack_data_for_bias_adjust(train_data, self.data_bias_nsample),
            bias_adjust_mode=(
                "set-by-statistic"
                if finetune_rule.get_random_fitting()
                else "change-by-statistic"
            ),
        )

    def _finetune_multi(self, train_data: dict[str, DeepmdDataSystem]) -> None:
        if not isinstance(self.model, ModelWrapper):
            raise TypeError("multitask JAX finetune expected a multitask model.")
        self._validate_shared_finetune_rules()
        if self.finetune_model_data is None:
            self.finetune_model_data = serialize_from_file(self.finetune_model)
        source_multi = "model_dict" in self.finetune_model_data.get("model", {})
        shared_source_overrides = self._collect_shared_source_overrides(source_multi)
        merged_branch_models: dict[str, Any] = {}
        for model_key in self.model_keys:
            branch_model = self.model[model_key]
            finetune_rule = self.finetune_links[model_key]
            source_key = finetune_rule.get_model_branch()
            source_model_data = (
                self.finetune_model_data["model"]["model_dict"][source_key]
                if source_multi
                else self.finetune_model_data["model"]
            )
            merged_branch_model = self._apply_single_finetune(
                branch_model,
                source_model_data,
                finetune_rule,
                source_overrides=shared_source_overrides.get(model_key),
            )
            if self.branch_has_hessian[model_key]:
                merged_branch_model.enable_hessian()
            if not finetune_rule.get_resuming():
                log.info(
                    "Model branch %s will be fine-tuned. This may take a long time...",
                    model_key,
                )
                merged_branch_model = model_change_out_bias(
                    merged_branch_model,
                    _pack_data_for_bias_adjust(
                        train_data[model_key], self.data_bias_nsample[model_key]
                    ),
                    bias_adjust_mode=(
                        "set-by-statistic"
                        if finetune_rule.get_random_fitting()
                        else "change-by-statistic"
                    ),
                )
            else:
                log.info("Model branch %s will resume training.", model_key)
            merged_branch_models[model_key] = merged_branch_model.serialize()
        self.model = ModelWrapper.deserialize(
            {"model_dict": merged_branch_models},
            shared_links=self.shared_links,
            case_embd_index=self.case_embd_index,
        )
        self._apply_hessian_flags(self.model)

    def train(
        self,
        train_data: DeepmdDataSystem | dict[str, DeepmdDataSystem],
        valid_data: DeepmdDataSystem | dict[str, DeepmdDataSystem | None] | None = None,
    ) -> None:
        if self.multi_task:
            assert isinstance(train_data, dict)
            valid_data = valid_data if isinstance(valid_data, dict) else {}
            self._train_multi(train_data, valid_data)
        else:
            assert isinstance(train_data, DeepmdDataSystem)
            valid_data = valid_data if isinstance(valid_data, DeepmdDataSystem) else None
            self._train_single(train_data, valid_data)

    def _train_single(
        self,
        train_data: DeepmdDataSystem,
        valid_data: DeepmdDataSystem | None = None,
    ) -> None:
        model = self.model
        if isinstance(model, ModelWrapper):
            raise TypeError("single-task JAX training expected a single-task model.")
        _clear_jax_mesh_for_host_ops()
        tx = optax.adam(
            learning_rate=lambda step: self.lr.value(self.start_step + step),
        )

        finetune_rule = self.finetune_links.get("Default")
        finetune_has_new_type = (
            self.finetune_model is not None
            and finetune_rule is not None
            and finetune_rule.get_has_new_type()
        )
        if (
            self.init_model is None
            and self.restart is None
            and (self.finetune_model is None or finetune_has_new_type)
        ):
            _compute_single_data_stat(model, train_data)

        if self.finetune_model is not None:
            self._finetune_single(train_data)
            model = self.model
            if isinstance(model, ModelWrapper):
                raise TypeError("single-task JAX finetune produced a multitask model unexpectedly.")

        auto_mesh = jax.make_mesh(
            (jax.process_count(), jax.local_device_count()),
            ("data", "natoms"),
        )
        _set_nnx_eager_sharding(True)
        _set_jax_mesh(auto_mesh)
        sharding = (
            NamedSharding(auto_mesh, P("data"))
            if int(os.environ.get("DP_JAX_MULTI_NPROC", "0")) > 1
            else None
        )
        model = BaseModel.deserialize(model.serialize())
        self._apply_hessian_flags(model)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        def loss_fn(
            model: BaseModel,
            lr: float,
            label_dict: dict[str, jnp.ndarray],
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            mapping: jnp.ndarray | None,
            fp: jnp.ndarray | None,
            ap: jnp.ndarray | None,
        ) -> jnp.ndarray:
            model_dict_lower = model.call_common_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            model_dict = communicate_extended_output(
                model_dict_lower,
                model.model_output_def(),
                mapping,
                do_atomic_virial=False,
            )
            loss, _ = self.loss(
                learning_rate=lr,
                natoms=label_dict["coord"].shape[1],
                model_dict=model_dict,
                label_dict=label_dict,
            )
            return loss

        @nnx.jit
        def loss_fn_more_loss(
            model: BaseModel,
            lr: float,
            label_dict: dict[str, jnp.ndarray],
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            mapping: jnp.ndarray | None,
            fp: jnp.ndarray | None,
            ap: jnp.ndarray | None,
        ) -> dict[str, jnp.ndarray]:
            model_dict_lower = model.call_common_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            model_dict = communicate_extended_output(
                model_dict_lower,
                model.model_output_def(),
                mapping,
                do_atomic_virial=False,
            )
            _, more_loss = self.loss(
                learning_rate=lr,
                natoms=label_dict["coord"].shape[1],
                model_dict=model_dict,
                label_dict=label_dict,
            )
            return more_loss

        @nnx.jit
        def train_step(
            model: BaseModel,
            optimizer: nnx.Optimizer,
            lr: float,
            label_dict: dict[str, jnp.ndarray],
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            mapping: jnp.ndarray | None,
            fp: jnp.ndarray | None,
            ap: jnp.ndarray | None,
        ) -> None:
            grads = nnx.grad(loss_fn)(
                model,
                lr,
                label_dict,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            if Version(flax_version) >= Version("0.11.0"):
                optimizer.update(model, grads)
            else:
                optimizer.update(grads)

        start_time = time.time()
        disp_file_fp = open(self.disp_file, "w")
        for step in range(self.start_step, self.num_steps):
            batch_data = train_data.get_batch()
            jax_data = convert_numpy_data_to_jax_data(
                batch_data,
                sharding,
                natoms_axis_size=auto_mesh.shape.get("natoms", 1),
            )
            extended_coord, extended_atype, nlist, mapping, fp, ap = prepare_input(
                rcut=model.get_rcut(),
                sel=model.get_sel(),
                coord=jax_data["coord"],
                atype=jax_data["type"],
                box=jax_data["box"] if jax_data["default_mesh"].size > 1 else None,
                fparam=jax_data.get("fparam", None),
                aparam=jax_data.get("aparam", None),
            )
            train_step(
                model,
                optimizer,
                self.lr.value(step),
                jax_data,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            if self.display_in_training and (step == 0 or (step + 1) % self.disp_freq == 0):
                wall_time = time.time() - start_time
                log.info(format_training_message(batch=step + 1, wall_time=wall_time))
                more_loss = loss_fn_more_loss(
                    model,
                    self.lr.value(step),
                    jax_data,
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping,
                    fp,
                    ap,
                )
                if valid_data is not None:
                    valid_batch_data = valid_data.get_batch()
                    jax_valid_data = convert_numpy_data_to_jax_data(
                        valid_batch_data,
                        sharding,
                        natoms_axis_size=auto_mesh.shape.get("natoms", 1),
                    )
                    extended_coord, extended_atype, nlist, mapping, fp, ap = prepare_input(
                        rcut=model.get_rcut(),
                        sel=model.get_sel(),
                        coord=jax_valid_data["coord"],
                        atype=jax_valid_data["type"],
                        box=jax_valid_data["box"] if jax_valid_data["find_box"] else None,
                            fparam=jax_valid_data.get("fparam", None),
                            aparam=jax_valid_data.get("aparam", None),
                        )
                    valid_more_loss = loss_fn_more_loss(
                        model,
                        self.lr.value(step),
                        jax_valid_data,
                        extended_coord,
                        extended_atype,
                        nlist,
                        mapping,
                        fp,
                        ap,
                    )
                else:
                    valid_more_loss = None
                if step == 0:
                    self.print_header(disp_file_fp, more_loss, valid_more_loss)
                self.print_on_training(
                    disp_file_fp,
                    more_loss,
                    valid_more_loss,
                    cur_batch=step + 1,
                    cur_lr=self.lr.value(step),
                )
                start_time = time.time()
            if (step + 1) % self.save_freq == 0:
                self._save_checkpoint(model, step + 1)
                log.info(f"Trained model has been saved to: {Path(f'{self.save_ckpt}-{step + 1}.jax')!s}")
        disp_file_fp.close()
        self.model = model

    def _train_multi(
        self,
        train_data: dict[str, DeepmdDataSystem],
        valid_data: dict[str, DeepmdDataSystem | None],
    ) -> None:
        model = self.model
        assert isinstance(model, ModelWrapper)
        _clear_jax_mesh_for_host_ops()
        self.model_prob, self.num_steps = _resolve_model_prob_multi(
            self.model_keys,
            self.training_param,
            train_data,
        )
        finetune_has_new_type = (
            self.finetune_model is not None
            and any(rule.get_has_new_type() for rule in self.finetune_links.values())
        )
        if self.init_model is None and self.restart is None:
            if self.finetune_model is None or finetune_has_new_type:
                data_stat_protect_map = {
                    model_key: float(
                        self.model_def_script["model_dict"][model_key].get(
                            "data_stat_protect", 1e-2
                        )
                    )
                    for model_key in self.model_keys
                }
                _compute_multitask_data_stat(
                    model,
                    train_data,
                    dict(zip(self.model_keys, self.model_prob, strict=True)),
                    data_stat_protect_map,
                )
        if self.finetune_model is not None:
            self._finetune_multi(train_data)
            model = self.model
            assert isinstance(model, ModelWrapper)
        self.lr = LearningRateExp(
            **self.learning_rate_param,
            num_steps=self.num_steps,
        )
        tx = optax.adam(
            learning_rate=lambda step: self.lr.value(self.start_step + step),
        )

        auto_mesh = jax.make_mesh(
            (jax.process_count(), jax.local_device_count()),
            ("data", "natoms"),
        )
        _set_nnx_eager_sharding(True)
        _set_jax_mesh(auto_mesh)
        sharding = (
            NamedSharding(auto_mesh, P("data"))
            if int(os.environ.get("DP_JAX_MULTI_NPROC", "0")) > 1
            else None
        )
        model = ModelWrapper.deserialize(
            model.serialize(),
            shared_links=self.shared_links,
            case_embd_index=self.case_embd_index,
        )
        self._apply_hessian_flags(model)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        loss_fns = {}
        more_loss_fns = {}
        train_step_fns = {}
        for model_key in self.model_keys:
            branch_loss = self.loss[model_key]

            def make_loss_fn(task_key: str, task_loss: DefLoss):
                def loss_fn(
                    wrapper: ModelWrapper,
                    lr: float,
                    label_dict: dict[str, jnp.ndarray],
                    extended_coord: jnp.ndarray,
                    extended_atype: jnp.ndarray,
                    nlist: jnp.ndarray,
                    mapping: jnp.ndarray | None,
                    fp: jnp.ndarray | None,
                    ap: jnp.ndarray | None,
                ) -> jnp.ndarray:
                    branch_model = wrapper[task_key]
                    model_dict_lower = branch_model.call_common_lower(
                        extended_coord,
                        extended_atype,
                        nlist,
                        mapping,
                        fp,
                        ap,
                    )
                    model_dict = communicate_extended_output(
                        model_dict_lower,
                        branch_model.model_output_def(),
                        mapping,
                        do_atomic_virial=False,
                    )
                    loss, _ = task_loss(
                        learning_rate=lr,
                        natoms=label_dict["coord"].shape[1],
                        model_dict=model_dict,
                        label_dict=label_dict,
                    )
                    return loss

                return loss_fn

            def make_more_loss_fn(task_key: str, task_loss: DefLoss):
                @nnx.jit
                def more_loss_fn(
                    wrapper: ModelWrapper,
                    lr: float,
                    label_dict: dict[str, jnp.ndarray],
                    extended_coord: jnp.ndarray,
                    extended_atype: jnp.ndarray,
                    nlist: jnp.ndarray,
                    mapping: jnp.ndarray | None,
                    fp: jnp.ndarray | None,
                    ap: jnp.ndarray | None,
                ) -> dict[str, jnp.ndarray]:
                    branch_model = wrapper[task_key]
                    model_dict_lower = branch_model.call_common_lower(
                        extended_coord,
                        extended_atype,
                        nlist,
                        mapping,
                        fp,
                        ap,
                    )
                    model_dict = communicate_extended_output(
                        model_dict_lower,
                        branch_model.model_output_def(),
                        mapping,
                        do_atomic_virial=False,
                    )
                    _, more_loss = task_loss(
                        learning_rate=lr,
                        natoms=label_dict["coord"].shape[1],
                        model_dict=model_dict,
                        label_dict=label_dict,
                    )
                    return more_loss

                return more_loss_fn

            def make_train_step(task_loss_fn):
                @nnx.jit
                def train_step(
                    wrapper: ModelWrapper,
                    optimizer: nnx.Optimizer,
                    lr: float,
                    label_dict: dict[str, jnp.ndarray],
                    extended_coord: jnp.ndarray,
                    extended_atype: jnp.ndarray,
                    nlist: jnp.ndarray,
                    mapping: jnp.ndarray | None,
                    fp: jnp.ndarray | None,
                    ap: jnp.ndarray | None,
                ) -> None:
                    grads = nnx.grad(task_loss_fn)(
                        wrapper,
                        lr,
                        label_dict,
                        extended_coord,
                        extended_atype,
                        nlist,
                        mapping,
                        fp,
                        ap,
                    )
                    if Version(flax_version) >= Version("0.11.0"):
                        optimizer.update(wrapper, grads)
                    else:
                        optimizer.update(grads)

                return train_step

            loss_fns[model_key] = make_loss_fn(model_key, branch_loss)
            more_loss_fns[model_key] = make_more_loss_fn(model_key, branch_loss)
            train_step_fns[model_key] = make_train_step(loss_fns[model_key])

        start_time = time.time()
        disp_file_fp = open(self.disp_file, "w")
        for step in range(self.start_step, self.num_steps):
            model_index = np.random.choice(
                np.arange(len(self.model_keys), dtype=np.int_),
                p=self.model_prob,
            )
            task_key = self.model_keys[model_index]
            model.set_case_embd(task_key)
            batch_data = train_data[task_key].get_batch()
            jax_data = convert_numpy_data_to_jax_data(
                batch_data,
                sharding,
                natoms_axis_size=auto_mesh.shape.get("natoms", 1),
            )
            branch_model = model[task_key]
            extended_coord, extended_atype, nlist, mapping, fp, ap = prepare_input(
                rcut=branch_model.get_rcut(),
                sel=branch_model.get_sel(),
                coord=jax_data["coord"],
                atype=jax_data["type"],
                box=jax_data["box"] if jax_data["default_mesh"].size > 1 else None,
                fparam=jax_data.get("fparam", None),
                aparam=jax_data.get("aparam", None),
            )
            train_step_fns[task_key](
                model,
                optimizer,
                self.lr.value(step),
                jax_data,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            if self.display_in_training and (step == 0 or (step + 1) % self.disp_freq == 0):
                wall_time = time.time() - start_time
                log.info(format_training_message(batch=step + 1, wall_time=wall_time))
                train_results = {_key: {} for _key in self.model_keys}
                valid_results = {_key: {} for _key in self.model_keys}
                model.set_case_embd(task_key)
                train_results[task_key] = more_loss_fns[task_key](
                    model,
                    self.lr.value(step),
                    jax_data,
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping,
                    fp,
                    ap,
                )
                for _key in self.model_keys:
                    if _key != task_key:
                        train_batch_data = train_data[_key].get_batch()
                        jax_train_data = convert_numpy_data_to_jax_data(
                            train_batch_data,
                            sharding,
                            natoms_axis_size=auto_mesh.shape.get("natoms", 1),
                        )
                        branch_model = model[_key]
                        (
                            train_extended_coord,
                            train_extended_atype,
                            train_nlist,
                            train_mapping,
                            train_fp,
                            train_ap,
                        ) = prepare_input(
                            rcut=branch_model.get_rcut(),
                            sel=branch_model.get_sel(),
                            coord=jax_train_data["coord"],
                            atype=jax_train_data["type"],
                            box=jax_train_data["box"]
                            if jax_train_data["default_mesh"].size > 1
                            else None,
                            fparam=jax_train_data.get("fparam", None),
                            aparam=jax_train_data.get("aparam", None),
                        )
                        model.set_case_embd(_key)
                        train_results[_key] = more_loss_fns[_key](
                            model,
                            self.lr.value(step),
                            jax_train_data,
                            train_extended_coord,
                            train_extended_atype,
                            train_nlist,
                            train_mapping,
                            train_fp,
                            train_ap,
                        )
                    if valid_data.get(_key) is not None:
                        valid_batch_data = valid_data[_key].get_batch()
                        jax_valid_data = convert_numpy_data_to_jax_data(
                            valid_batch_data,
                            sharding,
                            natoms_axis_size=auto_mesh.shape.get("natoms", 1),
                        )
                        branch_model = model[_key]
                        (
                            valid_extended_coord,
                            valid_extended_atype,
                            valid_nlist,
                            valid_mapping,
                            valid_fp,
                            valid_ap,
                        ) = prepare_input(
                            rcut=branch_model.get_rcut(),
                            sel=branch_model.get_sel(),
                            coord=jax_valid_data["coord"],
                            atype=jax_valid_data["type"],
                            box=jax_valid_data["box"] if jax_valid_data["find_box"] else None,
                            fparam=jax_valid_data.get("fparam", None),
                            aparam=jax_valid_data.get("aparam", None),
                        )
                        model.set_case_embd(_key)
                        valid_results[_key] = more_loss_fns[_key](
                            model,
                            self.lr.value(step),
                            jax_valid_data,
                            valid_extended_coord,
                            valid_extended_atype,
                            valid_nlist,
                            valid_mapping,
                            valid_fp,
                            valid_ap,
                        )
                if step == 0:
                    self.print_header_multitask(disp_file_fp, train_results, valid_results)
                self.print_on_training_multitask(
                    disp_file_fp,
                    train_results,
                    valid_results,
                    cur_batch=step + 1,
                    cur_lr=self.lr.value(step),
                )
                start_time = time.time()
            if (step + 1) % self.save_freq == 0:
                self._save_checkpoint(model, step + 1)
                log.info(f"Trained model has been saved to: {Path(f'{self.save_ckpt}-{step + 1}.jax')!s}")
        disp_file_fp.close()
        self.model = model

    def _save_checkpoint(self, model: DefModel, step: int) -> None:
        _, state = nnx.split(model)
        ckpt_path = Path(f"{self.save_ckpt}-{step}.jax")
        if ckpt_path.is_dir():
            shutil.rmtree(ckpt_path)
        model_def_script_cpy = deepcopy(self.model_def_script)
        model_def_script_cpy["current_step"] = step
        with ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "model_def_script")
        ) as checkpointer:
            checkpointer.save(
                ckpt_path.absolute(),
                ocp.args.Composite(
                    state=ocp.args.StandardSave(state.to_pure_dict()),
                    model_def_script=ocp.args.JsonSave(model_def_script_cpy),
                ),
            )
        symlink_prefix_files(f"{self.save_ckpt}-{step}", self.save_ckpt)
        with open("checkpoint", "w") as fp:
            fp.write(f"{self.save_ckpt}.jax")

    @staticmethod
    def print_on_training(
        fp: TextIO,
        train_results: dict[str, float],
        valid_results: dict[str, float] | None,
        cur_batch: int,
        cur_lr: float,
    ) -> None:
        print_str = f"{cur_batch:7d}"
        if valid_results is not None:
            prop_fmt = "   %11.2e %11.2e"
            for key in valid_results.keys():
                print_str += prop_fmt % (valid_results[key], train_results[key])
        else:
            prop_fmt = "   %11.2e"
            for key in train_results.keys():
                print_str += prop_fmt % (train_results[key])
        print_str += f"   {cur_lr:8.1e}\n"
        log.info(
            format_training_message_per_task(
                batch=cur_batch,
                task_name="trn",
                rmse=train_results,
                learning_rate=cur_lr,
            )
        )
        if valid_results is not None:
            log.info(
                format_training_message_per_task(
                    batch=cur_batch,
                    task_name="val",
                    rmse=valid_results,
                    learning_rate=None,
                )
            )
        fp.write(print_str)
        fp.flush()

    @staticmethod
    def print_on_training_multitask(
        fp: TextIO,
        train_results: dict[str, dict[str, float]],
        valid_results: dict[str, dict[str, float]],
        cur_batch: int,
        cur_lr: float,
    ) -> None:
        print_str = f"{cur_batch:7d}"
        for model_key in train_results.keys():
            if valid_results.get(model_key):
                prop_fmt = "   %11.2e %11.2e"
                for key in sorted(train_results[model_key].keys()):
                    print_str += prop_fmt % (
                        valid_results[model_key][key],
                        train_results[model_key][key],
                    )
            else:
                prop_fmt = "   %11.2e"
                for key in sorted(train_results[model_key].keys()):
                    print_str += prop_fmt % (train_results[model_key][key])
        print_str += f"   {cur_lr:8.1e}\n"
        for model_key in train_results.keys():
            log.info(
                format_training_message_per_task(
                    batch=cur_batch,
                    task_name=model_key + "_trn",
                    rmse=train_results[model_key],
                    learning_rate=cur_lr,
                )
            )
            if valid_results.get(model_key):
                log.info(
                    format_training_message_per_task(
                        batch=cur_batch,
                        task_name=model_key + "_val",
                        rmse=valid_results[model_key],
                        learning_rate=None,
                    )
                )
        fp.write(print_str)
        fp.flush()

    @staticmethod
    def print_header(
        fp: TextIO,
        train_results: dict[str, float],
        valid_results: dict[str, float] | None,
    ) -> None:
        print_str = "# {:5s}".format("step")
        if valid_results is not None:
            prop_fmt = "   %11s %11s"
            for key in train_results.keys():
                print_str += prop_fmt % (key + "_val", key + "_trn")
        else:
            prop_fmt = "   %11s"
            for key in train_results.keys():
                print_str += prop_fmt % (key + "_trn")
        print_str += "   {:8s}\n".format("lr")
        print_str += "# If there is no available reference data, rmse_*_{val,trn} will print nan\n"
        fp.write(print_str)
        fp.flush()

    @staticmethod
    def print_header_multitask(
        fp: TextIO,
        train_results: dict[str, dict[str, float]],
        valid_results: dict[str, dict[str, float]],
    ) -> None:
        print_str = "# {:5s}".format("step")
        for model_key in train_results.keys():
            if valid_results.get(model_key):
                prop_fmt = "   %11s %11s"
                for key in sorted(train_results[model_key].keys()):
                    print_str += prop_fmt % (
                        key + f"_val_{model_key}",
                        key + f"_trn_{model_key}",
                    )
            else:
                prop_fmt = "   %11s"
                for key in sorted(train_results[model_key].keys()):
                    print_str += prop_fmt % (key + f"_trn_{model_key}")
        print_str += "   {:8s}\n".format("lr")
        print_str += "# If there is no available reference data, rmse_*_{val,trn} will print nan\n"
        fp.write(print_str)
        fp.flush()


def prepare_input(
    *,
    rcut: float,
    sel: list[int],
    coord: np.ndarray,
    atype: np.ndarray,
    box: Optional[np.ndarray] = None,
    fparam: Optional[np.ndarray] = None,
    aparam: Optional[np.ndarray] = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    nframes, nloc = atype.shape[:2]
    cc, bb, fp, ap = coord, box, fparam, aparam
    del coord, box, fparam, aparam
    if bb is not None:
        coord_normalized = normalize_coord(
            cc.reshape(nframes, nloc, 3),
            bb.reshape(nframes, 3, 3),
        )
    else:
        coord_normalized = cc.copy()
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord_normalized, atype, bb, rcut
    )
    nlist = build_neighbor_list(
        extended_coord,
        extended_atype,
        nloc,
        rcut,
        sel,
        distinguish_types=False,
    )
    extended_coord = extended_coord.reshape(nframes, -1, 3)
    return extended_coord, extended_atype, nlist, mapping, fp, ap


def convert_numpy_data_to_jax_data(
    numpy_data: dict[str, np.ndarray | np.floating],
    sharding: Any | None = None,
    natoms_axis_size: int = 1,
) -> dict[str, jnp.ndarray | bool]:
    jax_data = {
        kk: jnp.asarray(vv) if not kk.startswith("find_") else bool(vv.item())
        for kk, vv in numpy_data.items()
    }
    if sharding is not None:
        jax_data = {
            kk: jax.make_array_from_process_local_data(sharding, vv)
            if not kk.startswith("find_")
            and vv is not None
            and kk not in {"natoms_vec", "default_mesh"}
            else vv
            for kk, vv in jax_data.items()
        }

    def _label_sharding(key: str, value: jnp.ndarray) -> Any:
        if key in {"energy", "box", "numb_copy", "virial", "real_natoms_vec"}:
            return P("data")
        if natoms_axis_size <= 1 or value.ndim < 2:
            return P("data")
        if value.shape[1] % natoms_axis_size != 0:
            return P("data")
        return P("data", "natoms")

    jax_data = {
        kk: jax.device_put(
            vv,
            _label_sharding(kk, vv),
        )
        if not kk.startswith("find_")
        and vv is not None
        and kk not in {"natoms_vec", "default_mesh"}
        else vv
        for kk, vv in jax_data.items()
    }
    return jax_data
