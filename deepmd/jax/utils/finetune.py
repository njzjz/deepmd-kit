# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.jax.utils.serialization import (
    serialize_from_file,
)
from deepmd.utils.finetune import (
    FinetuneRuleItem,
)
from deepmd.utils.model_branch_dict import (
    get_model_dict,
)

log = logging.getLogger(__name__)


def _validate_finetune_source(finetune_model: str) -> dict[str, Any]:
    if finetune_model.endswith(".savedmodel"):
        raise ValueError(
            "JAX fine-tuning does not support loading from .savedmodel. "
            "Please use a .jax checkpoint or a .hlo frozen model."
        )
    if not (finetune_model.endswith(".jax") or finetune_model.endswith(".hlo")):
        raise ValueError(
            "JAX fine-tuning only supports .jax checkpoints or .hlo frozen models."
        )
    return serialize_from_file(finetune_model)


def get_finetune_rule_single(
    target_model_config: dict[str, Any],
    pretrained_model_config: dict[str, Any],
    *,
    from_multitask: bool = False,
    model_branch: str = "Default",
    model_branch_from: str = "",
    change_model_params: bool = False,
) -> tuple[dict[str, Any], FinetuneRuleItem]:
    single_config = deepcopy(target_model_config)
    new_fitting = False
    model_branch_chosen = "Default"

    if not from_multitask:
        single_config_chosen = deepcopy(pretrained_model_config)
        if model_branch_from == "RANDOM":
            new_fitting = True
    else:
        model_dict_params = pretrained_model_config["model_dict"]
        if model_branch_from in ("", "RANDOM"):
            model_branch_chosen = next(iter(model_dict_params.keys()))
            new_fitting = True
            log.warning(
                "The fitting net will be re-initialized instead of using that in the "
                "pretrained multitask model because no explicit source branch was "
                "selected."
            )
        else:
            model_branch_chosen = model_branch_from
        model_alias_dict, _ = get_model_dict(model_dict_params)
        if model_branch_from not in model_alias_dict:
            if model_branch_from in ("", "RANDOM"):
                model_branch_from = model_branch_chosen
            else:
                raise ValueError(
                    f"No model branch or alias named '{model_branch_from}'. "
                    f"Available ones are {list(model_dict_params.keys())}."
                )
        if model_branch_from not in ("", "RANDOM"):
            model_branch_chosen = model_branch_from
        if model_branch_chosen not in model_alias_dict:
            raise ValueError(
                f"No model branch or alias named '{model_branch_chosen}'. "
                f"Available ones are {list(model_dict_params.keys())}."
            )
        model_branch_chosen = model_alias_dict[model_branch_chosen]
        single_config_chosen = deepcopy(model_dict_params[model_branch_chosen])

    finetune_rule = FinetuneRuleItem(
        p_type_map=single_config_chosen["type_map"],
        type_map=single_config["type_map"],
        model_branch=model_branch_chosen,
        random_fitting=new_fitting,
    )
    if change_model_params:
        trainable_param = {
            "descriptor": single_config.get("descriptor", {}).get("trainable", True),
            "fitting_net": single_config.get("fitting_net", {}).get("trainable", True),
        }
        single_config["descriptor"] = deepcopy(single_config_chosen["descriptor"])
        if not new_fitting:
            single_config["fitting_net"] = deepcopy(single_config_chosen["fitting_net"])
        log.info(
            "Change the '%s' model configurations according to the model branch '%s' in the pretrained one...",
            model_branch,
            model_branch_chosen,
        )
        for net_type, trainable in trainable_param.items():
            if net_type in single_config:
                single_config[net_type]["trainable"] = trainable
            else:
                single_config[net_type] = {"trainable": trainable}
    return single_config, finetune_rule


def get_finetune_rules(
    finetune_model: str,
    model_config: dict[str, Any],
    *,
    model_branch: str = "",
    change_model_params: bool = True,
) -> tuple[dict[str, Any], dict[str, FinetuneRuleItem], dict[str, Any]]:
    finetune_data = _validate_finetune_source(finetune_model)
    pretrained_model_config = finetune_data["model_def_script"]
    finetune_from_multi_task = "model_dict" in pretrained_model_config
    multi_task = "model_dict" in model_config
    finetune_links: dict[str, FinetuneRuleItem] = {}

    if not multi_task:
        if model_branch == "" and "finetune_head" in model_config:
            model_branch = model_config["finetune_head"]

        updated_model_config, finetune_rule = get_finetune_rule_single(
            model_config,
            pretrained_model_config,
            from_multitask=finetune_from_multi_task,
            model_branch="Default",
            model_branch_from=model_branch,
            change_model_params=change_model_params,
        )
        finetune_links["Default"] = finetune_rule
        return updated_model_config, finetune_links, finetune_data

    if model_branch != "":
        raise AssertionError(
            "Multi-task fine-tuning does not support command-line branches chosen!"
            "Please define the 'finetune_head' in each model params!"
        )

    target_keys = model_config["model_dict"].keys()
    pretrained_keys = (
        pretrained_model_config["model_dict"].keys()
        if finetune_from_multi_task
        else ["Default"]
    )

    for model_key in target_keys:
        branch_config = model_config["model_dict"][model_key]
        model_branch_from = "RANDOM"
        resuming = False
        if (
            "finetune_head" in branch_config
            and branch_config["finetune_head"] != "RANDOM"
        ):
            pretrained_key = branch_config["finetune_head"]
            if pretrained_key not in pretrained_keys:
                raise AssertionError(
                    f"'{pretrained_key}' head chosen to finetune not exist in the pretrained model!"
                    f"Available heads are: {list(pretrained_keys)}"
                )
            model_branch_from = pretrained_key
        elif "finetune_head" not in branch_config and model_key in pretrained_keys:
            model_branch_from = model_key
            resuming = True

        model_config["model_dict"][model_key], finetune_rule = get_finetune_rule_single(
            branch_config,
            pretrained_model_config,
            from_multitask=finetune_from_multi_task,
            model_branch=model_key,
            model_branch_from=model_branch_from,
            change_model_params=change_model_params,
        )
        finetune_rule.resuming = resuming
        finetune_links[model_key] = finetune_rule
    return model_config, finetune_links, finetune_data


def _merge_array_leaves(
    target_node: Any,
    source_node: Any,
    *,
    path: tuple[Any, ...] = (),
    keep_target_on_shape_mismatch: bool = False,
) -> Any:
    if isinstance(target_node, dict):
        if not isinstance(source_node, dict):
            raise TypeError(
                f"Expected dict at {path}, got {type(source_node).__name__}"
            )
        merged = deepcopy(target_node)
        for key, value in target_node.items():
            if key not in source_node:
                raise KeyError(f"Missing key {'.'.join(map(str, (*path, key)))}")
            merged[key] = _merge_array_leaves(
                value,
                source_node[key],
                path=(*path, key),
                keep_target_on_shape_mismatch=keep_target_on_shape_mismatch,
            )
        return merged
    if isinstance(target_node, list):
        if not isinstance(source_node, list) or len(target_node) != len(source_node):
            raise ValueError(f"List mismatch at {path}")
        return [
            _merge_array_leaves(
                tv,
                sv,
                path=(*path, idx),
                keep_target_on_shape_mismatch=keep_target_on_shape_mismatch,
            )
            for idx, (tv, sv) in enumerate(zip(target_node, source_node, strict=True))
        ]
    if isinstance(target_node, tuple):
        if not isinstance(source_node, tuple) or len(target_node) != len(source_node):
            raise ValueError(f"Tuple mismatch at {path}")
        return tuple(
            _merge_array_leaves(
                tv,
                sv,
                path=(*path, idx),
                keep_target_on_shape_mismatch=keep_target_on_shape_mismatch,
            )
            for idx, (tv, sv) in enumerate(zip(target_node, source_node, strict=True))
        )
    if isinstance(target_node, np.ndarray):
        if not isinstance(source_node, np.ndarray):
            raise TypeError(
                f"Expected ndarray at {path}, got {type(source_node).__name__}"
            )
        if target_node.shape == source_node.shape:
            return np.array(source_node, copy=True)
        if target_node.size == source_node.size:
            return np.array(source_node, copy=True).reshape(target_node.shape)
        if keep_target_on_shape_mismatch:
            log.info(
                "Keeping target-initialized leaf at %s due to shape mismatch: target %s, source %s",
                ".".join(map(str, path)),
                target_node.shape,
                source_node.shape,
            )
            return np.array(target_node, copy=True)
        raise ValueError(
            f"Shape mismatch at {path}: target {target_node.shape}, source {source_node.shape}"
        )
    return deepcopy(target_node)


def _get_by_path(node: Any, path: tuple[Any, ...]) -> Any:
    current = node
    for key in path:
        current = current[key]
    return current


def _set_by_path(node: Any, path: tuple[Any, ...], value: Any) -> None:
    current = node
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def merge_finetune_model_data(
    target_model_data: dict[str, Any],
    pretrained_model_data: dict[str, Any],
    finetune_rule: FinetuneRuleItem,
    *,
    source_overrides: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    target_model_data = deepcopy(target_model_data)
    target_template = deepcopy(target_model_data)
    pretrained_model_data = deepcopy(pretrained_model_data)
    source_overrides = source_overrides or {}
    if finetune_rule.get_random_fitting():
        if (
            "descriptor" not in target_model_data
            or "descriptor" not in pretrained_model_data
        ):
            raise NotImplementedError(
                "JAX random-fitting fine-tuning currently requires standard descriptor/fitting models."
            )
        target_model_data["descriptor"] = _merge_array_leaves(
            target_model_data["descriptor"],
            pretrained_model_data["descriptor"],
            path=("descriptor",),
        )
        merged_model_data = target_model_data
    else:
        target_case_embd_dim = target_model_data.get("fitting", {}).get(
            "dim_case_embd", 0
        )
        source_case_embd_dim = pretrained_model_data.get("fitting", {}).get(
            "dim_case_embd", 0
        )
        if target_case_embd_dim != source_case_embd_dim:
            if (
                "descriptor" in target_model_data
                and "descriptor" in pretrained_model_data
            ):
                target_model_data["descriptor"] = _merge_array_leaves(
                    target_model_data["descriptor"],
                    pretrained_model_data["descriptor"],
                    path=("descriptor",),
                )
            if (
                "fitting" not in target_model_data
                or "fitting" not in pretrained_model_data
            ):
                raise NotImplementedError(
                    "JAX case embedding fine-tuning currently requires standard descriptor/fitting models."
                )
            target_model_data["fitting"] = _merge_array_leaves(
                target_model_data["fitting"],
                pretrained_model_data["fitting"],
                path=("fitting",),
                keep_target_on_shape_mismatch=True,
            )
            if (
                "@variables" in target_model_data
                and "@variables" in pretrained_model_data
            ):
                target_model_data["@variables"] = _merge_array_leaves(
                    target_model_data["@variables"],
                    pretrained_model_data["@variables"],
                    path=("@variables",),
                )
            merged_model_data = target_model_data
        else:
            merged_model_data = _merge_array_leaves(
                target_model_data, pretrained_model_data
            )
    for path, override_source_model_data in source_overrides.items():
        source_subtree = _get_by_path(override_source_model_data, path)
        target_subtree = _get_by_path(target_template, path)
        _set_by_path(
            merged_model_data,
            path,
            _merge_array_leaves(
                target_subtree,
                source_subtree,
                path=path,
            ),
        )
    return merged_model_data
