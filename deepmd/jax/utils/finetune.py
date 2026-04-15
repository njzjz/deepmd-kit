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


def _preserve_trainable_flags(
    updated_config: dict[str, Any],
    target_config: dict[str, Any],
) -> None:
    for key, value in target_config.items():
        if key.startswith("trainable"):
            updated_config[key] = deepcopy(value)
        elif (
            isinstance(value, dict)
            and key in updated_config
            and isinstance(updated_config[key], dict)
        ):
            _preserve_trainable_flags(updated_config[key], value)


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
        updated_descriptor = deepcopy(single_config_chosen["descriptor"])
        _preserve_trainable_flags(
            updated_descriptor,
            single_config.get("descriptor", {}),
        )
        single_config["descriptor"] = updated_descriptor
        if not new_fitting:
            updated_fitting = deepcopy(single_config_chosen["fitting_net"])
            _preserve_trainable_flags(
                updated_fitting,
                single_config.get("fitting_net", {}),
            )
            single_config["fitting_net"] = updated_fitting
        log.info(
            "Change the '%s' model configurations according to the model branch '%s' in the pretrained one...",
            model_branch,
            model_branch_chosen,
        )
    return single_config, finetune_rule


def get_finetune_rules(
    finetune_model: str,
    model_config: dict[str, Any],
    *,
    model_branch: str = "",
    change_model_params: bool = True,
) -> tuple[dict[str, Any], dict[str, FinetuneRuleItem], dict[str, Any]]:
    if "model_dict" in model_config:
        raise NotImplementedError(
            "JAX single-task fine-tuning currently does not support multitask targets."
        )

    finetune_data = _validate_finetune_source(finetune_model)
    pretrained_model_config = finetune_data["model_def_script"]
    finetune_from_multi_task = "model_dict" in pretrained_model_config
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
    return updated_model_config, {"Default": finetune_rule}, finetune_data


def _merge_array_leaves(
    target_node: Any,
    source_node: Any,
    *,
    path: tuple[Any, ...] = (),
) -> Any:
    if isinstance(target_node, dict):
        if not isinstance(source_node, dict):
            raise TypeError(f"Expected dict at {path}, got {type(source_node).__name__}")
        merged = deepcopy(target_node)
        for key, value in target_node.items():
            if key not in source_node:
                raise KeyError(f"Missing key {'.'.join(map(str, path + (key,)))}")
            merged[key] = _merge_array_leaves(
                value,
                source_node[key],
                path=path + (key,),
            )
        return merged
    if isinstance(target_node, list):
        if not isinstance(source_node, list) or len(target_node) != len(source_node):
            raise ValueError(f"List mismatch at {path}")
        return [
            _merge_array_leaves(tv, sv, path=path + (idx,))
            for idx, (tv, sv) in enumerate(zip(target_node, source_node))
        ]
    if isinstance(target_node, tuple):
        if not isinstance(source_node, tuple) or len(target_node) != len(source_node):
            raise ValueError(f"Tuple mismatch at {path}")
        return tuple(
            _merge_array_leaves(tv, sv, path=path + (idx,))
            for idx, (tv, sv) in enumerate(zip(target_node, source_node))
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
        raise ValueError(
            f"Shape mismatch at {path}: target {target_node.shape}, source {source_node.shape}"
        )
    return deepcopy(target_node)


def merge_finetune_model_data(
    target_model_data: dict[str, Any],
    pretrained_model_data: dict[str, Any],
    finetune_rule: FinetuneRuleItem,
) -> dict[str, Any]:
    target_model_data = deepcopy(target_model_data)
    pretrained_model_data = deepcopy(pretrained_model_data)
    if finetune_rule.get_random_fitting():
        if "descriptor" not in target_model_data or "descriptor" not in pretrained_model_data:
            raise NotImplementedError(
                "JAX random-fitting fine-tuning currently requires standard descriptor/fitting models."
            )
        target_model_data["descriptor"] = _merge_array_leaves(
            target_model_data["descriptor"],
            pretrained_model_data["descriptor"],
            path=("descriptor",),
        )
        return target_model_data
    return _merge_array_leaves(target_model_data, pretrained_model_data)
