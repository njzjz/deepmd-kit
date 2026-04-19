# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)
from typing import (
    Any,
)

from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.fitting.base_fitting import (
    BaseFitting,
)


def get_class_name(item_key: str, item_params: dict[str, Any]) -> str:
    if item_key == "descriptor":
        return BaseDescriptor.get_class_by_type(
            item_params.get("type", "se_e2_a")
        ).__name__
    if item_key == "fitting_net":
        return BaseFitting.get_class_by_type(item_params.get("type", "ener")).__name__
    raise RuntimeError(f"Unknown class_name type {item_key}")


def preprocess_shared_params(
    model_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert "model_dict" in model_config, "only multi-task model can use this method!"
    supported_types = ["type_map", "descriptor", "fitting_net"]
    shared_dict = model_config.get("shared_dict", {})
    shared_links: dict[str, Any] = {}
    type_map_keys: list[str] = []

    def replace_one_item(
        params_dict: dict[str, Any],
        key_type: str,
        key_in_dict: str,
        model_key: str,
        suffix: str = "",
        index: int | None = None,
    ) -> None:
        shared_type = key_type
        shared_key = key_in_dict
        shared_level = 0
        if ":" in key_in_dict:
            shared_key = key_in_dict.split(":")[0]
            shared_level = int(key_in_dict.split(":")[1])
        assert shared_key in shared_dict, (
            f"Appointed {shared_type} {shared_key} are not in the shared_dict! Please check the input params."
        )
        if index is None:
            params_dict[shared_type] = deepcopy(shared_dict[shared_key])
        else:
            params_dict[index] = deepcopy(shared_dict[shared_key])
        if shared_type == "type_map":
            if key_in_dict not in type_map_keys:
                type_map_keys.append(key_in_dict)
        else:
            if shared_key not in shared_links:
                class_name = get_class_name(shared_type, shared_dict[shared_key])
                shared_links[shared_key] = {"type": class_name, "links": []}
            link_item = {
                "model_key": model_key,
                "shared_type": shared_type + suffix,
                "shared_level": shared_level,
            }
            shared_links[shared_key]["links"].append(link_item)

    for model_key in model_config["model_dict"]:
        model_params_item = model_config["model_dict"][model_key]
        for item_key in list(model_params_item.keys()):
            if item_key in supported_types:
                item_params = model_params_item[item_key]
                if isinstance(item_params, str):
                    replace_one_item(
                        model_params_item,
                        item_key,
                        item_params,
                        model_key,
                    )
                elif isinstance(item_params, dict) and item_params.get("type", "") == "hybrid":
                    for ii, hybrid_item in enumerate(item_params["list"]):
                        if isinstance(hybrid_item, str):
                            replace_one_item(
                                model_params_item[item_key]["list"],
                                item_key,
                                hybrid_item,
                                model_key,
                                suffix=f"_hybrid_{ii}",
                                index=ii,
                            )
    for shared_key in shared_links:
        shared_links[shared_key]["links"] = sorted(
            shared_links[shared_key]["links"],
            key=lambda x: x["shared_level"],
        )
    assert len(type_map_keys) == 1, "Multitask model must have only one type_map!"
    return model_config, shared_links


def get_case_embd_config(model_params: dict[str, Any]) -> tuple[bool, dict[str, int]]:
    assert "model_dict" in model_params, (
        "Only support setting case embedding for multi-task model!"
    )
    model_keys = list(model_params["model_dict"])
    sorted_model_keys = sorted(model_keys)
    numb_case_embd_list = [
        model_params["model_dict"][model_key]
        .get("fitting_net", {})
        .get("dim_case_embd", 0)
        for model_key in sorted_model_keys
    ]
    if not all(item == numb_case_embd_list[0] for item in numb_case_embd_list):
        raise ValueError(
            "All models must have the same dimension of case embedding, "
            f"while the settings are: {numb_case_embd_list}"
        )
    if numb_case_embd_list[0] == 0:
        return False, {}
    case_embd_index = {
        model_key: idx for idx, model_key in enumerate(sorted_model_keys)
    }
    return True, case_embd_index
