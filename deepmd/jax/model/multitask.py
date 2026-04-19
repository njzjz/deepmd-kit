# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

from typing import (
    Any,
)

from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)


def _branch_attr(branch: str) -> str:
    return f"branch__{branch}"


def _get_component(model: BaseModel, shared_type: str) -> Any:
    if shared_type == "descriptor":
        return model.atomic_model.descriptor
    if shared_type == "fitting_net":
        return model.atomic_model.fitting
    if shared_type.startswith("descriptor_hybrid_"):
        idx = int(shared_type.rsplit("_", 1)[1])
        return model.atomic_model.descriptor.list[idx]
    raise NotImplementedError(f"Unsupported shared_type {shared_type}")


def _set_component(model: BaseModel, shared_type: str, value: Any) -> None:
    if shared_type == "descriptor":
        model.atomic_model.descriptor = value
        return
    if shared_type == "fitting_net":
        model.atomic_model.fitting = value
        return
    if shared_type.startswith("descriptor_hybrid_"):
        idx = int(shared_type.rsplit("_", 1)[1])
        model.atomic_model.descriptor.list[idx] = value
        return
    raise NotImplementedError(f"Unsupported shared_type {shared_type}")


def _share_fitting_component(base_model: BaseModel, link_model: BaseModel) -> None:
    """Mirror PT fitting sharing semantics without aliasing branch-local state.

    PT shared fitting keeps `bias_atom_e` and `case_embd` per branch, while
    sharing the inner network modules plus the fparam/aparam normalization
    buffers. JAX needs the same behavior for multitask parity.
    """

    base_fitting = base_model.atomic_model.fitting
    link_fitting = link_model.atomic_model.fitting
    if base_fitting.__class__ is not link_fitting.__class__:
        raise TypeError("Only fitting nets of the same type can share params!")
    object.__setattr__(link_fitting, "nets", base_fitting.nets)
    if getattr(base_fitting, "numb_fparam", 0) > 0:
        object.__setattr__(link_fitting, "fparam_avg", base_fitting.fparam_avg)
        object.__setattr__(
            link_fitting,
            "fparam_inv_std",
            base_fitting.fparam_inv_std,
        )
    if getattr(base_fitting, "numb_aparam", 0) > 0:
        object.__setattr__(link_fitting, "aparam_avg", base_fitting.aparam_avg)
        object.__setattr__(
            link_fitting,
            "aparam_inv_std",
            base_fitting.aparam_inv_std,
        )


def _check_supported_share(link_info: dict[str, Any]) -> None:
    links = link_info.get("links", [])
    for link in links:
        shared_type = link["shared_type"]
        shared_level = int(link.get("shared_level", 0))
        if shared_level != 0:
            raise NotImplementedError(
                "JAX multitask only supports full parameter sharing "
                f"(shared_level=0), but got {shared_type}:{shared_level}."
            )
        if shared_type.startswith("descriptor_hybrid_"):
            raise NotImplementedError(
                "JAX multitask does not support hybrid descriptor sub-component "
                "sharing yet."
            )


@flax_module
class ModelWrapper:
    def __init__(
        self,
        model_dict: dict[str, BaseModel],
        *,
        shared_links: dict[str, Any] | None = None,
        case_embd_index: dict[str, int] | None = None,
    ) -> None:
        self.model_keys = list(model_dict.keys())
        self.shared_links = shared_links or {}
        self.case_embd_index = case_embd_index or {}
        for key, model in model_dict.items():
            setattr(self, _branch_attr(key), model)
        if self.shared_links:
            self.share_params(self.shared_links)
        for key in self.model_keys:
            self.set_case_embd(key)

    def keys(self) -> list[str]:
        return list(self.model_keys)

    def items(self) -> list[tuple[str, BaseModel]]:
        return [(key, self[key]) for key in self.model_keys]

    def __getitem__(self, key: str) -> BaseModel:
        return getattr(self, _branch_attr(key))

    def __setitem__(self, key: str, model: BaseModel) -> None:
        setattr(self, _branch_attr(key), model)
        if self.shared_links:
            self.share_params(self.shared_links)
        self.set_case_embd(key)

    def get_type_map(self, key: str) -> list[str]:
        return self[key].get_type_map()

    def share_params(self, shared_links: dict[str, Any] | None = None) -> None:
        shared_links = shared_links or self.shared_links
        for _, link_info in shared_links.items():
            _check_supported_share(link_info)
            links = link_info.get("links", [])
            if not links:
                continue
            base_link = links[0]
            base_model = self[base_link["model_key"]]
            if base_link["shared_type"] == "fitting_net":
                for link in links[1:]:
                    _share_fitting_component(base_model, self[link["model_key"]])
                continue
            base_component = _get_component(base_model, base_link["shared_type"])
            for link in links[1:]:
                _set_component(
                    self[link["model_key"]],
                    link["shared_type"],
                    base_component,
                )

    def set_case_embd(self, key: str) -> None:
        if key in self.case_embd_index:
            self[key].set_case_embd(self.case_embd_index[key])

    def serialize(self) -> dict[str, Any]:
        return {
            "model_dict": {
                key: self[key].serialize() for key in self.model_keys
            }
        }

    @classmethod
    def deserialize(
        cls,
        data: dict[str, Any],
        *,
        shared_links: dict[str, Any] | None = None,
        case_embd_index: dict[str, int] | None = None,
    ) -> "ModelWrapper":
        model_dict = {
            key: BaseModel.deserialize(value) for key, value in data["model_dict"].items()
        }
        return cls(
            model_dict,
            shared_links=shared_links,
            case_embd_index=case_embd_index,
        )
