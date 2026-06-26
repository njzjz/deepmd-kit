# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import numpy as np
import orbax.checkpoint as ocp

from deepmd.dpmodel.utils.serialization import (
    load_dp_model,
    save_dp_model,
)
from deepmd.jax.env import (
    jax,
    jax_export,
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
from deepmd.jax.utils.multi_task import (
    get_case_embd_config,
)
from deepmd.utils.model_branch_dict import (
    get_model_dict,
)


def _is_topology_mismatch_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "Topology mismatch detected" in message
        or "available devices are different from the devices used to save the checkpoint"
        in message
    )


def select_model_branch(
    data: dict,
    model_branch: str | None,
) -> dict:
    model_def_script = data["model_def_script"]
    if "model_dict" not in model_def_script:
        return data
    if not model_branch:
        raise ValueError(
            "Freezing a multitask JAX checkpoint to a single model requires "
            "selecting a branch with --head/--model-branch."
        )
    model_alias_dict, _ = get_model_dict(model_def_script["model_dict"])
    if model_branch not in model_alias_dict:
        raise ValueError(
            f"No model branch or alias named '{model_branch}'. "
            f"Available branches are: {list(model_def_script['model_dict'].keys())}"
        )
    model_branch = model_alias_dict[model_branch]
    return {
        **data,
        "model_def_script": deepcopy(model_def_script["model_dict"][model_branch]),
        "model": deepcopy(data["model"]["model_dict"][model_branch]),
    }


def deserialize_to_file(model_file: str, data: dict, hessian: bool = False) -> None:
    """Deserialize the dictionary to a model file."""
    if model_file.endswith(".jax"):
        model_def_script = data["model_def_script"].copy()
        shared_links = model_def_script.get("shared_links")
        if "model_dict" in model_def_script:
            _, case_embd_index = get_case_embd_config(model_def_script)
            model = ModelWrapper.deserialize(
                data["model"],
                shared_links=shared_links,
                case_embd_index=case_embd_index,
            )
            if hessian:
                raise ValueError(
                    "Freezing Hessian into a multitask .jax checkpoint is not supported. "
                    "Please select a single branch first."
                )
        else:
            model = BaseModel.deserialize(data["model"])
            if hessian:
                model.enable_hessian()
                model_def_script["hessian_mode"] = True
        _, state = nnx.split(model)
        with ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "model_def_script")
        ) as checkpointer:
            checkpointer.save(
                Path(model_file).absolute(),
                ocp.args.Composite(
                    state=ocp.args.StandardSave(state.to_pure_dict()),
                    model_def_script=ocp.args.JsonSave(model_def_script),
                ),
            )
    elif model_file.endswith(".hlo"):
        if "model_dict" in data["model_def_script"]:
            raise ValueError(
                "Freezing a multitask JAX checkpoint to .hlo requires selecting a single branch with --head/--model-branch."
            )
        model = BaseModel.deserialize(data["model"])
        model_def_script = data["model_def_script"]
        if hessian:
            model.enable_hessian()
            model_def_script["hessian_mode"] = True
        call_lower = model.call_common_lower

        nf, nloc, nghost = jax_export.symbolic_shape("nf, nloc, nghost")

        def exported_whether_do_atomic_virial(
            do_atomic_virial: bool, has_ghost_atoms: bool
        ) -> "jax_export.Exported":
            def call_lower_with_fixed_do_atomic_virial(
                coord: jnp.ndarray,
                atype: jnp.ndarray,
                nlist: jnp.ndarray,
                mapping: jnp.ndarray,
                fparam: jnp.ndarray,
                aparam: jnp.ndarray,
            ) -> dict[str, jnp.ndarray]:
                return call_lower(
                    coord,
                    atype,
                    nlist,
                    mapping,
                    fparam,
                    aparam,
                    do_atomic_virial=do_atomic_virial,
                )

            if has_ghost_atoms:
                nghost_ = nghost
            else:
                nghost_ = 0

            return jax_export.export(jax.jit(call_lower_with_fixed_do_atomic_virial))(
                jax.ShapeDtypeStruct((nf, nloc + nghost_, 3), jnp.float64),
                jax.ShapeDtypeStruct((nf, nloc + nghost_), jnp.int32),
                jax.ShapeDtypeStruct((nf, nloc, model.get_nnei()), jnp.int64),
                jax.ShapeDtypeStruct((nf, nloc + nghost_), jnp.int64),
                jax.ShapeDtypeStruct((nf, model.get_dim_fparam()), jnp.float64)
                if model.get_dim_fparam()
                else None,
                jax.ShapeDtypeStruct((nf, nloc, model.get_dim_aparam()), jnp.float64)
                if model.get_dim_aparam()
                else None,
            )

        exported = exported_whether_do_atomic_virial(
            do_atomic_virial=False, has_ghost_atoms=True
        )
        exported_atomic_virial = exported_whether_do_atomic_virial(
            do_atomic_virial=True, has_ghost_atoms=True
        )
        serialized: bytearray = exported.serialize()
        serialized_atomic_virial = exported_atomic_virial.serialize()

        exported_no_ghost = exported_whether_do_atomic_virial(
            do_atomic_virial=False, has_ghost_atoms=False
        )
        exported_atomic_virial_no_ghost = exported_whether_do_atomic_virial(
            do_atomic_virial=True, has_ghost_atoms=False
        )
        serialized_no_ghost: bytearray = exported_no_ghost.serialize()
        serialized_atomic_virial_no_ghost = exported_atomic_virial_no_ghost.serialize()

        data = data.copy()
        data.setdefault("@variables", {})
        data["@variables"]["stablehlo"] = np.void(serialized)
        data["@variables"]["stablehlo_atomic_virial"] = np.void(
            serialized_atomic_virial
        )
        data["@variables"]["stablehlo_no_ghost"] = np.void(serialized_no_ghost)
        data["@variables"]["stablehlo_atomic_virial_no_ghost"] = np.void(
            serialized_atomic_virial_no_ghost
        )
        data["constants"] = {
            "type_map": model.get_type_map(),
            "rcut": model.get_rcut(),
            "dim_fparam": model.get_dim_fparam(),
            "dim_aparam": model.get_dim_aparam(),
            "sel_type": model.get_sel_type(),
            "is_aparam_nall": model.is_aparam_nall(),
            "model_output_type": model.model_output_type(),
            "mixed_types": model.mixed_types(),
            "min_nbor_dist": model.get_min_nbor_dist(),
            "sel": model.get_sel(),
            "has_default_fparam": model.has_default_fparam(),
            "default_fparam": model.get_default_fparam(),
        }
        save_dp_model(filename=model_file, model_dict=data)
    elif model_file.endswith(".savedmodel"):
        from deepmd.jax.jax2tf.serialization import (
            deserialize_to_file as deserialize_to_savedmodel,
        )

        return deserialize_to_savedmodel(model_file, data)
    else:
        raise ValueError("Unsupported file extension")


def serialize_from_file(model_file: str) -> dict:
    """Serialize the model file to a dictionary."""
    if model_file.endswith(".jax"):
        with ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "model_def_script")
        ) as checkpointer:
            try:
                data = checkpointer.restore(
                    Path(model_file).absolute(),
                    ocp.args.Composite(
                        state=ocp.args.StandardRestore(),
                        model_def_script=ocp.args.JsonRestore(),
                    ),
                )
            except ValueError as exc:
                if not _is_topology_mismatch_error(exc):
                    raise
                model_def_script = checkpointer.restore(
                    Path(model_file).absolute(),
                    ocp.args.Composite(model_def_script=ocp.args.JsonRestore()),
                ).model_def_script
                shared_links = model_def_script.get("shared_links")
                abstract_model = get_model_for_wrapper(
                    model_def_script,
                    shared_links=shared_links,
                )
                if "model_dict" in model_def_script:
                    for model_key in model_def_script["model_dict"]:
                        if model_def_script["model_dict"][model_key].get(
                            "hessian_mode", False
                        ):
                            abstract_model[model_key].enable_hessian()
                elif model_def_script.get("hessian_mode", False):
                    abstract_model.enable_hessian()
                _, abstract_state = nnx.split(abstract_model)
                data = checkpointer.restore(
                    Path(model_file).absolute(),
                    ocp.args.Composite(
                        state=ocp.args.StandardRestore(
                            item=abstract_state.to_pure_dict(),
                            strict=False,
                        ),
                        model_def_script=ocp.args.JsonRestore(),
                    ),
                )
        state = data.state

        def convert_str_to_int_key(item: dict) -> None:
            for key, value in item.copy().items():
                if isinstance(value, dict):
                    convert_str_to_int_key(value)
                if key.isdigit():
                    item[int(key)] = item.pop(key)

        convert_str_to_int_key(state)

        model_def_script = data.model_def_script
        current_step = model_def_script.pop("current_step", 0)
        shared_links = model_def_script.get("shared_links")
        abstract_model = get_model_for_wrapper(
            model_def_script,
            shared_links=shared_links,
        )
        if "model_dict" in model_def_script:
            for model_key in model_def_script["model_dict"]:
                if model_def_script["model_dict"][model_key].get("hessian_mode", False):
                    abstract_model[model_key].enable_hessian()
        elif model_def_script.get("hessian_mode", False):
            abstract_model.enable_hessian()
        graphdef, abstract_state = nnx.split(abstract_model)
        abstract_state.replace_by_pure_dict(state)
        model = nnx.merge(graphdef, abstract_state)
        return {
            "backend": "JAX",
            "jax_version": jax.__version__,
            "model": model.serialize(),
            "model_def_script": model_def_script,
            "@variables": {
                "current_step": current_step,
            },
        }
    elif model_file.endswith(".hlo"):
        data = load_dp_model(model_file)
        data.pop("constants")
        data["@variables"].pop("stablehlo")
        return data
    else:
        raise ValueError("JAX backend only supports converting .jax directory")
