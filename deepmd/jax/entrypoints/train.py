# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeePMD training entrypoint script.

Can handle local or distributed training.
"""

import json
import logging
import os
import time
from typing import (
    Any,
)

from deepmd.common import (
    j_loader,
)
from deepmd.jax.env import (
    jax,
    jax_export,
)
from deepmd.jax.train.trainer import (
    DPTrainer,
)
from deepmd.jax.utils.finetune import (
    get_finetune_rules,
)
from deepmd.jax.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.jax.utils.serialization import (
    serialize_from_file,
)
from deepmd.utils import random as dp_random
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data_system import (
    get_data,
)
from deepmd.utils.summary import SummaryPrinter as BaseSummaryPrinter

__all__ = ["train"]

log = logging.getLogger(__name__)


class SummaryPrinter(BaseSummaryPrinter):
    """Summary printer for JAX."""

    def is_built_with_cuda(self) -> bool:
        return jax_export.default_export_platform() == "cuda"

    def is_built_with_rocm(self) -> bool:
        return jax_export.default_export_platform() == "rocm"

    def get_compute_device(self) -> str:
        return jax.default_backend()

    def get_ngpus(self) -> int:
        return jax.device_count()

    def get_backend_info(self) -> dict:
        return {
            "Backend": "JAX",
            "JAX ver": jax.__version__,
        }

    def get_device_name(self) -> str:
        devices = jax.devices()
        if devices:
            return devices[0].device_kind
        return "Unknown"


def train(
    *,
    INPUT: str,
    init_model: str | None,
    restart: str | None,
    output: str,
    init_frz_model: str,
    mpi_log: str,
    log_level: int,
    log_path: str | None,
    skip_neighbor_stat: bool = False,
    finetune: str | None = None,
    use_pretrain_script: bool = False,
    force_load: bool = False,
    model_branch: str = "",
    **kwargs: Any,
) -> None:
    if int(os.environ.get("DP_JAX_MULTI_NPROC", "0")) > 1:
        multi_nproc = int(os.environ.get("DP_JAX_MULTI_NPROC", "0"))
        if multi_nproc <= 0:
            raise ValueError("DP_JAX_MULTI_NPROC is less than or equal to 0")
        multi_iproc = int(os.environ.get("DP_JAX_MULTI_IPROC", "-1"))
        if multi_iproc < 0:
            raise ValueError("DP_JAX_MULTI_IPROC is less than 0")
        multi_host = os.environ.get("DP_JAX_MULTI_HOST")
        if multi_host is None:
            raise ValueError("DP_JAX_MULTI_HOST is not given")
        jax.distributed.initialize(
            coordinator_address=multi_host,
            num_processes=multi_nproc,
            process_id=multi_iproc,
        )

    jdata = j_loader(INPUT)

    multi_task = "model_dict" in jdata["model"]
    shared_links = None
    if multi_task:
        jdata["model"], shared_links = preprocess_shared_params(jdata["model"])
        if "RANDOM" in jdata["model"]["model_dict"]:
            raise ValueError("Model name can not be 'RANDOM' in multi-task mode!")

    finetune_links = None
    finetune_data = None
    if finetune is not None:
        jdata["model"], finetune_links, finetune_data = get_finetune_rules(
            finetune,
            jdata["model"],
            model_branch=model_branch,
            change_model_params=use_pretrain_script,
        )
    if (init_model is not None or init_frz_model) and use_pretrain_script:
        source_model = init_model if init_model is not None else init_frz_model
        source_model_data = serialize_from_file(source_model)
        jdata["model"] = source_model_data["model_def_script"]

    jdata = update_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")
    jdata = normalize(jdata, multi_task=multi_task)
    jdata = update_sel(jdata, multi_task=multi_task)

    with open(output, "w") as fp:
        json.dump(jdata, fp, indent=4)
    SummaryPrinter()()

    model = DPTrainer(
        jdata,
        init_model=init_model,
        restart=restart,
        init_frz_model=init_frz_model or None,
        finetune_model=finetune,
        force_load=force_load,
        shared_links=shared_links,
        finetune_links=finetune_links,
        finetune_model_data=finetune_data,
    )

    seed = jdata["training"].get("seed", None)
    if seed is not None:
        seed += jax.process_index()
        seed = seed % (2**32)
    dp_random.seed(seed)

    if not multi_task:
        rcut = model.model.get_rcut()
        type_map = model.model.get_type_map()
        ipt_type_map = None if len(type_map) == 0 else type_map
        train_data = get_data(
            jdata["training"]["training_data"], rcut, ipt_type_map, None
        )
        train_data.add_data_requirements(model.data_requirements)
        train_data.print_summary("training")
        if jdata["training"].get("validation_data", None) is not None:
            valid_data = get_data(
                jdata["training"]["validation_data"],
                rcut,
                train_data.type_map,
                None,
            )
            valid_data.add_data_requirements(model.data_requirements)
            valid_data.print_summary("validation")
        else:
            valid_data = None
    else:
        train_data = {}
        valid_data = {}
        for model_key in model.model_keys:
            branch_model = model.model[model_key]
            rcut = branch_model.get_rcut()
            type_map = branch_model.get_type_map()
            ipt_type_map = None if len(type_map) == 0 else type_map
            branch_train = get_data(
                jdata["training"]["data_dict"][model_key]["training_data"],
                rcut,
                ipt_type_map,
                None,
            )
            branch_train.add_data_requirements(model.data_requirements[model_key])
            branch_train.print_summary(f"training in {model_key}")
            train_data[model_key] = branch_train
            if (
                jdata["training"]["data_dict"][model_key].get("validation_data", None)
                is not None
            ):
                branch_valid = get_data(
                    jdata["training"]["data_dict"][model_key]["validation_data"],
                    rcut,
                    branch_train.type_map,
                    None,
                )
                branch_valid.add_data_requirements(model.data_requirements[model_key])
                branch_valid.print_summary(f"validation in {model_key}")
                valid_data[model_key] = branch_valid
            else:
                valid_data[model_key] = None

    start_time = time.time()
    model.train(train_data, valid_data)
    end_time = time.time()
    log.info("finished training")
    log.info(f"wall time: {(end_time - start_time):.3f} s")


def update_sel(jdata: dict, *, multi_task: bool = False) -> dict:
    log.info(
        "Calculate neighbor statistics... (add --skip-neighbor-stat to skip this step)"
    )
    jdata_cpy = jdata.copy()
    if not multi_task:
        type_map = jdata["model"].get("type_map")
        train_data = get_data(
            jdata["training"]["training_data"],
            0,
            type_map,
            None,
        )
        del train_data
    else:
        for model_key in jdata["model"]["model_dict"]:
            type_map = jdata["model"]["model_dict"][model_key].get("type_map")
            train_data = get_data(
                jdata["training"]["data_dict"][model_key]["training_data"],
                0,
                type_map,
                None,
            )
            del train_data
    return jdata_cpy
