# SPDX-License-Identifier: LGPL-3.0-or-later
import os

import numpy as np
import torch

from deepmd.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
    get_default_nthreads,
    set_default_nthreads,
)

SAMPLER_RECORD = os.environ.get("SAMPLER_RECORD", False)
try:
    # only linux
    ncpus = len(os.sched_getaffinity(0))
except AttributeError:
    ncpus = os.cpu_count()
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", min(8, ncpus)))
# Make sure DDP uses correct device if applicable
LOCAL_RANK = os.environ.get("LOCAL_RANK")
LOCAL_RANK = int(0 if LOCAL_RANK is None else LOCAL_RANK)

if torch.cuda.is_available():
    # One can use CUDA_VISIBLE_DEVICES to disable CUDA devices
    DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
elif torch.backends.mps.is_available() and GLOBAL_NP_FLOAT_PRECISION != np.float64:
    # float64 is not supported by mps
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

JIT = False
CACHE_PER_SYS = 5  # keep at most so many sets per sys in memory
ENERGY_BIAS_TRAINABLE = True

PRECISION_DICT = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "half": torch.float16,
    "single": torch.float32,
    "double": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}
GLOBAL_PT_FLOAT_PRECISION = PRECISION_DICT[np.dtype(GLOBAL_NP_FLOAT_PRECISION).name]
DEFAULT_PRECISION = "float64"

# throw warnings if threads not set
set_default_nthreads()
inter_nthreads, intra_nthreads = get_default_nthreads()
if inter_nthreads > 0:  # the behavior of 0 is not documented
    torch.set_num_interop_threads(inter_nthreads)
if intra_nthreads > 0:
    torch.set_num_threads(intra_nthreads)

__all__ = [
    "GLOBAL_ENER_FLOAT_PRECISION",
    "GLOBAL_NP_FLOAT_PRECISION",
    "GLOBAL_PT_FLOAT_PRECISION",
    "DEFAULT_PRECISION",
    "PRECISION_DICT",
    "SAMPLER_RECORD",
    "NUM_WORKERS",
    "DEVICE",
    "JIT",
    "CACHE_PER_SYS",
    "ENERGY_BIAS_TRAINABLE",
    "LOCAL_RANK",
]
