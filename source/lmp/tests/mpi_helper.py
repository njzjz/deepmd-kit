# SPDX-License-Identifier: LGPL-3.0-or-later
try:
    from mpi4py import (
        MPI,
        rc,
    )
except ModuleNotFoundError:
    MPI = None
    RANK = 0
else:
    RANK = MPI.COMM_WORLD.rank
    rc.finalize = True

__all__ = [
    "RANK",
]
