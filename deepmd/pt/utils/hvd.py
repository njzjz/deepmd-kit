# SPDX-License-Identifier: LGPL-3.0-or-later
try:
    import horovod.torch as hvd
except ImportError:
    # not installed, fallback to default behavior
    hvd = None
    local_rank = 0
    rank = 0
    size = 1
else:
    hvd.init()
    # local rank is the rank in a single node, for GPU configuration
    local_rank = hvd.local_rank()
    # rank is the global rank
    rank = hvd.rank()
    size = hvd.size()

__all__ = [
    "local_rank",
    "rank",
    "size",
    "hvd",
]
