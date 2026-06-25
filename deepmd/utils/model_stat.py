# SPDX-License-Identifier: LGPL-3.0-or-later
from collections import (
    defaultdict,
)
from typing import (
    Any,
)

import numpy as np


def _get_batch_by_system(data: Any, sys_idx: int) -> dict[str, Any]:
    """Get one statistics batch from a concrete underlying system."""
    if not getattr(data, "mixed_systems", False):
        return data.get_batch(sys_idx=sys_idx)

    # DeepmdDataSystem.get_batch(sys_idx=...) ignores sys_idx for mixed
    # systems and returns a padded mixed batch. Statistics need to stay
    # grouped by original system so arrays with different natoms are not
    # concatenated before model/stat code sees them.
    stat_data = data.data_systems[sys_idx].get_batch(int(data.batch_size[sys_idx]))
    stat_data["natoms_vec"] = data.natoms_vec[sys_idx]
    stat_data["real_natoms_vec"] = np.tile(
        data.natoms_vec[sys_idx], (stat_data["type"].shape[0], 1)
    )
    stat_data["default_mesh"] = data.default_mesh[sys_idx]
    return stat_data


def _make_all_stat_ref(data: Any, nbatches: int) -> dict[str, list[Any]]:
    all_stat = defaultdict(list)
    for ii in range(data.get_nsystems()):
        for jj in range(nbatches):
            stat_data = _get_batch_by_system(data, ii)
            for dd in stat_data:
                if dd == "natoms_vec":
                    stat_data[dd] = stat_data[dd].astype(np.int32)
                all_stat[dd].append(stat_data[dd])
    return all_stat


def make_stat_input(
    data: Any, nbatches: int, merge_sys: bool = True
) -> dict[str, list[Any]]:
    """Pack data for statistics.

    Parameters
    ----------
    data
        The data
    nbatches : int
        The number of batches
    merge_sys : bool (True)
        Merge system data

    Returns
    -------
    all_stat:
        A dictionary of list of list storing data for stat.
        if merge_sys == False data can be accessed by
            all_stat[key][sys_idx][batch_idx][frame_idx]
        else merge_sys == True can be accessed by
            all_stat[key][batch_idx][frame_idx]
    """
    all_stat = defaultdict(list)
    for ii in range(data.get_nsystems()):
        sys_stat = defaultdict(list)
        for jj in range(nbatches):
            stat_data = _get_batch_by_system(data, ii)
            for dd in stat_data:
                if dd == "natoms_vec":
                    stat_data[dd] = stat_data[dd].astype(np.int32)
                sys_stat[dd].append(stat_data[dd])
        for dd in sys_stat:
            if merge_sys:
                for bb in sys_stat[dd]:
                    all_stat[dd].append(bb)
            else:
                all_stat[dd].append(sys_stat[dd])
    return all_stat


def merge_sys_stat(all_stat: dict[str, list[Any]]) -> dict[str, list[Any]]:
    first_key = next(iter(all_stat.keys()))
    nsys = len(all_stat[first_key])
    ret = defaultdict(list)
    for ii in range(nsys):
        for dd in all_stat:
            for bb in all_stat[dd][ii]:
                ret[dd].append(bb)
    return ret
