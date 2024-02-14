# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Iterator,
    Optional,
    Tuple,
)

import numpy as np

from deepmd.tf.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    GLOBAL_TF_FLOAT_PRECISION,
    default_tf_session_config,
    tf,
)
from deepmd.tf.utils.batch_size import (
    AutoBatchSize,
)
from deepmd.tf.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.tf.utils.nlist import (
    extend_coord_with_ghosts,
)
from deepmd.tf.utils.sess import (
    run_sess,
)
from deepmd.utils.neighbor_stat import NeighborStat as BaseNeighborStat

log = logging.getLogger(__name__)


class NeighborStatOP:
    """Class for getting neighbor statics data information.

    Parameters
    ----------
    ntypes
        The num of atom types
    rcut
        The cut-off radius
    distinguish_types : bool, optional
        If False, treat all types as a single type.
    """

    def __init__(
        self,
        ntypes: int,
        rcut: float,
        distinguish_types: bool,
    ) -> None:
        super().__init__()
        self.rcut = rcut
        self.ntypes = ntypes
        self.distinguish_types = distinguish_types

    def build(
        self,
        coord: tf.Tensor,
        atype: tf.Tensor,
        cell: tf.Tensor,
        pbc: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calculate the nearest neighbor distance between atoms, maximum nbor size of
        atoms and the output data range of the environment matrix.

        Parameters
        ----------
        coord
            The coordinates of atoms.
        atype
            The atom types.
        cell
            The cell.

        Returns
        -------
        tf.Tensor
            The minimal squared distance between two atoms, in the shape of (nframes,)
        tf.Tensor
            The maximal number of neighbors
        """
        nframes = tf.shape(coord)[0]
        coord = tf.reshape(coord, [nframes, -1, 3])
        nloc = tf.shape(coord)[1]
        coord = tf.reshape(coord, [nframes, nloc * 3])
        extend_coord, extend_atype, _ = extend_coord_with_ghosts(
            coord, atype, cell, self.rcut, pbc
        )

        coord1 = tf.reshape(extend_coord, [nframes, -1])
        nall = tf.shape(coord1)[1] // 3
        coord0 = coord1[:, : nloc * 3]
        diff = (
            tf.reshape(coord1, [nframes, -1, 3])[:, None, :, :]
            - tf.reshape(coord0, [nframes, -1, 3])[:, :, None, :]
        )
        # shape of diff: nframes, nloc, nall, 3
        # remove the diagonal elements
        mask = tf.eye(nloc, nall, dtype=tf.bool)
        # expand mask
        mask = tf.tile(mask[None, :, :], [nframes, 1, 1])
        # expand inf
        inf_mask = tf.constant(
            float("inf"), dtype=GLOBAL_TF_FLOAT_PRECISION, shape=[1, 1, 1]
        )
        inf_mask = tf.tile(inf_mask, [nframes, nloc, nall])
        rr2 = tf.reduce_sum(tf.square(diff), axis=-1)
        rr2 = tf.where(mask, inf_mask, rr2)
        min_rr2 = tf.reduce_min(rr2, axis=(1, 2))
        # count the number of neighbors
        if self.distinguish_types:
            mask = rr2 < self.rcut**2
            nnei = []
            for ii in range(self.ntypes):
                nnei.append(
                    tf.reduce_sum(
                        tf.cast(
                            mask & (tf.equal(extend_atype, ii))[:, None, :], tf.int32
                        ),
                        axis=-1,
                    )
                )
            # shape: nframes, nloc, ntypes
            nnei = tf.stack(nnei, axis=-1)
        else:
            mask = rr2 < self.rcut**2
            # virtual types (<0) are not counted
            nnei = tf.reshape(
                tf.reduce_sum(
                    tf.cast(
                        mask & tf.greater_equal(extend_atype, 0)[:, None, :], tf.int32
                    ),
                    axis=-1,
                ),
                [nframes, nloc, 1],
            )
        max_nnei = tf.reduce_max(nnei, axis=1)
        return min_rr2, max_nnei


class NeighborStat(BaseNeighborStat):
    """Class for getting training data information.

    It loads data from DeepmdData object, and measures the data info, including neareest nbor distance between atoms, max nbor size of atoms and the output data range of the environment matrix.

    Parameters
    ----------
    ntypes
            The num of atom types
    rcut
            The cut-off radius
    one_type : bool, optional, default=False
        Treat all types as a single type.
    """

    def __init__(
        self,
        ntypes: int,
        rcut: float,
        one_type: bool = False,
    ) -> None:
        """Constructor."""
        super().__init__(ntypes, rcut, one_type)
        self.auto_batch_size = AutoBatchSize()
        self.neighbor_stat = NeighborStatOP(ntypes, rcut, not one_type)
        self.place_holders = {}
        with tf.Graph().as_default() as sub_graph:
            self.op = self.build()
        self.sub_sess = tf.Session(graph=sub_graph, config=default_tf_session_config)

    def build(self):
        for ii in ["coord", "box"]:
            self.place_holders[ii] = tf.placeholder(
                GLOBAL_NP_FLOAT_PRECISION, [None, None], name="t_" + ii
            )
        self.place_holders["type"] = tf.placeholder(
            tf.int32, [None, None], name="t_type"
        )
        self.place_holders["pbc"] = tf.placeholder(tf.bool, [], name="t_pbc")
        ret = self.neighbor_stat.build(
            self.place_holders["coord"],
            self.place_holders["type"],
            self.place_holders["box"],
            self.place_holders["pbc"],
        )
        return ret

    def iterator(
        self, data: DeepmdDataSystem
    ) -> Iterator[Tuple[np.ndarray, float, str]]:
        """Produce data.

        Parameters
        ----------
        data
            The data system

        Yields
        ------
        np.ndarray
            The maximal number of neighbors
        float
            The squared minimal distance between two atoms
        str
            The directory of the data system
        """
        for ii in range(len(data.system_dirs)):
            for jj in data.data_systems[ii].dirs:
                data_set = data.data_systems[ii]
                data_set_data = data_set._load_set(jj)
                minrr2, max_nnei = self.auto_batch_size.execute_all(
                    self._execute,
                    data_set_data["coord"].shape[0],
                    data_set.get_natoms(),
                    data_set_data["coord"],
                    data_set_data["type"],
                    data_set_data["box"],
                    data_set.pbc,
                )
                yield np.max(max_nnei, axis=0), np.min(minrr2), jj

    def _execute(
        self,
        coord: np.ndarray,
        atype: np.ndarray,
        box: Optional[np.ndarray],
        pbc: bool,
    ):
        feed_dict = {
            self.place_holders["coord"]: coord,
            self.place_holders["type"]: atype,
            self.place_holders["box"]: box,
            self.place_holders["pbc"]: pbc,
        }
        minrr2, max_nnei = run_sess(self.sub_sess, self.op, feed_dict=feed_dict)
        return minrr2, max_nnei
