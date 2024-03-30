# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    abstractmethod,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np

from deepmd.common import (
    j_get_type,
)
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    tf,
)
from deepmd.tf.utils import (
    PluginVariant,
)
from deepmd.utils.plugin import (
    make_plugin_registry,
)


class Descriptor(PluginVariant, make_plugin_registry("descriptor")):
    r"""The abstract class for descriptors. All specific descriptors should
    be based on this class.

    The descriptor :math:`\mathcal{D}` describes the environment of an atom,
    which should be a function of coordinates and types of its neighbour atoms.

    Examples
    --------
    >>> descript = Descriptor(type="se_e2_a", rcut=6.0, rcut_smth=0.5, sel=[50])
    >>> type(descript)
    <class 'deepmd.tf.descriptor.se_a.DescrptSeA'>

    Notes
    -----
    Only methods and attributes defined in this class are generally public,
    that can be called by other classes.
    """

    def __new__(cls, *args, **kwargs):
        if cls is Descriptor:
            cls = cls.get_class_by_type(j_get_type(kwargs, cls.__name__))
        return super().__new__(cls)

    @abstractmethod
    def get_rcut(self) -> float:
        """Returns the cut-off radius.

        Returns
        -------
        float
            the cut-off radius

        Notes
        -----
        This method must be implemented, as it's called by other classes.
        """

    @abstractmethod
    def get_ntypes(self) -> int:
        """Returns the number of atom types.

        Returns
        -------
        int
            the number of atom types

        Notes
        -----
        This method must be implemented, as it's called by other classes.
        """

    @abstractmethod
    def get_dim_out(self) -> int:
        """Returns the output dimension of this descriptor.

        Returns
        -------
        int
            the output dimension of this descriptor

        Notes
        -----
        This method must be implemented, as it's called by other classes.
        """

    def get_dim_rot_mat_1(self) -> int:
        """Returns the first dimension of the rotation matrix. The rotation is of shape
        dim_1 x 3.

        Returns
        -------
        int
            the first dimension of the rotation matrix
        """
        raise NotImplementedError

    def get_nlist(self) -> Tuple[tf.Tensor, tf.Tensor, List[int], List[int]]:
        """Returns neighbor information.

        Returns
        -------
        nlist : tf.Tensor
            Neighbor list
        rij : tf.Tensor
            The relative distance between the neighbor and the center atom.
        sel_a : list[int]
            The number of neighbors with full information
        sel_r : list[int]
            The number of neighbors with only radial information
        """
        raise NotImplementedError

    @abstractmethod
    def compute_input_stats(
        self,
        data_coord: List[np.ndarray],
        data_box: List[np.ndarray],
        data_atype: List[np.ndarray],
        natoms_vec: List[np.ndarray],
        mesh: List[np.ndarray],
        input_dict: Dict[str, List[np.ndarray]],
        **kwargs,
    ) -> None:
        """Compute the statisitcs (avg and std) of the training data. The input will be
        normalized by the statistics.

        Parameters
        ----------
        data_coord : list[np.ndarray]
            The coordinates. Can be generated by
            :meth:`deepmd.tf.model.model_stat.make_stat_input`
        data_box : list[np.ndarray]
            The box. Can be generated by
            :meth:`deepmd.tf.model.model_stat.make_stat_input`
        data_atype : list[np.ndarray]
            The atom types. Can be generated by :meth:`deepmd.tf.model.model_stat.make_stat_input`
        natoms_vec : list[np.ndarray]
            The vector for the number of atoms of the system and different types of
            atoms. Can be generated by :meth:`deepmd.tf.model.model_stat.make_stat_input`
        mesh : list[np.ndarray]
            The mesh for neighbor searching. Can be generated by
            :meth:`deepmd.tf.model.model_stat.make_stat_input`
        input_dict : dict[str, list[np.ndarray]]
            Dictionary for additional input
        **kwargs
            Additional keyword arguments which may contain `mixed_type` and `real_natoms_vec`.

        Notes
        -----
        This method must be implemented, as it's called by other classes.
        """

    @abstractmethod
    def build(
        self,
        coord_: tf.Tensor,
        atype_: tf.Tensor,
        natoms: tf.Tensor,
        box_: tf.Tensor,
        mesh: tf.Tensor,
        input_dict: Dict[str, Any],
        reuse: Optional[bool] = None,
        suffix: str = "",
    ) -> tf.Tensor:
        """Build the computational graph for the descriptor.

        Parameters
        ----------
        coord_ : tf.Tensor
            The coordinate of atoms
        atype_ : tf.Tensor
            The type of atoms
        natoms : tf.Tensor
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        box_ : tf.Tensor
            The box of frames
        mesh : tf.Tensor
            For historical reasons, only the length of the Tensor matters.
            if size of mesh == 6, pbc is assumed.
            if size of mesh == 0, no-pbc is assumed.
        input_dict : dict[str, Any]
            Dictionary for additional inputs
        reuse : bool, optional
            The weights in the networks should be reused when get the variable.
        suffix : str, optional
            Name suffix to identify this descriptor

        Returns
        -------
        descriptor: tf.Tensor
            The output descriptor

        Notes
        -----
        This method must be implemented, as it's called by other classes.
        """

    def enable_compression(
        self,
        min_nbor_dist: float,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        table_extrapolate: float = 5.0,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
        suffix: str = "",
    ) -> None:
        """Reveive the statisitcs (distance, max_nbor_size and env_mat_range) of the
        training data.

        Parameters
        ----------
        min_nbor_dist : float
            The nearest distance between atoms
        graph : tf.Graph
            The graph of the model
        graph_def : tf.GraphDef
            The graph definition of the model
        table_extrapolate : float, default: 5.
            The scale of model extrapolation
        table_stride_1 : float, default: 0.01
            The uniform stride of the first table
        table_stride_2 : float, default: 0.1
            The uniform stride of the second table
        check_frequency : int, default: -1
            The overflow check frequency
        suffix : str, optional
            The suffix of the scope

        Notes
        -----
        This method is called by others when the descriptor supported compression.
        """
        raise NotImplementedError(
            "Descriptor %s doesn't support compression!" % type(self).__name__
        )

    def enable_mixed_precision(self, mixed_prec: Optional[dict] = None) -> None:
        """Reveive the mixed precision setting.

        Parameters
        ----------
        mixed_prec
            The mixed precision setting used in the embedding net

        Notes
        -----
        This method is called by others when the descriptor supported compression.
        """
        raise NotImplementedError(
            "Descriptor %s doesn't support mixed precision training!"
            % type(self).__name__
        )

    @abstractmethod
    def prod_force_virial(
        self, atom_ener: tf.Tensor, natoms: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute force and virial.

        Parameters
        ----------
        atom_ener : tf.Tensor
            The atomic energy
        natoms : tf.Tensor
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms

        Returns
        -------
        force : tf.Tensor
            The force on atoms
        virial : tf.Tensor
            The total virial
        atom_virial : tf.Tensor
            The atomic virial
        """

    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        suffix: str = "",
    ) -> None:
        """Init the embedding net variables with the given dict.

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str, optional
            The suffix of the scope

        Notes
        -----
        This method is called by others when the descriptor supported initialization from the given variables.
        """
        raise NotImplementedError(
            "Descriptor %s doesn't support initialization from the given variables!"
            % type(self).__name__
        )

    def get_tensor_names(self, suffix: str = "") -> Tuple[str]:
        """Get names of tensors.

        Parameters
        ----------
        suffix : str
            The suffix of the scope

        Returns
        -------
        Tuple[str]
            Names of tensors
        """
        raise NotImplementedError(
            "Descriptor %s doesn't support this property!" % type(self).__name__
        )

    def pass_tensors_from_frz_model(
        self,
        *tensors: tf.Tensor,
    ) -> None:
        """Pass the descrpt_reshape tensor as well as descrpt_deriv tensor from the frz graph_def.

        Parameters
        ----------
        *tensors : tf.Tensor
            passed tensors

        Notes
        -----
        The number of parameters in the method must be equal to the numbers of returns in
        :meth:`get_tensor_names`.
        """
        raise NotImplementedError(
            "Descriptor %s doesn't support this method!" % type(self).__name__
        )

    def build_type_exclude_mask(
        self,
        exclude_types: Set[Tuple[int, int]],
        ntypes: int,
        sel: List[int],
        ndescrpt: int,
        atype: tf.Tensor,
        shape0: tf.Tensor,
    ) -> tf.Tensor:
        r"""Build the type exclude mask for the descriptor.

        Notes
        -----
        To exclude the interaction between two types, the derivative of energy with
        respect to distances (or angles) between two atoms should be zero[1]_, i.e.

        .. math::
            \forall i \in \text{type 1}, j \in \text{type 2},
            \frac{\partial{E}}{\partial{r_{ij}}} = 0

        When embedding networks between every two types are built, we can just remove
        that network. But when `type_one_side` is enabled, a network may be built for
        multiple pairs of types. In this case, we need to build a mask to exclude the
        interaction between two types.

        The mask assumes the descriptors are sorted by neighbro type with the fixed
        number of given `sel` and each neighbor has the same number of descriptors
        (for example 4).

        Parameters
        ----------
        exclude_types : List[Tuple[int, int]]
            The list of excluded types, e.g. [(0, 1), (1, 0)] means the interaction
            between type 0 and type 1 is excluded.
        ntypes : int
            The number of types.
        sel : List[int]
            The list of the number of selected neighbors for each type.
        ndescrpt : int
            The number of descriptors for each atom.
        atype : tf.Tensor
            The type of atoms, with the size of shape0.
        shape0 : tf.Tensor
            The shape of the first dimension of the inputs, which is equal to
            nsamples * natoms.

        Returns
        -------
        tf.Tensor
            The type exclude mask, with the shape of (shape0, ndescrpt), and the
            precision of GLOBAL_TF_FLOAT_PRECISION. The mask has the value of 1 if the
            interaction between two types is not excluded, and 0 otherwise.

        References
        ----------
        .. [1] Jinzhe Zeng, Timothy J. Giese, ̧Sölen Ekesan, Darrin M. York,
           Development of Range-Corrected Deep Learning Potentials for Fast,
           Accurate Quantum Mechanical/molecular Mechanical Simulations of
           Chemical Reactions in Solution, J. Chem. Theory Comput., 2021,
           17 (11), 6993-7009.
        """
        # generate a mask
        type_mask = np.array(
            [
                [
                    1 if (tt_i, tt_j) not in exclude_types else 0
                    for tt_i in range(ntypes)
                ]
                for tt_j in range(ntypes)
            ],
            dtype=bool,
        )
        type_mask = tf.convert_to_tensor(type_mask, dtype=GLOBAL_TF_FLOAT_PRECISION)
        type_mask = tf.reshape(type_mask, [-1])

        # (nsamples * natoms, 1)
        atype_expand = tf.reshape(atype, [-1, 1])
        # (nsamples * natoms, ndescrpt)
        idx_i = tf.tile(atype_expand * ntypes, (1, ndescrpt))
        ndescrpt_per_neighbor = ndescrpt // np.sum(sel)
        # assume the number of neighbors for each type is the same
        assert ndescrpt_per_neighbor * np.sum(sel) == ndescrpt
        atype_descrpt = np.repeat(
            np.arange(ntypes), np.array(sel) * ndescrpt_per_neighbor
        )
        atype_descrpt = tf.convert_to_tensor(atype_descrpt, dtype=tf.int32)
        # (1, ndescrpt)
        atype_descrpt = tf.reshape(atype_descrpt, (1, ndescrpt))
        # (nsamples * natoms, ndescrpt)
        idx_j = tf.tile(atype_descrpt, (shape0, 1))
        # the index to mask (row index * ntypes + col index)
        idx = idx_i + idx_j
        idx = tf.reshape(idx, [-1])
        mask = tf.nn.embedding_lookup(type_mask, idx)
        # same as inputs_i, (nsamples * natoms, ndescrpt)
        mask = tf.reshape(mask, [-1, ndescrpt])
        return mask

    @property
    def explicit_ntypes(self) -> bool:
        """Explicit ntypes with type embedding."""
        return False

    @classmethod
    @abstractmethod
    def update_sel(cls, global_jdata: dict, local_jdata: dict):
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class
        """
        # call subprocess
        cls = cls.get_class_by_type(j_get_type(local_jdata, cls.__name__))
        return cls.update_sel(global_jdata, local_jdata)

    @classmethod
    def deserialize(cls, data: dict, suffix: str = "") -> "Descriptor":
        """Deserialize the model.

        There is no suffix in a native DP model, but it is important
        for the TF backend.

        Parameters
        ----------
        data : dict
            The serialized data
        suffix : str, optional
            Name suffix to identify this descriptor

        Returns
        -------
        Descriptor
            The deserialized descriptor
        """
        if cls is Descriptor:
            return Descriptor.get_class_by_type(
                j_get_type(data, cls.__name__)
            ).deserialize(data, suffix=suffix)
        raise NotImplementedError("Not implemented in class %s" % cls.__name__)

    def serialize(self, suffix: str = "") -> dict:
        """Serialize the model.

        There is no suffix in a native DP model, but it is important
        for the TF backend.

        Returns
        -------
        dict
            The serialized data
        suffix : str, optional
            Name suffix to identify this descriptor
        """
        raise NotImplementedError("Not implemented in class %s" % self.__name__)
