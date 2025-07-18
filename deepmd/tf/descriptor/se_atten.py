# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import re
import warnings
from typing import (
    Any,
    Optional,
    Union,
)

import numpy as np
from packaging.version import (
    Version,
)

from deepmd.dpmodel.utils.env_mat import (
    EnvMat,
)
from deepmd.dpmodel.utils.network import (
    EmbeddingNet,
    LayerNorm,
    NativeLayer,
    NetworkCollection,
)
from deepmd.tf.common import (
    cast_precision,
    get_np_precision,
)
from deepmd.tf.env import (
    ATTENTION_LAYER_PATTERN,
    GLOBAL_NP_FLOAT_PRECISION,
    GLOBAL_TF_FLOAT_PRECISION,
    TF_VERSION,
    default_tf_session_config,
    op_module,
    tf,
)
from deepmd.tf.nvnmd.descriptor.se_atten import (
    build_davg_dstd,
    build_op_descriptor,
    check_switch_range,
    descrpt2r4,
    filter_GR2D,
    filter_lower_R42GR,
)
from deepmd.tf.nvnmd.utils.config import (
    nvnmd_cfg,
)
from deepmd.tf.utils.compress import (
    get_extra_side_embedding_net_variable,
    get_two_side_type_embedding,
    make_data,
)
from deepmd.tf.utils.graph import (
    get_attention_layer_variables_from_graph_def,
    get_extra_embedding_net_suffix,
    get_extra_embedding_net_variables_from_graph_def,
    get_pattern_nodes_from_graph_def,
    get_tensor_by_name_from_graph,
)
from deepmd.tf.utils.network import (
    embedding_net,
    embedding_net_rand_seed_shift,
    layernorm,
    one_layer,
)
from deepmd.tf.utils.sess import (
    run_sess,
)
from deepmd.tf.utils.tabulate import (
    DPTabulate,
)
from deepmd.tf.utils.type_embed import (
    TypeEmbedNet,
)
from deepmd.tf.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .descriptor import (
    Descriptor,
)
from .se_a import (
    DescrptSeA,
)

log = logging.getLogger(__name__)


@Descriptor.register("dpa1")
@Descriptor.register("se_atten")
class DescrptSeAtten(DescrptSeA):
    r"""Smooth version descriptor with attention.

    Parameters
    ----------
    rcut: float
            The cut-off radius :math:`r_c`
    rcut_smth: float
            From where the environment matrix should be smoothed :math:`r_s`
    sel: list[int], int
            list[int]: sel[i] specifies the maxmum number of type i atoms in the cut-off radius
            int: the total maxmum number of atoms in the cut-off radius
    neuron: list[int]
            Number of neurons in each hidden layers of the embedding net :math:`\mathcal{N}`
    axis_neuron: int
            Number of the axis neuron :math:`M_2` (number of columns of the sub-matrix of the embedding matrix)
    resnet_dt: bool
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    trainable: bool
            If the weights of embedding net are trainable.
    seed: int, Optional
            Random seed for initializing the network parameters.
    type_one_side: bool
            If 'False', type embeddings of both neighbor and central atoms are considered.
            If 'True', only type embeddings of neighbor atoms are considered.
            Default is 'False'.
    exclude_types : list[list[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    set_davg_zero: bool
            Set the shift of embedding net input to zero.
    activation_function: str
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision: str
            The precision of the embedding net parameters. Supported options are |PRECISION|
    uniform_seed: bool
            Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    attn: int
            The length of hidden vector during scale-dot attention computation.
    attn_layer: int
            The number of layers in attention mechanism.
    attn_dotr: bool
            Whether to dot the relative coordinates on the attention weights as a gated scheme.
    attn_mask: bool
            Whether to mask the diagonal in the attention weights.
    ln_eps: float, Optional
            The epsilon value for layer normalization.
    tebd_input_mode: str
            The input mode of the type embedding. Supported modes are ["concat", "strip"].
            - "concat": Concatenate the type embedding with the smoothed radial information as the union input for the embedding network.
            - "strip": Use a separated embedding network for the type embedding and combine the output with the radial embedding network output.
            Default value will be `strip` in `se_atten_v2` descriptor.
    smooth_type_embedding: bool
            Whether to use smooth process in attention weights calculation.
            And when using stripped type embedding, whether to dot smooth factor on the network output of type embedding
            to keep the network smooth, instead of setting `set_davg_zero` to be True.
            Default value will be True in `se_atten_v2` descriptor.
    stripped_type_embedding: bool, Optional
            (Deprecated, kept only for compatibility.)
            Whether to strip the type embedding into a separate embedding network.
            Setting this parameter to `True` is equivalent to setting `tebd_input_mode` to 'strip'.
            Setting it to `False` is equivalent to setting `tebd_input_mode` to 'concat'.
            The default value is `None`, which means the `tebd_input_mode` setting will be used instead.
    type_map: list[str], Optional
            A list of strings. Give the name to each type of atoms.

    Raises
    ------
    ValueError
        if ntypes is 0.
    """

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: Union[list[int], int],
        ntypes: int,
        neuron: list[int] = [25, 50, 100],
        axis_neuron: int = 8,
        resnet_dt: bool = False,
        trainable: bool = True,
        seed: Optional[int] = None,
        type_one_side: bool = True,
        set_davg_zero: bool = True,
        exclude_types: list[list[int]] = [],
        activation_function: str = "tanh",
        precision: str = "default",
        uniform_seed: bool = False,
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        smooth_type_embedding: bool = False,
        tebd_input_mode: str = "concat",
        # not implemented
        scaling_factor=1.0,
        normalize=True,
        temperature=None,
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-3,
        concat_output_tebd: bool = True,
        env_protection: float = 0.0,  # not implement!!
        stripped_type_embedding: Optional[bool] = None,
        type_map: Optional[list[str]] = None,  # to be compat with input
        **kwargs,
    ) -> None:
        # Ensure compatibility with the deprecated stripped_type_embedding option.
        if stripped_type_embedding is None:
            stripped_type_embedding = tebd_input_mode == "strip"
        else:
            # Use the user-set stripped_type_embedding parameter first
            tebd_input_mode = "strip" if stripped_type_embedding else "concat"
        if not set_davg_zero and not (
            stripped_type_embedding and smooth_type_embedding
        ):
            warnings.warn(
                "Set 'set_davg_zero' False in descriptor 'se_atten' "
                "may cause unexpected incontinuity during model inference!"
            )
        if scaling_factor != 1.0:
            raise NotImplementedError("scaling_factor is not supported.")
        if not normalize:
            raise NotImplementedError("Disabling normalize is not supported.")
        if temperature is not None:
            raise NotImplementedError("temperature is not supported.")
        if not concat_output_tebd:
            raise NotImplementedError("Disbaling concat_output_tebd is not supported.")
        if env_protection != 0.0:
            raise NotImplementedError("env_protection != 0.0 is not supported.")
        #  to keep consistent with default value in this backends
        if ln_eps is None:
            ln_eps = 1e-3
        if isinstance(sel, list):
            sel = sum(sel)
        DescrptSeA.__init__(
            self,
            rcut,
            rcut_smth,
            [sel],
            neuron=neuron,
            axis_neuron=axis_neuron,
            resnet_dt=resnet_dt,
            trainable=trainable,
            seed=seed,
            type_one_side=type_one_side,
            exclude_types=exclude_types,
            set_davg_zero=set_davg_zero,
            activation_function=activation_function,
            precision=precision,
            uniform_seed=uniform_seed,
            type_map=type_map,
        )
        """
        Constructor
        """
        if not (nvnmd_cfg.enable and (nvnmd_cfg.version == 1)):
            assert Version(TF_VERSION) > Version("2"), (
                "se_atten only support tensorflow version 2.0 or higher."
            )
        if ntypes == 0:
            raise ValueError("`model/type_map` is not set or empty!")
        self.stripped_type_embedding = stripped_type_embedding
        self.tebd_input_mode = tebd_input_mode
        self.smooth = smooth_type_embedding
        self.trainable_ln = trainable_ln
        self.ln_eps = ln_eps
        self.ntypes = ntypes
        self.att_n = attn
        self.attn_layer = attn_layer
        self.attn_mask = attn_mask
        self.attn_dotr = attn_dotr
        self.filter_np_precision = get_np_precision(precision)
        self.two_side_embeeding_net_variables = None
        self.layer_size = len(neuron)

        # descrpt config
        self.sel_all_a = [sel]
        self.sel_all_r = [0]
        avg_zero = np.zeros([self.ntypes, self.ndescrpt]).astype(  # pylint: disable=no-explicit-dtype
            GLOBAL_NP_FLOAT_PRECISION
        )
        std_ones = np.ones([self.ntypes, self.ndescrpt]).astype(  # pylint: disable=no-explicit-dtype
            GLOBAL_NP_FLOAT_PRECISION
        )
        self.attention_layer_variables = None
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            name_pfx = "d_sea_"
            for ii in ["coord", "box"]:
                self.place_holders[ii] = tf.placeholder(
                    GLOBAL_NP_FLOAT_PRECISION, [None, None], name=name_pfx + "t_" + ii
                )
            self.place_holders["type"] = tf.placeholder(
                tf.int32, [None, None], name=name_pfx + "t_type"
            )
            self.place_holders["natoms_vec"] = tf.placeholder(
                tf.int32, [self.ntypes + 2], name=name_pfx + "t_natoms"
            )
            self.place_holders["default_mesh"] = tf.placeholder(
                tf.int32, [None], name=name_pfx + "t_mesh"
            )
            (
                self.stat_descrpt,
                self.descrpt_deriv_t,
                self.rij_t,
                self.nlist_t,
                self.nei_type_vec_t,
                self.nmask_t,
            ) = op_module.prod_env_mat_a_mix(
                self.place_holders["coord"],
                self.place_holders["type"],
                self.place_holders["natoms_vec"],
                self.place_holders["box"],
                self.place_holders["default_mesh"],
                tf.constant(avg_zero),
                tf.constant(std_ones),
                rcut_a=self.rcut_a,
                rcut_r=self.rcut_r,
                rcut_r_smth=self.rcut_r_smth,
                sel_a=self.sel_all_a,
                sel_r=self.sel_all_r,
            )
            if len(self.exclude_types):
                # exclude types applied to data stat
                mask = self.build_type_exclude_mask_mixed(
                    self.exclude_types,
                    self.ntypes,
                    self.sel_a,
                    self.ndescrpt,
                    # for data stat, nloc == nall
                    self.place_holders["type"],
                    tf.size(self.place_holders["type"]),
                    self.nei_type_vec_t,  # extra input for atten
                )
                self.stat_descrpt *= tf.reshape(mask, tf.shape(self.stat_descrpt))
        self.sub_sess = tf.Session(graph=sub_graph, config=default_tf_session_config)

    def compute_input_stats(
        self,
        data_coord: list,
        data_box: list,
        data_atype: list,
        natoms_vec: list,
        mesh: list,
        input_dict: dict,
        mixed_type: bool = False,
        real_natoms_vec: Optional[list] = None,
        **kwargs,
    ) -> None:
        """Compute the statisitcs (avg and std) of the training data. The input will be normalized by the statistics.

        Parameters
        ----------
        data_coord
            The coordinates. Can be generated by deepmd.tf.model.make_stat_input
        data_box
            The box. Can be generated by deepmd.tf.model.make_stat_input
        data_atype
            The atom types. Can be generated by deepmd.tf.model.make_stat_input
        natoms_vec
            The vector for the number of atoms of the system and different types of atoms.
            If mixed_type is True, this para is blank. See real_natoms_vec.
        mesh
            The mesh for neighbor searching. Can be generated by deepmd.tf.model.make_stat_input
        input_dict
            Dictionary for additional input
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.
        real_natoms_vec
            If mixed_type is True, it takes in the real natoms_vec for each frame.
        **kwargs
            Additional keyword arguments.
        """
        if True:
            sumr = []
            suma = []
            sumn = []
            sumr2 = []
            suma2 = []
            if mixed_type:
                sys_num = 0
                for cc, bb, tt, nn, mm, r_n in zip(
                    data_coord, data_box, data_atype, natoms_vec, mesh, real_natoms_vec
                ):
                    sysr, sysr2, sysa, sysa2, sysn = self._compute_dstats_sys_smth(
                        cc, bb, tt, nn, mm, mixed_type, r_n
                    )
                    sys_num += 1
                    sumr.append(sysr)
                    suma.append(sysa)
                    sumn.append(sysn)
                    sumr2.append(sysr2)
                    suma2.append(sysa2)
            else:
                for cc, bb, tt, nn, mm in zip(
                    data_coord, data_box, data_atype, natoms_vec, mesh
                ):
                    sysr, sysr2, sysa, sysa2, sysn = self._compute_dstats_sys_smth(
                        cc, bb, tt, nn, mm
                    )
                    sumr.append(sysr)
                    suma.append(sysa)
                    sumn.append(sysn)
                    sumr2.append(sysr2)
                    suma2.append(sysa2)
            stat_dict = {
                "sumr": sumr,
                "suma": suma,
                "sumn": sumn,
                "sumr2": sumr2,
                "suma2": suma2,
            }
            self.merge_input_stats(stat_dict)

    def enable_compression(
        self,
        min_nbor_dist: float,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
        suffix: str = "",
        tebd_suffix: str = "",
    ) -> None:
        """Receive the statisitcs (distance, max_nbor_size and env_mat_range) of the training data.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        graph : tf.Graph
            The graph of the model
        graph_def : tf.GraphDef
            The graph_def of the model
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        suffix : str, optional
            The suffix of the scope
        tebd_suffix : str, optional
            The suffix of the type embedding scope, only for DescrptDPA1Compat
        """
        # do some checks before the mocel compression process
        assert not self.filter_resnet_dt, (
            "Model compression error: descriptor resnet_dt must be false!"
        )
        for tt in self.exclude_types:
            if (tt[0] not in range(self.ntypes)) or (tt[1] not in range(self.ntypes)):
                raise RuntimeError(
                    "exclude types"
                    + str(tt)
                    + " must within the number of atomic types "
                    + str(self.ntypes)
                    + "!"
                )
        if self.ntypes * self.ntypes - len(self.exclude_types) == 0:
            raise RuntimeError(
                "empty embedding-net are not supported in model compression!"
            )

        if self.attn_layer != 0:
            raise RuntimeError("can not compress model when attention layer is not 0.")

        ret = get_pattern_nodes_from_graph_def(
            graph_def,
            f"filter_type_all{suffix}/.+{get_extra_embedding_net_suffix(type_one_side=False)}",
        )
        if len(ret) == 0:
            raise RuntimeError(
                f"can not find variables of embedding net `*{get_extra_embedding_net_suffix(type_one_side=False)}` from graph_def, maybe it is not a compressible model."
            )

        self.compress = True
        self.table = DPTabulate(
            self,
            self.filter_neuron,
            graph,
            graph_def,
            True,
            self.exclude_types,
            self.compress_activation_fn,
            suffix=suffix,
        )
        self.table_config = [
            table_extrapolate,
            table_stride_1,
            table_stride_2,
            check_frequency,
        ]
        self.lower, self.upper = self.table.build(
            min_nbor_dist, table_extrapolate, table_stride_1, table_stride_2
        )

        self.final_type_embedding = get_two_side_type_embedding(
            self, graph, suffix=tebd_suffix
        )
        type_side_suffix = get_extra_embedding_net_suffix(type_one_side=False)
        self.matrix = get_extra_side_embedding_net_variable(
            self, graph_def, type_side_suffix, "matrix", suffix
        )
        self.bias = get_extra_side_embedding_net_variable(
            self, graph_def, type_side_suffix, "bias", suffix
        )
        self.two_embd = make_data(self, self.final_type_embedding)

        self.davg = get_tensor_by_name_from_graph(graph, f"descrpt_attr{suffix}/t_avg")
        self.dstd = get_tensor_by_name_from_graph(graph, f"descrpt_attr{suffix}/t_std")

    def build(
        self,
        coord_: tf.Tensor,
        atype_: tf.Tensor,
        natoms: tf.Tensor,
        box_: tf.Tensor,
        mesh: tf.Tensor,
        input_dict: dict,
        reuse: Optional[bool] = None,
        suffix: str = "",
    ) -> tf.Tensor:
        """Build the computational graph for the descriptor.

        Parameters
        ----------
        coord_
            The coordinate of atoms
        atype_
            The type of atoms
        natoms
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        box_ : tf.Tensor
            The box of the system
        mesh
            For historical reasons, only the length of the Tensor matters.
            if size of mesh == 6, pbc is assumed.
            if size of mesh == 0, no-pbc is assumed.
        input_dict
            Dictionary for additional inputs
        reuse
            The weights in the networks should be reused when get the variable.
        suffix
            Name suffix to identify this descriptor

        Returns
        -------
        descriptor
            The output descriptor
        """
        davg = self.davg
        dstd = self.dstd
        if nvnmd_cfg.enable:
            nvnmd_cfg.set_ntype(self.ntypes)
            if nvnmd_cfg.restore_descriptor:
                davg, dstd = build_davg_dstd()
            check_switch_range(davg, dstd)
        with tf.variable_scope("descrpt_attr" + suffix, reuse=reuse):
            if davg is None:
                davg = np.zeros([self.ntypes, self.ndescrpt])  # pylint: disable=no-explicit-dtype
            if dstd is None:
                dstd = np.ones([self.ntypes, self.ndescrpt])  # pylint: disable=no-explicit-dtype
            t_rcut = tf.constant(
                np.max([self.rcut_r, self.rcut_a]),
                name="rcut",
                dtype=GLOBAL_TF_FLOAT_PRECISION,
            )
            t_ntypes = tf.constant(self.ntypes, name="ntypes", dtype=tf.int32)
            t_ndescrpt = tf.constant(self.ndescrpt, name="ndescrpt", dtype=tf.int32)
            t_sel = tf.constant(self.sel_a, name="sel", dtype=tf.int32)
            t_original_sel = tf.constant(
                self.original_sel if self.original_sel is not None else self.sel_a,
                name="original_sel",
                dtype=tf.int32,
            )
            self.t_avg = tf.get_variable(
                "t_avg",
                davg.shape,
                dtype=GLOBAL_TF_FLOAT_PRECISION,
                trainable=False,
                initializer=tf.constant_initializer(davg),
            )
            self.t_std = tf.get_variable(
                "t_std",
                dstd.shape,
                dtype=GLOBAL_TF_FLOAT_PRECISION,
                trainable=False,
                initializer=tf.constant_initializer(dstd),
            )

        with tf.control_dependencies([t_sel, t_original_sel]):
            coord = tf.reshape(coord_, [-1, natoms[1] * 3])
        box = tf.reshape(box_, [-1, 9])
        atype = tf.reshape(atype_, [-1, natoms[1]])
        self.attn_weight = [None for i in range(self.attn_layer)]
        self.angular_weight = [None for i in range(self.attn_layer)]
        self.attn_weight_final = [None for i in range(self.attn_layer)]

        op_descriptor = (
            build_op_descriptor() if nvnmd_cfg.enable else op_module.prod_env_mat_a_mix
        )
        (
            self.descrpt,
            self.descrpt_deriv,
            self.rij,
            self.nlist,
            self.nei_type_vec,
            self.nmask,
        ) = op_descriptor(
            coord,
            atype,
            natoms,
            box,
            mesh,
            self.t_avg,
            self.t_std,
            rcut_a=self.rcut_a,
            rcut_r=self.rcut_r,
            rcut_r_smth=self.rcut_r_smth,
            sel_a=self.sel_all_a,
            sel_r=self.sel_all_r,
        )

        self.nei_type_vec = tf.reshape(self.nei_type_vec, [-1])
        self.nmask = tf.cast(
            tf.reshape(self.nmask, [-1, 1, self.sel_all_a[0]]),
            self.filter_precision,
        )
        self.negative_mask = -(2 << 32) * (1.0 - self.nmask)
        # hard coding the magnitude of attention weight shift
        self.smth_attn_w_shift = 20.0
        # only used when tensorboard was set as true
        tf.summary.histogram("descrpt", self.descrpt)
        tf.summary.histogram("rij", self.rij)
        tf.summary.histogram("nlist", self.nlist)

        self.descrpt_reshape = tf.reshape(self.descrpt, [-1, self.ndescrpt])
        # prevent lookup error; the actual atype already used for nlist
        atype = tf.clip_by_value(atype, 0, self.ntypes - 1)
        self.atype_nloc = tf.reshape(
            tf.slice(atype, [0, 0], [-1, natoms[0]]), [-1]
        )  ## lammps will have error without this
        self._identity_tensors(suffix=suffix)
        if self.smooth:
            self.sliced_avg = tf.reshape(
                tf.slice(
                    tf.reshape(self.t_avg, [self.ntypes, -1, 4]), [0, 0, 0], [-1, 1, 1]
                ),
                [self.ntypes, 1],
            )
            self.sliced_std = tf.reshape(
                tf.slice(
                    tf.reshape(self.t_std, [self.ntypes, -1, 4]), [0, 0, 0], [-1, 1, 1]
                ),
                [self.ntypes, 1],
            )
            self.avg_looked_up = tf.reshape(
                tf.nn.embedding_lookup(self.sliced_avg, self.atype_nloc),
                [-1, natoms[0], 1],
            )
            self.std_looked_up = tf.reshape(
                tf.nn.embedding_lookup(self.sliced_std, self.atype_nloc),
                [-1, natoms[0], 1],
            )
            self.recovered_r = (
                tf.reshape(
                    tf.slice(
                        tf.reshape(self.descrpt_reshape, [-1, 4]), [0, 0], [-1, 1]
                    ),
                    [-1, natoms[0], self.sel_all_a[0]],
                )
                * self.std_looked_up
                + self.avg_looked_up
            )
            uu = 1 - self.rcut_r_smth * self.recovered_r
            self.recovered_switch = -uu * uu * uu + 1
            self.recovered_switch = tf.clip_by_value(self.recovered_switch, 0.0, 1.0)
            self.recovered_switch = tf.cast(
                self.recovered_switch, self.filter_precision
            )

        self.dout, self.qmat = self._pass_filter(
            self.descrpt_reshape,
            self.atype_nloc,
            natoms,
            input_dict,
            suffix=suffix,
            reuse=reuse,
            trainable=self.trainable,
        )

        # only used when tensorboard was set as true
        tf.summary.histogram("embedding_net_output", self.dout)
        return self.dout

    def _pass_filter(
        self, inputs, atype, natoms, input_dict, reuse=None, suffix="", trainable=True
    ):
        assert (
            input_dict is not None
            and input_dict.get("type_embedding", None) is not None
        ), "se_atten descriptor must use type_embedding"
        type_embedding = input_dict.get("type_embedding", None)
        inputs = tf.reshape(inputs, [-1, natoms[0], self.ndescrpt])
        output = []
        output_qmat = []
        inputs_i = inputs
        inputs_i = tf.reshape(inputs_i, [-1, self.ndescrpt])
        type_i = -1
        if len(self.exclude_types):
            mask = self.build_type_exclude_mask_mixed(
                self.exclude_types,
                self.ntypes,
                self.sel_a,
                self.ndescrpt,
                self.atype_nloc,  # when nloc != nall, pass nloc to mask
                tf.shape(inputs_i)[0],
                self.nei_type_vec,  # extra input for atten
            )
            #  (nframes * nloc * nnei, 1)
            nei_exclude_mask = tf.slice(
                tf.reshape(tf.cast(mask, self.filter_precision), [-1, 4]),
                [0, 0],
                [-1, 1],
            )
            if self.smooth:
                inputs_i = tf.where(
                    tf.cast(mask, tf.bool),
                    inputs_i,
                    # (nframes * nloc, 1) -> (nframes * nloc, ndescrpt)
                    tf.tile(
                        tf.reshape(self.avg_looked_up, [-1, 1]), [1, self.ndescrpt]
                    ),
                )
                #  (nframes, nloc, nnei)
                self.recovered_switch *= tf.reshape(
                    nei_exclude_mask,
                    [-1, natoms[0], self.sel_all_a[0]],
                )
            else:
                #  (nframes * nloc, 1,  nnei)
                self.nmask *= tf.reshape(
                    nei_exclude_mask,
                    [-1, 1, self.sel_all_a[0]],
                )
                self.negative_mask = -(2 << 32) * (1.0 - self.nmask)
                inputs_i *= mask
        if nvnmd_cfg.enable and nvnmd_cfg.quantize_descriptor:
            inputs_i = descrpt2r4(inputs_i, atype)
        layer, qmat = self._filter(
            inputs_i,
            type_i,
            natoms,
            name="filter_type_all" + suffix,
            suffix=suffix,
            reuse=reuse,
            trainable=trainable,
            activation_fn=self.filter_activation_fn,
            type_embedding=type_embedding,
            atype=atype,
        )
        layer = tf.reshape(
            layer,
            [
                tf.shape(inputs)[0],
                natoms[0],
                self.filter_neuron[-1] * self.n_axis_neuron,
            ],
        )
        qmat = tf.reshape(
            qmat, [tf.shape(inputs)[0], natoms[0], self.get_dim_rot_mat_1() * 3]
        )
        output.append(layer)
        output_qmat.append(qmat)
        output = tf.concat(output, axis=1)
        output_qmat = tf.concat(output_qmat, axis=1)
        return output, output_qmat

    def _compute_dstats_sys_smth(
        self,
        data_coord,
        data_box,
        data_atype,
        natoms_vec,
        mesh,
        mixed_type=False,
        real_natoms_vec=None,
    ):
        dd_all, descrpt_deriv_t, rij_t, nlist_t, nei_type_vec_t, nmask_t = run_sess(
            self.sub_sess,
            [
                self.stat_descrpt,
                self.descrpt_deriv_t,
                self.rij_t,
                self.nlist_t,
                self.nei_type_vec_t,
                self.nmask_t,
            ],
            feed_dict={
                self.place_holders["coord"]: data_coord,
                self.place_holders["type"]: data_atype,
                self.place_holders["natoms_vec"]: natoms_vec,
                self.place_holders["box"]: data_box,
                self.place_holders["default_mesh"]: mesh,
            },
        )
        if mixed_type:
            nframes = dd_all.shape[0]
            sysr = [0.0 for i in range(self.ntypes)]
            sysa = [0.0 for i in range(self.ntypes)]
            sysn = [0 for i in range(self.ntypes)]
            sysr2 = [0.0 for i in range(self.ntypes)]
            sysa2 = [0.0 for i in range(self.ntypes)]
            for ff in range(nframes):
                natoms = real_natoms_vec[ff]
                dd_ff = np.reshape(dd_all[ff], [-1, self.ndescrpt * natoms_vec[0]])
                start_index = 0
                for type_i in range(self.ntypes):
                    end_index = (
                        start_index + self.ndescrpt * natoms[2 + type_i]
                    )  # center atom split
                    dd = dd_ff[:, start_index:end_index]
                    dd = np.reshape(
                        dd, [-1, self.ndescrpt]
                    )  # nframes * typen_atoms , nnei * 4
                    start_index = end_index
                    # compute
                    dd = np.reshape(dd, [-1, 4])  # nframes * typen_atoms * nnei, 4
                    ddr = dd[:, :1]
                    dda = dd[:, 1:]
                    sumr = np.sum(ddr)
                    suma = np.sum(dda) / 3.0
                    sumn = dd.shape[0]
                    sumr2 = np.sum(np.multiply(ddr, ddr))
                    suma2 = np.sum(np.multiply(dda, dda)) / 3.0
                    sysr[type_i] += sumr
                    sysa[type_i] += suma
                    sysn[type_i] += sumn
                    sysr2[type_i] += sumr2
                    sysa2[type_i] += suma2
        else:
            natoms = natoms_vec
            dd_all = np.reshape(dd_all, [-1, self.ndescrpt * natoms[0]])
            start_index = 0
            sysr = []
            sysa = []
            sysn = []
            sysr2 = []
            sysa2 = []
            for type_i in range(self.ntypes):
                end_index = (
                    start_index + self.ndescrpt * natoms[2 + type_i]
                )  # center atom split
                dd = dd_all[:, start_index:end_index]
                dd = np.reshape(
                    dd, [-1, self.ndescrpt]
                )  # nframes * typen_atoms , nnei * 4
                start_index = end_index
                # compute
                dd = np.reshape(dd, [-1, 4])  # nframes * typen_atoms * nnei, 4
                ddr = dd[:, :1]
                dda = dd[:, 1:]
                sumr = np.sum(ddr)
                suma = np.sum(dda) / 3.0
                sumn = dd.shape[0]
                sumr2 = np.sum(np.multiply(ddr, ddr))
                suma2 = np.sum(np.multiply(dda, dda)) / 3.0
                sysr.append(sumr)
                sysa.append(suma)
                sysn.append(sumn)
                sysr2.append(sumr2)
                sysa2.append(suma2)
        return sysr, sysr2, sysa, sysa2, sysn

    def _lookup_type_embedding(
        self,
        xyz_scatter,
        natype,
        type_embedding,
    ):
        """Concatenate `type_embedding` of neighbors and `xyz_scatter`.
        If not self.type_one_side, concatenate `type_embedding` of center atoms as well.

        Parameters
        ----------
        xyz_scatter:
            shape is [nframes*natoms[0]*self.nnei, 1]
        natype:
            neighbor atom type
        type_embedding:
            shape is [self.ntypes, Y] where Y=jdata['type_embedding']['neuron'][-1]

        Returns
        -------
        embedding:
            environment of each atom represented by embedding.
        """
        te_out_dim = type_embedding.get_shape().as_list()[-1]
        self.test_type_embedding = type_embedding
        self.test_nei_embed = tf.nn.embedding_lookup(
            type_embedding, self.nei_type_vec
        )  # shape is [self.nnei, 1+te_out_dim]
        # nei_embed = tf.tile(nei_embed, (nframes * natoms[0], 1))  # shape is [nframes*natoms[0]*self.nnei, te_out_dim]
        nei_embed = tf.reshape(self.test_nei_embed, [-1, te_out_dim])
        self.embedding_input = tf.concat(
            [xyz_scatter, nei_embed], 1
        )  # shape is [nframes*natoms[0]*self.nnei, 1+te_out_dim]
        if not self.type_one_side:
            self.atm_embed = tf.nn.embedding_lookup(
                type_embedding, natype
            )  # shape is [nframes*natoms[0], te_out_dim]
            self.atm_embed = tf.tile(
                self.atm_embed, [1, self.nnei]
            )  # shape is [nframes*natoms[0], self.nnei*te_out_dim]
            self.atm_embed = tf.reshape(
                self.atm_embed, [-1, te_out_dim]
            )  # shape is [nframes*natoms[0]*self.nnei, te_out_dim]
            self.embedding_input_2 = tf.concat(
                [self.embedding_input, self.atm_embed], 1
            )  # shape is [nframes*natoms[0]*self.nnei, 1+te_out_dim+te_out_dim]
            return self.embedding_input_2
        return self.embedding_input

    def _scaled_dot_attn(
        self,
        Q,
        K,
        V,
        temperature,
        input_r,
        dotr=False,
        do_mask=False,
        layer=0,
        save_weights=True,
    ):
        attn = tf.matmul(Q / temperature, K, transpose_b=True)
        if self.smooth:
            # (nb x nloc) x nsel
            nsel = self.sel_all_a[0]
            attn = (attn + self.smth_attn_w_shift) * tf.reshape(
                self.recovered_switch, [-1, 1, nsel]
            ) * tf.reshape(
                self.recovered_switch, [-1, nsel, 1]
            ) - self.smth_attn_w_shift
        else:
            attn *= self.nmask
            attn += self.negative_mask
        attn = tf.nn.softmax(attn, axis=-1)
        if self.smooth:
            attn = (
                attn
                * tf.reshape(self.recovered_switch, [-1, 1, nsel])
                * tf.reshape(self.recovered_switch, [-1, nsel, 1])
            )
        else:
            attn *= tf.reshape(self.nmask, [-1, attn.shape[-1], 1])
        if save_weights:
            self.attn_weight[layer] = attn[0]  # atom 0
        if dotr:
            angular_weight = tf.matmul(input_r, input_r, transpose_b=True)  # normalized
            attn *= angular_weight
            if save_weights:
                self.angular_weight[layer] = angular_weight[0]  # atom 0
                self.attn_weight_final[layer] = attn[0]  # atom 0
        if do_mask:
            nei = int(attn.shape[-1])
            mask = tf.cast(tf.ones((nei, nei)) - tf.eye(nei), self.filter_precision)  # pylint: disable=no-explicit-dtype
            attn *= mask
        output = tf.matmul(attn, V)
        return output

    def _attention_layers(
        self,
        input_xyz,
        layer_num,
        shape_i,
        outputs_size,
        input_r,
        dotr=False,
        do_mask=False,
        trainable=True,
        suffix="",
    ):
        sd_k = tf.sqrt(tf.cast(1.0, dtype=self.filter_precision))
        for i in range(layer_num):
            name = f"attention_layer_{i}{suffix}"
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                # input_xyz_in = tf.nn.l2_normalize(input_xyz, -1)
                Q_c = one_layer(
                    input_xyz,
                    self.att_n,
                    name="c_query",
                    scope=name + "/",
                    reuse=tf.AUTO_REUSE,
                    seed=self.seed,
                    activation_fn=None,
                    precision=self.filter_precision,
                    trainable=trainable,
                    uniform_seed=self.uniform_seed,
                    initial_variables=self.attention_layer_variables,
                )
                if not self.uniform_seed and self.seed is not None:
                    self.seed += 1
                K_c = one_layer(
                    input_xyz,
                    self.att_n,
                    name="c_key",
                    scope=name + "/",
                    reuse=tf.AUTO_REUSE,
                    seed=self.seed,
                    activation_fn=None,
                    precision=self.filter_precision,
                    trainable=trainable,
                    uniform_seed=self.uniform_seed,
                    initial_variables=self.attention_layer_variables,
                )
                if not self.uniform_seed and self.seed is not None:
                    self.seed += 1
                V_c = one_layer(
                    input_xyz,
                    self.att_n,
                    name="c_value",
                    scope=name + "/",
                    reuse=tf.AUTO_REUSE,
                    seed=self.seed,
                    activation_fn=None,
                    precision=self.filter_precision,
                    trainable=trainable,
                    uniform_seed=self.uniform_seed,
                    initial_variables=self.attention_layer_variables,
                )
                if not self.uniform_seed and self.seed is not None:
                    self.seed += 1
                # # natom x nei_type_i x out_size
                # xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1] // 4, outputs_size[-1]))
                # natom x nei_type_i x att_n
                Q_c = tf.nn.l2_normalize(
                    tf.reshape(Q_c, (-1, shape_i[1] // 4, self.att_n)), -1
                )
                K_c = tf.nn.l2_normalize(
                    tf.reshape(K_c, (-1, shape_i[1] // 4, self.att_n)), -1
                )
                V_c = tf.nn.l2_normalize(
                    tf.reshape(V_c, (-1, shape_i[1] // 4, self.att_n)), -1
                )

                input_att = self._scaled_dot_attn(
                    Q_c, K_c, V_c, sd_k, input_r, dotr=dotr, do_mask=do_mask, layer=i
                )
                input_att = tf.reshape(input_att, (-1, self.att_n))

                # (natom x nei_type_i) x out_size
                input_xyz += one_layer(
                    input_att,
                    outputs_size[-1],
                    name="c_out",
                    scope=name + "/",
                    reuse=tf.AUTO_REUSE,
                    seed=self.seed,
                    activation_fn=None,
                    precision=self.filter_precision,
                    trainable=trainable,
                    uniform_seed=self.uniform_seed,
                    initial_variables=self.attention_layer_variables,
                )
                if not self.uniform_seed and self.seed is not None:
                    self.seed += 1
                input_xyz = layernorm(
                    input_xyz,
                    outputs_size[-1],
                    precision=self.filter_precision,
                    name="layer_normalization",
                    scope=name + "/",
                    reuse=tf.AUTO_REUSE,
                    seed=self.seed,
                    uniform_seed=self.uniform_seed,
                    trainable=self.trainable_ln,
                    eps=self.ln_eps,
                    initial_variables=self.attention_layer_variables,
                )
                if not self.uniform_seed and self.seed is not None:
                    self.seed += 1
        return input_xyz

    def _filter_lower(
        self,
        type_i,
        type_input,
        start_index,
        incrs_index,
        inputs,
        type_embedding=None,
        atype=None,
        is_exclude=False,
        activation_fn=None,
        bavg=0.0,
        stddev=1.0,
        trainable=True,
        suffix="",
        name="filter_",
        reuse=None,
    ):
        """Input env matrix, returns R.G."""
        outputs_size = [1, *self.filter_neuron]
        # cut-out inputs
        # with natom x (nei_type_i x 4)
        inputs_i = tf.slice(inputs, [0, start_index * 4], [-1, incrs_index * 4])
        shape_i = inputs_i.get_shape().as_list()
        natom = tf.shape(inputs_i)[0]
        # with (natom x nei_type_i) x 4
        inputs_reshape = tf.reshape(inputs_i, [-1, 4])
        # with (natom x nei_type_i) x 1
        xyz_scatter = tf.reshape(tf.slice(inputs_reshape, [0, 0], [-1, 1]), [-1, 1])
        assert atype is not None, "atype must exist!!"
        type_embedding = tf.cast(type_embedding, self.filter_precision)  # ntypes * Y
        # natom x 4 x outputs_size
        if not is_exclude:
            with tf.variable_scope(name, reuse=reuse):
                # with (natom x nei_type_i) x out_size
                if not self.stripped_type_embedding:
                    log.info("use the previous se_atten model")
                    xyz_scatter = self._lookup_type_embedding(
                        xyz_scatter, atype, type_embedding
                    )
                    xyz_scatter = embedding_net(
                        xyz_scatter,
                        self.filter_neuron,
                        self.filter_precision,
                        activation_fn=activation_fn,
                        resnet_dt=self.filter_resnet_dt,
                        name_suffix="",
                        stddev=stddev,
                        bavg=bavg,
                        seed=self.seed,
                        trainable=trainable,
                        uniform_seed=self.uniform_seed,
                        initial_variables=self.embedding_net_variables,
                        mixed_prec=self.mixed_prec,
                    )
                    if (not self.uniform_seed) and (self.seed is not None):
                        self.seed += self.seed_shift
                else:
                    if self.attn_layer == 0:
                        log.info(
                            "use the compressible model with stripped type embedding"
                        )
                    else:
                        log.info(
                            "use the non-compressible model with stripped type embedding"
                        )
                    if nvnmd_cfg.enable:
                        if nvnmd_cfg.quantize_descriptor:
                            return filter_lower_R42GR(
                                inputs_i,
                                atype,
                                self.nei_type_vec,
                            )
                        elif nvnmd_cfg.restore_descriptor:
                            self.embedding_net_variables = (
                                nvnmd_cfg.get_dp_init_weights()
                            )
                            self.two_side_embeeding_net_variables = (
                                nvnmd_cfg.get_dp_init_weights()
                            )
                    if not self.compress:
                        xyz_scatter = embedding_net(
                            xyz_scatter,
                            self.filter_neuron,
                            self.filter_precision,
                            activation_fn=activation_fn,
                            resnet_dt=self.filter_resnet_dt,
                            name_suffix="",
                            stddev=stddev,
                            bavg=bavg,
                            seed=self.seed,
                            trainable=trainable,
                            uniform_seed=self.uniform_seed,
                            initial_variables=self.embedding_net_variables,
                            mixed_prec=self.mixed_prec,
                        )
                        if (not self.uniform_seed) and (self.seed is not None):
                            self.seed += self.seed_shift
                    else:
                        net = "filter_net"
                        info = [
                            self.lower[net],
                            self.upper[net],
                            self.upper[net] * self.table_config[0],
                            self.table_config[1],
                            self.table_config[2],
                            self.table_config[3],
                        ]

                    padding_ntypes = type_embedding.shape[
                        0
                    ]  # this must be self.ntypes + 1
                    atype_expand = tf.reshape(atype, [-1, 1])
                    idx_i = tf.tile(atype_expand * padding_ntypes, [1, self.nnei])
                    idx_j = tf.reshape(self.nei_type_vec, [-1, self.nnei])
                    idx = idx_i + idx_j
                    index_of_two_side = tf.reshape(idx, [-1])

                    if self.compress:
                        two_embd = tf.nn.embedding_lookup(
                            self.two_embd, index_of_two_side
                        )
                    else:
                        type_embedding_nei = tf.tile(
                            tf.reshape(type_embedding, [1, padding_ntypes, -1]),
                            [padding_ntypes, 1, 1],
                        )  # (ntypes) * ntypes * Y
                        type_embedding_center = tf.tile(
                            tf.reshape(type_embedding, [padding_ntypes, 1, -1]),
                            [1, padding_ntypes, 1],
                        )  # ntypes * (ntypes) * Y
                        two_side_type_embedding = tf.concat(
                            [type_embedding_nei, type_embedding_center], -1
                        )  # ntypes * ntypes * (Y+Y)
                        two_side_type_embedding = tf.reshape(
                            two_side_type_embedding,
                            [-1, two_side_type_embedding.shape[-1]],
                        )
                        embedding_of_two_side_type_embedding = embedding_net(
                            two_side_type_embedding,
                            self.filter_neuron,
                            self.filter_precision,
                            activation_fn=activation_fn,
                            resnet_dt=self.filter_resnet_dt,
                            name_suffix=get_extra_embedding_net_suffix(
                                type_one_side=False
                            ),
                            stddev=stddev,
                            bavg=bavg,
                            seed=self.seed,
                            trainable=trainable,
                            uniform_seed=self.uniform_seed,
                            initial_variables=self.two_side_embeeding_net_variables,
                            mixed_prec=self.mixed_prec,
                        )
                        if (not self.uniform_seed) and (self.seed is not None):
                            self.seed += self.seed_shift
                        two_embd = tf.nn.embedding_lookup(
                            embedding_of_two_side_type_embedding, index_of_two_side
                        )
                    if self.smooth:
                        two_embd = two_embd * tf.reshape(self.recovered_switch, [-1, 1])
                    if not self.compress:
                        xyz_scatter = xyz_scatter * two_embd + xyz_scatter
                    else:
                        return op_module.tabulate_fusion_se_atten(
                            tf.cast(self.table.data[net], self.filter_precision),
                            info,
                            xyz_scatter,
                            tf.reshape(inputs_i, [natom, shape_i[1] // 4, 4]),
                            two_embd,
                            last_layer_size=outputs_size[-1],
                            is_sorted=len(self.exclude_types) == 0,
                        )

            input_r = tf.slice(
                tf.reshape(inputs_i, (-1, shape_i[1] // 4, 4)), [0, 0, 1], [-1, -1, 3]
            )
            input_r = tf.nn.l2_normalize(input_r, -1)
            # natom x nei_type_i x out_size
            xyz_scatter_att = tf.reshape(
                self._attention_layers(
                    xyz_scatter,
                    self.attn_layer,
                    shape_i,
                    outputs_size,
                    input_r,
                    dotr=self.attn_dotr,
                    do_mask=self.attn_mask,
                    trainable=trainable,
                    suffix=suffix,
                ),
                (-1, shape_i[1] // 4, outputs_size[-1]),
            )
            # xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1] // 4, outputs_size[-1]))
        else:
            raise RuntimeError("this should not be touched")
        # When using tf.reshape(inputs_i, [-1, shape_i[1]//4, 4]) below
        # [588 24] -> [588 6 4] correct
        # but if sel is zero
        # [588 0] -> [147 0 4] incorrect; the correct one is [588 0 4]
        # So we need to explicitly assign the shape to tf.shape(inputs_i)[0] instead of -1
        return tf.matmul(
            tf.reshape(inputs_i, [natom, shape_i[1] // 4, 4]),
            xyz_scatter_att,
            transpose_a=True,
        )

    @cast_precision
    def _filter(
        self,
        inputs,
        type_input,
        natoms,
        type_embedding=None,
        atype=None,
        activation_fn=tf.nn.tanh,
        stddev=1.0,
        bavg=0.0,
        suffix="",
        name="linear",
        reuse=None,
        trainable=True,
    ):
        nframes = tf.shape(tf.reshape(inputs, [-1, natoms[0], self.ndescrpt]))[0]
        # natom x (nei x 4)
        shape = inputs.get_shape().as_list()
        outputs_size = [1, *self.filter_neuron]
        outputs_size_2 = self.n_axis_neuron

        start_index = 0
        type_i = 0
        # natom x 4 x outputs_size
        xyz_scatter_1 = self._filter_lower(
            type_i,
            type_input,
            start_index,
            np.cumsum(self.sel_a)[-1],
            inputs,
            type_embedding=type_embedding,
            is_exclude=False,
            activation_fn=activation_fn,
            stddev=stddev,
            bavg=bavg,
            trainable=trainable,
            suffix=suffix,
            name=name,
            reuse=reuse,
            atype=atype,
        )
        if nvnmd_cfg.enable:
            return filter_GR2D(xyz_scatter_1)
        # natom x nei x outputs_size
        # xyz_scatter = tf.concat(xyz_scatter_total, axis=1)
        # natom x nei x 4
        # inputs_reshape = tf.reshape(inputs, [-1, shape[1]//4, 4])
        # natom x 4 x outputs_size
        # xyz_scatter_1 = tf.matmul(inputs_reshape, xyz_scatter, transpose_a = True)
        if self.original_sel is None:
            # shape[1] = nnei * 4
            nnei = shape[1] / 4
        else:
            nnei = tf.cast(
                tf.Variable(
                    np.sum(self.original_sel),
                    dtype=tf.int32,
                    trainable=False,
                    name="nnei",
                ),
                self.filter_precision,
            )
        xyz_scatter_1 = xyz_scatter_1 / nnei
        # natom x 4 x outputs_size_2
        xyz_scatter_2 = tf.slice(xyz_scatter_1, [0, 0, 0], [-1, -1, outputs_size_2])
        # # natom x 3 x outputs_size_2
        # qmat = tf.slice(xyz_scatter_2, [0,1,0], [-1, 3, -1])
        # natom x 3 x outputs_size_1
        qmat = tf.slice(xyz_scatter_1, [0, 1, 0], [-1, 3, -1])
        # natom x outputs_size_1 x 3
        qmat = tf.transpose(qmat, perm=[0, 2, 1])
        # natom x outputs_size x outputs_size_2
        result = tf.matmul(xyz_scatter_1, xyz_scatter_2, transpose_a=True)
        # natom x (outputs_size x outputs_size_2)
        result = tf.reshape(result, [-1, outputs_size_2 * outputs_size[-1]])

        return result, qmat

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
        """
        super().init_variables(graph=graph, graph_def=graph_def, suffix=suffix)

        self.attention_layer_variables = get_attention_layer_variables_from_graph_def(
            graph_def, suffix=suffix
        )

        def compat_ln_pattern(old_key) -> None:
            pattern = r"attention_layer_(\d+)/(layer_normalization)_\d+"
            replacement = r"attention_layer_\1/\2"
            if bool(re.search(pattern, old_key)):
                new_key = re.sub(pattern, replacement, old_key)
                v = self.attention_layer_variables.pop(old_key)
                self.attention_layer_variables[new_key] = v

        for item_key in list(self.attention_layer_variables.keys()):
            compat_ln_pattern(item_key)

        if self.stripped_type_embedding:
            self.two_side_embeeding_net_variables = (
                get_extra_embedding_net_variables_from_graph_def(
                    graph_def,
                    suffix,
                    get_extra_embedding_net_suffix(type_one_side=False),
                )
            )

    def build_type_exclude_mask_mixed(
        self,
        exclude_types: set[tuple[int, int]],
        ntypes: int,
        sel: list[int],
        ndescrpt: int,
        atype: tf.Tensor,
        shape0: tf.Tensor,
        nei_type_vec: tf.Tensor,
    ) -> tf.Tensor:
        r"""Build the type exclude mask for the attention descriptor.

        Notes
        -----
        This method has the similar way to build the type exclude mask as
        :meth:`deepmd.tf.descriptor.descriptor.Descriptor.build_type_exclude_mask`.
        The mathematical expression has been explained in that method.
        The difference is that the attention descriptor has provided the type of
        the neighbors (idx_j) that is not in order, so we use it from an extra
        input.

        Parameters
        ----------
        exclude_types : list[tuple[int, int]]
            The list of excluded types, e.g. [(0, 1), (1, 0)] means the interaction
            between type 0 and type 1 is excluded.
        ntypes : int
            The number of types.
        sel : list[int]
            The list of the number of selected neighbors for each type.
        ndescrpt : int
            The number of descriptors for each atom.
        atype : tf.Tensor
            The type of atoms, with the size of shape0.
        shape0 : tf.Tensor
            The shape of the first dimension of the inputs, which is equal to
            nsamples * natoms.
        nei_type_vec : tf.Tensor
            The type of neighbors, with the size of (shape0, nnei).

        Returns
        -------
        tf.Tensor
            The type exclude mask, with the shape of (shape0, ndescrpt), and the
            precision of GLOBAL_TF_FLOAT_PRECISION. The mask has the value of 1 if the
            interaction between two types is not excluded, and 0 otherwise.

        See Also
        --------
        deepmd.tf.descriptor.descriptor.Descriptor.build_type_exclude_mask
        """
        # generate a mask
        # op returns ntypes when the neighbor doesn't exist, so we need to add 1
        type_mask = np.array(
            [
                [
                    1 if (tt_i, tt_j) not in exclude_types else 0
                    for tt_i in range(ntypes + 1)
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
        idx_i = tf.tile(atype_expand * (ntypes + 1), (1, ndescrpt))
        # idx_j has been provided by atten op
        # (nsamples * natoms, nnei, 1)
        idx_j = tf.reshape(nei_type_vec, [shape0, sel[0], 1])
        # (nsamples * natoms, nnei, ndescrpt // nnei)
        idx_j = tf.tile(idx_j, (1, 1, ndescrpt // sel[0]))
        # (nsamples * natoms, ndescrpt)
        idx_j = tf.reshape(idx_j, [shape0, ndescrpt])
        idx = idx_i + idx_j
        idx = tf.reshape(idx, [-1])
        mask = tf.nn.embedding_lookup(type_mask, idx)
        # same as inputs_i, (nsamples * natoms, ndescrpt)
        mask = tf.reshape(mask, [-1, ndescrpt])
        return mask

    @property
    def explicit_ntypes(self) -> bool:
        """Explicit ntypes with type embedding."""
        return True

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: Optional[list[str]],
        local_jdata: dict,
    ) -> tuple[dict, Optional[float]]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statistics
        type_map : list[str], optional
            The name of each type of atoms
        local_jdata : dict
            The local data refer to the current class

        Returns
        -------
        dict
            The updated local data
        float
            The minimum distance between two atoms
        """
        local_jdata_cpy = local_jdata.copy()
        min_nbor_dist, sel = UpdateSel().update_one_sel(
            train_data, type_map, local_jdata_cpy["rcut"], local_jdata_cpy["sel"], True
        )
        local_jdata_cpy["sel"] = sel[0]
        return local_jdata_cpy, min_nbor_dist

    def serialize_attention_layers(
        self,
        nlayer: int,
        nnei: int,
        embed_dim: int,
        hidden_dim: int,
        dotr: bool,
        do_mask: bool,
        trainable_ln: bool,
        ln_eps: float,
        variables: dict,
        bias: bool = True,
        suffix: str = "",
    ) -> dict:
        data = {
            "layer_num": nlayer,
            "nnei": nnei,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "dotr": dotr,
            "do_mask": do_mask,
            "trainable_ln": trainable_ln,
            "ln_eps": ln_eps,
            "precision": self.precision.name,
            "attention_layers": [],
        }
        if suffix != "":
            attention_layer_pattern = (
                ATTENTION_LAYER_PATTERN.replace("/(c_query)", suffix + "/(c_query)")
                .replace("/(c_key)", suffix + "/(c_key)")
                .replace("/(c_value)", suffix + "/(c_value)")
                .replace("/(c_out)", suffix + "/(c_out)")
                .replace("/(layer_normalization)", suffix + "/(layer_normalization)")
            )
        else:
            attention_layer_pattern = ATTENTION_LAYER_PATTERN
        attention_layer_params = [{} for _ in range(nlayer)]
        for key, value in variables.items():
            m = re.search(attention_layer_pattern, key)
            m = [mm for mm in m.groups() if mm is not None]
            assert len(m) == 3
            if m[1] not in attention_layer_params[int(m[0])]:
                attention_layer_params[int(m[0])][m[1]] = {}
            attention_layer_params[int(m[0])][m[1]][m[2]] = value

        for layer_idx in range(nlayer):
            in_proj = NativeLayer(
                embed_dim,
                hidden_dim * 3,
                bias=bias,
                use_timestep=False,
                precision=self.precision.name,
                trainable=self.trainable,
            )
            matrix_list = [
                attention_layer_params[layer_idx][key]["matrix"]
                for key in ["c_query", "c_key", "c_value"]
            ]
            in_proj["matrix"] = np.concatenate(matrix_list, axis=-1)
            if bias:
                bias_list = [
                    attention_layer_params[layer_idx][key]["bias"]
                    for key in ["c_query", "c_key", "c_value"]
                ]
                in_proj["bias"] = np.concatenate(bias_list, axis=-1)
            out_proj = NativeLayer(
                hidden_dim,
                embed_dim,
                bias=bias,
                use_timestep=False,
                precision=self.precision.name,
                trainable=self.trainable,
            )
            out_proj["matrix"] = attention_layer_params[layer_idx]["c_out"]["matrix"]
            if bias:
                out_proj["bias"] = attention_layer_params[layer_idx]["c_out"]["bias"]

            layer_norm = LayerNorm(
                embed_dim,
                trainable=self.trainable_ln,
                eps=self.ln_eps,
                precision=self.precision.name,
            )
            layer_norm["matrix"] = attention_layer_params[layer_idx][
                "layer_normalization"
            ]["gamma"]
            layer_norm["bias"] = attention_layer_params[layer_idx][
                "layer_normalization"
            ]["beta"]
            data["attention_layers"].append(
                {
                    "attention_layer": {
                        "in_proj": in_proj.serialize(),
                        "out_proj": out_proj.serialize(),
                        "bias": bias,
                        "smooth": self.smooth,
                    },
                    "attn_layer_norm": layer_norm.serialize(),
                    "trainable_ln": self.trainable_ln,
                    "ln_eps": self.ln_eps,
                }
            )
        return data

    def serialize_network_strip(
        self,
        ntypes: int,
        ndim: int,
        in_dim: int,
        neuron: list[int],
        activation_function: str,
        resnet_dt: bool,
        variables: dict,
        suffix: str = "",
        type_one_side: bool = False,
        trainable: bool = True,
    ) -> dict:
        """Serialize network.

        Parameters
        ----------
        ntypes : int
            The number of types
        ndim : int
            The dimension of elements
        in_dim : int
            The input dimension
        neuron : list[int]
            The neuron list
        activation_function : str
            The activation function
        resnet_dt : bool
            Whether to use resnet
        variables : dict
            The input variables
        suffix : str, optional
            The suffix of the scope
        type_one_side : bool, optional
            If 'False', type embeddings of both neighbor and central atoms are considered.
            If 'True', only type embeddings of neighbor atoms are considered.
            Default is 'False'.
        trainable : bool
            Whether the network is trainable

        Returns
        -------
        dict
            The converted network data
        """
        assert ndim == 0, "only supports descriptors with type embedding!"
        embeddings = NetworkCollection(
            ntypes=ntypes,
            ndim=ndim,
            network_type="embedding_network",
        )
        name_suffix = get_extra_embedding_net_suffix(type_one_side=type_one_side)
        embedding_net_pattern_strip = str(
            rf"filter_type_(all)/(matrix)_(\d+){name_suffix}|"
            rf"filter_type_(all)/(bias)_(\d+){name_suffix}|"
            rf"filter_type_(all)/(idt)_(\d+){name_suffix}|"
        )[:-1]
        if suffix != "":
            embedding_net_pattern = (
                embedding_net_pattern_strip.replace("/(idt)", suffix + "/(idt)")
                .replace("/(bias)", suffix + "/(bias)")
                .replace("/(matrix)", suffix + "/(matrix)")
            )
        else:
            embedding_net_pattern = embedding_net_pattern_strip
        for key, value in variables.items():
            m = re.search(embedding_net_pattern, key)
            m = [mm for mm in m.groups() if mm is not None]
            layer_idx = int(m[2]) - 1
            weight_name = m[1]
            network_idx = ()
            if embeddings[network_idx] is None:
                # initialize the network if it is not initialized
                embeddings[network_idx] = EmbeddingNet(
                    in_dim=in_dim,
                    neuron=neuron,
                    activation_function=activation_function,
                    resnet_dt=resnet_dt,
                    precision=self.precision.name,
                    trainable=trainable,
                )
            assert embeddings[network_idx] is not None
            if weight_name == "idt":
                value = value.ravel()
            embeddings[network_idx][layer_idx][weight_name] = value
        return embeddings.serialize()

    @classmethod
    def deserialize_attention_layers(cls, data: dict, suffix: str = "") -> dict:
        """Deserialize attention layers.

        Parameters
        ----------
        data : dict
            The input attention layer data
        suffix : str, optional
            The suffix of the scope

        Returns
        -------
        variables : dict
            The input variables
        """
        attention_layer_variables = {}
        nlayer = data["layer_num"]
        hidden_dim = data["hidden_dim"]

        for layer_idx in range(nlayer):
            in_proj = NativeLayer.deserialize(
                data["attention_layers"][layer_idx]["attention_layer"]["in_proj"]
            )
            out_proj = NativeLayer.deserialize(
                data["attention_layers"][layer_idx]["attention_layer"]["out_proj"]
            )
            layer_norm = LayerNorm.deserialize(
                data["attention_layers"][layer_idx]["attn_layer_norm"]
            )

            # Deserialize in_proj
            c_query_matrix = in_proj["matrix"][:, :hidden_dim]
            c_key_matrix = in_proj["matrix"][:, hidden_dim : 2 * hidden_dim]
            c_value_matrix = in_proj["matrix"][:, 2 * hidden_dim :]
            attention_layer_variables[
                f"attention_layer_{layer_idx}{suffix}/c_query/matrix"
            ] = c_query_matrix
            attention_layer_variables[
                f"attention_layer_{layer_idx}{suffix}/c_key/matrix"
            ] = c_key_matrix
            attention_layer_variables[
                f"attention_layer_{layer_idx}{suffix}/c_value/matrix"
            ] = c_value_matrix
            if data["attention_layers"][layer_idx]["attention_layer"]["bias"]:
                c_query_bias = in_proj["bias"][:hidden_dim]
                c_key_bias = in_proj["bias"][hidden_dim : 2 * hidden_dim]
                c_value_bias = in_proj["bias"][2 * hidden_dim :]
                attention_layer_variables[
                    f"attention_layer_{layer_idx}{suffix}/c_query/bias"
                ] = c_query_bias
                attention_layer_variables[
                    f"attention_layer_{layer_idx}{suffix}/c_key/bias"
                ] = c_key_bias
                attention_layer_variables[
                    f"attention_layer_{layer_idx}{suffix}/c_value/bias"
                ] = c_value_bias

            # Deserialize out_proj
            attention_layer_variables[
                f"attention_layer_{layer_idx}{suffix}/c_out/matrix"
            ] = out_proj["matrix"]
            if data["attention_layers"][layer_idx]["attention_layer"]["bias"]:
                attention_layer_variables[
                    f"attention_layer_{layer_idx}{suffix}/c_out/bias"
                ] = out_proj["bias"]

            # Deserialize layer_norm
            attention_layer_variables[
                f"attention_layer_{layer_idx}{suffix}/layer_normalization/beta"
            ] = layer_norm["bias"]
            attention_layer_variables[
                f"attention_layer_{layer_idx}{suffix}/layer_normalization/gamma"
            ] = layer_norm["matrix"]
        return attention_layer_variables

    @classmethod
    def deserialize_network_strip(
        cls, data: dict, suffix: str = "", type_one_side: bool = False
    ) -> dict:
        """Deserialize network.

        Parameters
        ----------
        data : dict
            The input network data
        suffix : str, optional
            The suffix of the scope
        type_one_side : bool, optional
            If 'False', type embeddings of both neighbor and central atoms are considered.
            If 'True', only type embeddings of neighbor atoms are considered.
            Default is 'False'.

        Returns
        -------
        variables : dict
            The input variables
        """
        embedding_net_variables = {}
        embeddings = NetworkCollection.deserialize(data)
        assert embeddings.ndim == 0, "only supports descriptors with type embedding!"
        name_suffix = get_extra_embedding_net_suffix(type_one_side=type_one_side)
        net_idx = ()
        network = embeddings[net_idx]
        assert network is not None
        for layer_idx, layer in enumerate(network.layers):
            embedding_net_variables[
                f"filter_type_all{suffix}/matrix_{layer_idx + 1}{name_suffix}"
            ] = layer.w
            embedding_net_variables[
                f"filter_type_all{suffix}/bias_{layer_idx + 1}{name_suffix}"
            ] = layer.b
            if layer.idt is not None:
                embedding_net_variables[
                    f"filter_type_all{suffix}/idt_{layer_idx + 1}{name_suffix}"
                ] = layer.idt.reshape(1, -1)
            else:
                # prevent keyError
                embedding_net_variables[
                    f"filter_type_all{suffix}/idt_{layer_idx + 1}{name_suffix}"
                ] = 0.0
        return embedding_net_variables

    @classmethod
    def deserialize(cls, data: dict, suffix: str = ""):
        """Deserialize the model.

        Parameters
        ----------
        data : dict
            The serialized data

        Returns
        -------
        Model
            The deserialized model
        """
        if cls is not DescrptSeAtten:
            raise NotImplementedError(f"Not implemented in class {cls.__name__}")
        data = data.copy()
        if data["smooth_type_embedding"]:
            raise RuntimeError(
                "The implementation for smooth_type_embedding is inconsistent with other backends"
            )
        check_version_compatibility(data.pop("@version"), 2, 1)
        data.pop("@class")
        data.pop("type")
        embedding_net_variables = cls.deserialize_network(
            data.pop("embeddings"), suffix=suffix
        )
        attention_layer_variables = cls.deserialize_attention_layers(
            data.pop("attention_layers"), suffix=suffix
        )
        data.pop("env_mat")
        variables = data.pop("@variables")
        tebd_input_mode = data["tebd_input_mode"]
        type_embedding = TypeEmbedNet.deserialize(
            data.pop("type_embedding"), suffix=suffix
        )
        if "use_tebd_bias" not in data:
            # v1 compatibility
            data["use_tebd_bias"] = True
        type_embedding.use_tebd_bias = data.pop("use_tebd_bias")
        descriptor = cls(**data)
        descriptor.embedding_net_variables = embedding_net_variables
        descriptor.attention_layer_variables = attention_layer_variables
        descriptor.davg = variables["davg"].reshape(
            descriptor.ntypes, descriptor.ndescrpt
        )
        descriptor.dstd = variables["dstd"].reshape(
            descriptor.ntypes, descriptor.ndescrpt
        )
        descriptor.type_embedding = type_embedding
        if tebd_input_mode in ["strip"]:
            type_one_side = data["type_one_side"]
            two_side_embeeding_net_variables = cls.deserialize_network_strip(
                data.pop("embeddings_strip"),
                suffix=suffix,
                type_one_side=type_one_side,
            )
            descriptor.two_side_embeeding_net_variables = (
                two_side_embeeding_net_variables
            )
        return descriptor

    def serialize(self, suffix: str = "") -> dict:
        """Serialize the model.

        Parameters
        ----------
        suffix : str, optional
            The suffix of the scope

        Returns
        -------
        dict
            The serialized data
        """
        if self.smooth:
            raise RuntimeError(
                "The implementation for smooth_type_embedding is inconsistent with other backends"
            )
        # todo support serialization when tebd_input_mode=='strip' and type_one_side is True
        if self.stripped_type_embedding and self.type_one_side:
            raise NotImplementedError(
                "serialization is unsupported when tebd_input_mode=='strip' and type_one_side is True"
            )
        if (self.original_sel != self.sel_a).any():
            raise NotImplementedError(
                "Adjusting sel is unsupported by the native model"
            )
        if self.embedding_net_variables is None:
            raise RuntimeError("init_variables must be called before serialize")
        if self.spin is not None:
            raise NotImplementedError("spin is unsupported")
        assert self.davg is not None
        assert self.dstd is not None

        tebd_dim = self.type_embedding.neuron[0]
        if self.tebd_input_mode in ["concat"]:
            if not self.type_one_side:
                embd_input_dim = 1 + tebd_dim * 2
            else:
                embd_input_dim = 1 + tebd_dim
        else:
            embd_input_dim = 1
        data = {
            "@class": "Descriptor",
            "type": "dpa1",
            "@version": 2,
            "rcut": self.rcut_r,
            "rcut_smth": self.rcut_r_smth,
            "sel": self.sel_a,
            "ntypes": self.ntypes,
            "neuron": self.filter_neuron,
            "axis_neuron": self.n_axis_neuron,
            "set_davg_zero": self.set_davg_zero,
            "attn": self.att_n,
            "attn_layer": self.attn_layer,
            "attn_dotr": self.attn_dotr,
            "attn_mask": self.attn_mask,
            "activation_function": self.activation_function_name,
            "resnet_dt": self.filter_resnet_dt,
            "smooth_type_embedding": self.smooth,
            "tebd_input_mode": self.tebd_input_mode,
            "trainable_ln": self.trainable_ln,
            "ln_eps": self.ln_eps,
            "precision": self.filter_precision.name,
            "embeddings": self.serialize_network(
                ntypes=self.ntypes,
                ndim=0,
                in_dim=embd_input_dim,
                neuron=self.filter_neuron,
                activation_function=self.activation_function_name,
                resnet_dt=self.filter_resnet_dt,
                variables=self.embedding_net_variables,
                excluded_types=self.exclude_types,
                trainable=self.trainable,
                suffix=suffix,
            ),
            "attention_layers": self.serialize_attention_layers(
                nlayer=self.attn_layer,
                nnei=self.nnei_a,
                embed_dim=self.filter_neuron[-1],
                hidden_dim=self.att_n,
                dotr=self.attn_dotr,
                do_mask=self.attn_mask,
                trainable_ln=self.trainable_ln,
                ln_eps=self.ln_eps,
                variables=self.attention_layer_variables,
                suffix=suffix,
            ),
            "env_mat": EnvMat(self.rcut_r, self.rcut_r_smth).serialize(),
            "exclude_types": list(self.orig_exclude_types),
            "env_protection": self.env_protection,
            "@variables": {
                "davg": self.davg.reshape(self.ntypes, self.nnei_a, 4),
                "dstd": self.dstd.reshape(self.ntypes, self.nnei_a, 4),
            },
            "type_map": self.type_map,
            "trainable": self.trainable,
            "type_one_side": self.type_one_side,
            "spin": self.spin,
        }
        data["type_embedding"] = self.type_embedding.serialize(suffix=suffix)
        data["use_tebd_bias"] = self.type_embedding.use_tebd_bias
        data["tebd_dim"] = tebd_dim
        if len(self.type_embedding.neuron) > 1:
            raise NotImplementedError(
                "Only support single layer type embedding network"
            )
        if self.tebd_input_mode in ["strip"]:
            # assert (
            #     type(self) is not DescrptSeAtten
            # ), "only DescrptDPA1Compat and DescrptSeAttenV2 can serialize when tebd_input_mode=='strip'"
            data.update(
                {
                    "embeddings_strip": self.serialize_network_strip(
                        ntypes=self.ntypes,
                        ndim=0,
                        in_dim=2 * tebd_dim,
                        neuron=self.filter_neuron,
                        activation_function=self.activation_function_name,
                        resnet_dt=self.filter_resnet_dt,
                        variables=self.two_side_embeeding_net_variables,
                        suffix=suffix,
                        type_one_side=self.type_one_side,
                        trainable=self.trainable,
                    )
                }
            )
        # default values
        data.update(
            {
                "scaling_factor": 1.0,
                "normalize": True,
                "temperature": None,
                "concat_output_tebd": True,
                "use_econf_tebd": False,
            }
        )
        data["attention_layers"] = self.update_attention_layers_serialize(
            data["attention_layers"]
        )
        return data

    def update_attention_layers_serialize(self, data: dict):
        """Update the serialized data to be consistent with other backend references."""
        new_dict = {
            "@class": "NeighborGatedAttention",
            "@version": 1,
            "scaling_factor": 1.0,
            "normalize": True,
            "temperature": None,
        }
        new_dict.update(data)
        update_info = {
            "nnei": self.nnei_a,
            "embed_dim": self.filter_neuron[-1],
            "hidden_dim": self.att_n,
            "dotr": self.attn_dotr,
            "do_mask": self.attn_mask,
            "scaling_factor": 1.0,
            "normalize": True,
            "temperature": None,
            "precision": self.filter_precision.name,
        }
        for layer_idx in range(self.attn_layer):
            new_dict["attention_layers"][layer_idx].update(update_info)
            new_dict["attention_layers"][layer_idx]["attention_layer"].update(
                update_info
            )
            new_dict["attention_layers"][layer_idx]["attention_layer"].update(
                {
                    "num_heads": 1,
                }
            )
        return new_dict


class DescrptDPA1Compat(DescrptSeAtten):
    r"""Consistent version of the model for testing with other backend references.

    This model includes the type_embedding as attributes and other additional parameters.

    Parameters
    ----------
    rcut: float
            The cut-off radius :math:`r_c`
    rcut_smth: float
            From where the environment matrix should be smoothed :math:`r_s`
    sel: list[int], int
            list[int]: sel[i] specifies the maxmum number of type i atoms in the cut-off radius
            int: the total maxmum number of atoms in the cut-off radius
    ntypes: int
            Number of element types
    neuron: list[int]
            Number of neurons in each hidden layers of the embedding net :math:`\mathcal{N}`
    axis_neuron: int
            Number of the axis neuron :math:`M_2` (number of columns of the sub-matrix of the embedding matrix)
    tebd_dim: int
            Dimension of the type embedding
    tebd_input_mode: str
            The input mode of the type embedding. Supported modes are ["concat", "strip"].
            - "concat": Concatenate the type embedding with the smoothed radial information as the union input for the embedding network.
            - "strip": Use a separated embedding network for the type embedding and combine the output with the radial embedding network output.
    resnet_dt: bool
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    trainable: bool
            If the weights of this descriptors are trainable.
    trainable_ln: bool
            Whether to use trainable shift and scale weights in layer normalization.
    ln_eps: float, Optional
            The epsilon value for layer normalization.
    type_one_side: bool
            If 'False', type embeddings of both neighbor and central atoms are considered.
            If 'True', only type embeddings of neighbor atoms are considered.
            Default is 'False'.
    attn: int
            Hidden dimension of the attention vectors
    attn_layer: int
            Number of attention layers
    attn_dotr: bool
            If dot the angular gate to the attention weights
    attn_mask: bool
            (Only support False to keep consistent with other backend references.)
            If mask the diagonal of attention weights
    exclude_types : list[list[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    env_protection: float
            Protection parameter to prevent division by zero errors during environment matrix calculations.
    set_davg_zero: bool
            Set the shift of embedding net input to zero.
    activation_function: str
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision: str
            The precision of the embedding net parameters. Supported options are |PRECISION|
    scaling_factor: float
            (Only to keep consistent with other backend references.)
            (Not used in this version.)
            The scaling factor of normalization in calculations of attention weights.
            If `temperature` is None, the scaling of attention weights is (N_dim * scaling_factor)**0.5
    normalize: bool
            (Only support True to keep consistent with other backend references.)
            (Not used in this version.)
            Whether to normalize the hidden vectors in attention weights calculation.
    temperature: float
            (Only support 1.0 to keep consistent with other backend references.)
            (Not used in this version.)
            If not None, the scaling of attention weights is `temperature` itself.
    smooth_type_embedding: bool
            (Only support False to keep consistent with other backend references.)
            Whether to use smooth process in attention weights calculation.
    concat_output_tebd: bool
            Whether to concat type embedding at the output of the descriptor.
    use_econf_tebd: bool, Optional
            Whether to use electronic configuration type embedding.
    use_tebd_bias : bool, Optional
            Whether to use bias in the type embedding layer.
    type_map: list[str], Optional
            A list of strings. Give the name to each type of atoms.
    spin
            (Only support None to keep consistent with old implementation.)
            The old implementation of deepspin.
    """

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: Union[list[int], int],
        ntypes: int,
        neuron: list[int] = [25, 50, 100],
        axis_neuron: int = 8,
        tebd_dim: int = 8,
        tebd_input_mode: str = "concat",
        resnet_dt: bool = False,
        trainable: bool = True,
        type_one_side: bool = True,
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        exclude_types: list[list[int]] = [],
        env_protection: float = 0.0,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "default",
        scaling_factor=1.0,
        normalize: bool = True,
        temperature: Optional[float] = None,
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-3,
        smooth_type_embedding: bool = True,
        concat_output_tebd: bool = True,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        type_map: Optional[list[str]] = None,
        spin: Optional[Any] = None,
        # consistent with argcheck, not used though
        seed: Optional[int] = None,
        uniform_seed: bool = False,
    ) -> None:
        if not normalize:
            raise NotImplementedError("Only support normalize == True in this version.")
        if temperature != 1.0:
            raise NotImplementedError(
                "Only support temperature == 1.0 in this version."
            )
        if spin is not None:
            raise NotImplementedError("Only support spin is None in this version.")
        if attn_mask:
            raise NotImplementedError(
                "old implementation of attn_mask is not supported."
            )
        #  to keep consistent with default value in this backends
        if ln_eps is None:
            ln_eps = 1e-3

        super().__init__(
            rcut,
            rcut_smth,
            sel,
            ntypes,
            neuron=neuron,
            axis_neuron=axis_neuron,
            resnet_dt=resnet_dt,
            trainable=trainable,
            seed=seed,
            type_one_side=type_one_side,
            set_davg_zero=set_davg_zero,
            exclude_types=exclude_types,
            activation_function=activation_function,
            precision=precision,
            uniform_seed=uniform_seed,
            attn=attn,
            attn_layer=attn_layer,
            attn_dotr=attn_dotr,
            attn_mask=attn_mask,
            trainable_ln=trainable_ln,
            ln_eps=ln_eps,
            smooth_type_embedding=smooth_type_embedding,
            tebd_input_mode=tebd_input_mode,
            env_protection=env_protection,
            type_map=type_map,
        )
        self.tebd_dim = tebd_dim
        self.use_econf_tebd = use_econf_tebd
        self.use_tebd_bias = use_tebd_bias
        self.scaling_factor = scaling_factor
        self.normalize = normalize
        self.temperature = temperature
        self.type_embedding = TypeEmbedNet(
            ntypes=self.ntypes,
            neuron=[self.tebd_dim],
            padding=True,
            activation_function="Linear",
            use_econf_tebd=use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
            type_map=type_map,
            # precision=precision,
            seed=seed,
        )
        self.concat_output_tebd = concat_output_tebd
        if self.tebd_input_mode in ["concat"]:
            if not self.type_one_side:
                self.embd_input_dim = 1 + self.tebd_dim * 2
            else:
                self.embd_input_dim = 1 + self.tebd_dim
        else:
            self.embd_input_dim = 1

    def get_dim_out(self) -> int:
        """Returns the output dimension of this descriptor."""
        return (
            super().get_dim_out() + self.tebd_dim
            if self.concat_output_tebd
            else super().get_dim_out()
        )

    def build(
        self,
        coord_: tf.Tensor,
        atype_: tf.Tensor,
        natoms: tf.Tensor,
        box_: tf.Tensor,
        mesh: tf.Tensor,
        input_dict: dict,
        reuse: Optional[bool] = None,
        suffix: str = "",
    ) -> tf.Tensor:
        type_embedding = self.type_embedding.build(self.ntypes, suffix=suffix)
        if (not self.uniform_seed) and (self.seed is not None):
            self.seed += embedding_net_rand_seed_shift([self.tebd_dim])
        input_dict["type_embedding"] = type_embedding

        # nf x nloc x out_dim
        self.dout = super().build(
            coord_,
            atype_,
            natoms,
            box_,
            mesh,
            input_dict,
            reuse=reuse,
            suffix=suffix,
        )
        # self.dout = tf.cast(self.dout, self.filter_precision)
        if self.concat_output_tebd:
            atype = tf.reshape(atype_, [-1, natoms[1]])
            atype_nloc = tf.reshape(
                tf.slice(atype, [0, 0], [-1, natoms[0]]), [-1]
            )  ## lammps will have error without this
            atom_embed = tf.reshape(
                tf.nn.embedding_lookup(type_embedding, atype_nloc),
                [-1, natoms[0], self.tebd_dim],
            )
            atom_embed = tf.cast(atom_embed, GLOBAL_TF_FLOAT_PRECISION)
            # nf x nloc x (out_dim + tebd_dim)
            self.dout = tf.concat([self.dout, atom_embed], axis=-1)
        return self.dout

    def enable_compression(
        self,
        min_nbor_dist: float,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
        suffix: str = "",
        tebd_suffix: str = "",
    ) -> None:
        """Reveive the statisitcs (distance, max_nbor_size and env_mat_range) of the training data.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        graph : tf.Graph
            The graph of the model
        graph_def : tf.GraphDef
            The graph_def of the model
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        suffix : str, optional
            The suffix of the scope
        tebd_suffix : str, optional
            Same as suffix.
        """
        assert tebd_suffix == "", (
            "DescrptDPA1Compat must use the same tebd_suffix as suffix!"
        )
        super().enable_compression(
            min_nbor_dist,
            graph,
            graph_def,
            table_extrapolate=table_extrapolate,
            table_stride_1=table_stride_1,
            table_stride_2=table_stride_2,
            check_frequency=check_frequency,
            suffix=suffix,
            tebd_suffix=suffix,
        )

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
        """
        super().init_variables(graph=graph, graph_def=graph_def, suffix=suffix)
        self.type_embedding.init_variables(
            graph=graph, graph_def=graph_def, suffix=suffix
        )

    def update_attention_layers_serialize(self, data: dict):
        """Update the serialized data to be consistent with other backend references."""
        new_dict = {
            "@class": "NeighborGatedAttention",
            "@version": 1,
            "scaling_factor": self.scaling_factor,
            "normalize": self.normalize,
            "temperature": self.temperature,
        }
        new_dict.update(data)
        update_info = {
            "nnei": self.nnei_a,
            "embed_dim": self.filter_neuron[-1],
            "hidden_dim": self.att_n,
            "dotr": self.attn_dotr,
            "do_mask": self.attn_mask,
            "scaling_factor": self.scaling_factor,
            "normalize": self.normalize,
            "temperature": self.temperature,
            "precision": self.filter_precision.name,
        }
        for layer_idx in range(self.attn_layer):
            new_dict["attention_layers"][layer_idx].update(update_info)
            new_dict["attention_layers"][layer_idx]["attention_layer"].update(
                update_info
            )
            new_dict["attention_layers"][layer_idx]["attention_layer"].update(
                {
                    "num_heads": 1,
                }
            )
        return new_dict

    @classmethod
    def deserialize(cls, data: dict, suffix: str = ""):
        """Deserialize the model.

        Parameters
        ----------
        data : dict
            The serialized data

        Returns
        -------
        Model
            The deserialized model
        """
        if cls is not DescrptDPA1Compat:
            raise NotImplementedError(f"Not implemented in class {cls.__name__}")
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 2, 1)
        data.pop("@class")
        data.pop("type")
        embedding_net_variables = cls.deserialize_network(
            data.pop("embeddings"), suffix=suffix
        )
        attention_layer_variables = cls.deserialize_attention_layers(
            data.pop("attention_layers"), suffix=suffix
        )
        data.pop("env_mat")
        variables = data.pop("@variables")
        type_embedding = data.pop("type_embedding")
        tebd_input_mode = data["tebd_input_mode"]
        type_one_side = data["type_one_side"]
        if tebd_input_mode in ["strip"]:
            two_side_embeeding_net_variables = cls.deserialize_network_strip(
                data.pop("embeddings_strip"),
                suffix=suffix,
                type_one_side=type_one_side,
            )
        else:
            two_side_embeeding_net_variables = None
        # compat with version 1
        if "use_tebd_bias" not in data:
            data["use_tebd_bias"] = True
        descriptor = cls(**data)
        descriptor.embedding_net_variables = embedding_net_variables
        descriptor.attention_layer_variables = attention_layer_variables
        descriptor.two_side_embeeding_net_variables = two_side_embeeding_net_variables
        descriptor.davg = variables["davg"].reshape(
            descriptor.ntypes, descriptor.ndescrpt
        )
        descriptor.dstd = variables["dstd"].reshape(
            descriptor.ntypes, descriptor.ndescrpt
        )
        descriptor.type_embedding = TypeEmbedNet.deserialize(
            type_embedding, suffix=suffix
        )
        return descriptor

    def serialize(self, suffix: str = "") -> dict:
        """Serialize the model.

        Parameters
        ----------
        suffix : str, optional
            The suffix of the scope

        Returns
        -------
        dict
            The serialized data
        """
        data = super().serialize(suffix)
        data.update(
            {
                "type": "dpa1",
                "@version": 2,
                "scaling_factor": self.scaling_factor,
                "normalize": self.normalize,
                "temperature": self.temperature,
                "concat_output_tebd": self.concat_output_tebd,
                "use_econf_tebd": self.use_econf_tebd,
            }
        )
        return data
