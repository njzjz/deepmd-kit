from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from deepmd.common import (
    get_activation_func,
    get_precision,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    GLOBAL_TF_FLOAT_PRECISION,
    default_tf_session_config,
    op_module,
    tf,
)
from deepmd.utils.graph import (
    get_type_embedding_net_variables_from_graph_def,
)
from deepmd.utils.network import (
    embedding_net,
    one_layer,
)


def embed_atom_type(
    ntypes: int,
    natoms: tf.Tensor,
    type_embedding: tf.Tensor,
):
    """
    Make the embedded type for the atoms in system.
    The atoms are assumed to be sorted according to the type,
    thus their types are described by a `tf.Tensor` natoms, see explanation below.

    Parameters
    ----------
    ntypes:
        Number of types.
    natoms:
        The number of atoms. This tensor has the length of Ntypes + 2
        natoms[0]: number of local atoms
        natoms[1]: total number of atoms held by this processor
        natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
    type_embedding:
        The type embedding.
        It has the shape of [ntypes, embedding_dim]

    Returns
    -------
    atom_embedding
        The embedded type of each atom.
        It has the shape of [numb_atoms, embedding_dim]
    """
    te_out_dim = type_embedding.get_shape().as_list()[-1]
    atype = []
    for ii in range(ntypes):
        atype.append(tf.tile([ii], [natoms[2 + ii]]))
    atype = tf.concat(atype, axis=0)
    atm_embed = tf.nn.embedding_lookup(
        type_embedding, tf.cast(atype, dtype=tf.int32)
    )  # (nf*natom)*nchnl
    atm_embed = tf.reshape(atm_embed, [-1, te_out_dim])
    return atm_embed


class TypeEmbedNet:
    """Type embedding network.

    Parameters
    ----------
    neuron : list[int]
            Number of neurons in each hidden layers of the embedding net
    resnet_dt
            Time-step `dt` in the resnet construction:
            y = x + dt * \\phi (Wx + b)
    activation_function
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    trainable
            If the weights of embedding net are trainable.
    seed
            Random seed for initializing the network parameters.
    uniform_seed
            Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    padding
            Concat the zero padding to the output, as the default embedding of empty type.
    """

    def __init__(
        self,
        neuron: List[int] = [],
        resnet_dt: bool = False,
        activation_function: Union[str, None] = "tanh",
        precision: str = "default",
        trainable: bool = True,
        seed: Optional[int] = None,
        uniform_seed: bool = False,
        padding: bool = False,
    ) -> None:
        """
        Constructor
        """
        self.neuron = neuron
        self.seed = seed
        self.filter_resnet_dt = resnet_dt
        self.filter_precision = get_precision(precision)
        self.filter_activation_fn = get_activation_func(activation_function)
        self.trainable = trainable
        self.uniform_seed = uniform_seed
        self.type_embedding_net_variables = None
        self.padding = padding

    def build(
        self,
        ntypes: int,
        reuse=None,
        suffix="",
    ):
        """
        Build the computational graph for the descriptor

        Parameters
        ----------
        ntypes
            Number of atom types.
        reuse
            The weights in the networks should be reused when get the variable.
        suffix
            Name suffix to identify this descriptor

        Returns
        -------
        embedded_types
            The computational graph for embedded types
        """
        types = tf.convert_to_tensor([ii for ii in range(ntypes)], dtype=tf.int32)
        ebd_type = tf.cast(
            tf.one_hot(tf.cast(types, dtype=tf.int32), int(ntypes)),
            self.filter_precision,
        )
        ebd_type = tf.reshape(ebd_type, [-1, ntypes])
        name = "type_embed_net" + suffix
        with tf.variable_scope(name, reuse=reuse):
            ebd_type = embedding_net(
                ebd_type,
                self.neuron,
                activation_fn=self.filter_activation_fn,
                precision=self.filter_precision,
                resnet_dt=self.filter_resnet_dt,
                seed=self.seed,
                trainable=self.trainable,
                initial_variables=self.type_embedding_net_variables,
                uniform_seed=self.uniform_seed,
            )
        ebd_type = tf.reshape(ebd_type, [-1, self.neuron[-1]])  # ntypes * neuron[-1]
        if self.padding:
            last_type = tf.cast(tf.zeros([1, self.neuron[-1]]), self.filter_precision)
            ebd_type = tf.concat([ebd_type, last_type], 0)  # (ntypes + 1) * neuron[-1]
        self.ebd_type = tf.identity(ebd_type, name="t_typeebd")
        return self.ebd_type

    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        suffix="",
    ) -> None:
        """
        Init the type embedding net variables with the given dict

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix
            Name suffix to identify this descriptor
        """
        self.type_embedding_net_variables = (
            get_type_embedding_net_variables_from_graph_def(graph_def, suffix=suffix)
        )


class MessagePassingEmbedNet:
    """Message Passing embedding network.

    Parameters
    ----------
    neuron : list[int]
            Number of neurons in each hidden layers of the embedding net
    resnet_dt
            Time-step `dt` in the resnet construction:
            y = x + dt * \\phi (Wx + b)
    activation_function
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    trainable
            If the weights of embedding net are trainable.
    seed
            Random seed for initializing the network parameters.
    uniform_seed
            Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    padding
            Concat the zero padding to the output, as the default embedding of empty type.
    """

    def __init__(
        self,
        neuron: List[int] = [],
        resnet_dt: bool = False,
        activation_function: Union[str, None] = "tanh",
        precision: str = "default",
        trainable: bool = True,
        seed: Optional[int] = None,
        uniform_seed: bool = False,
        padding: bool = False,
    ) -> None:
        self.neuron = neuron
        self.seed = seed
        self.filter_resnet_dt = resnet_dt
        self.filter_precision = get_precision(precision)
        self.filter_activation_fn = get_activation_func(activation_function)
        self.trainable = trainable
        self.uniform_seed = uniform_seed
        self.type_embedding_net_variables = None
        self.padding = padding

    def build(
        self,
        natoms: tf.Tensor,
        atype: tf.Tensor,
        nnei: int,
        nlist: tf.Tensor,
        ebd_type: tf.Tensor,
        env_mat: tf.Tensor,
        dim_env_mat: int,
        reuse=None,
        suffix="",
    ) -> tf.Tensor:
        """Build the computational graph for the descriptor

        Parameters
        ----------
        natoms : tf.Tensor
            The number of atoms
        atype : tf.Tensor
            The atom type of the atoms, nframes * natoms
        nnei : int
            The number of neighbors
        nlist : tf.Tensor
            The neighbor list, -1 if not exist
        ebd_type : tf.Tensor
            The type embedding
        env_mat : tf.Tensor
            The environemtal matrix, size is nframes * natoms * nnei * dim
        dim_env_mat : int
            The dimension of the environmental matrix
        reuse
            The weights in the networks should be reused when get the variable.
        suffix
            Name suffix to identify this network

        Returns
        -------
        embedded_types
            The computational graph for embedded types
        """
        nframes = tf.shape(atype)[0]
        natoms_loc = tf.shape(atype)[1]
        natoms_tot = nframes * natoms_loc
        ebd_type = tf.cast(ebd_type, self.filter_precision)

        # (nframes * natoms) * nnei * dim
        env_mat = tf.reshape(env_mat, [natoms_tot, nnei, dim_env_mat])
        env_mat = tf.cast(env_mat, self.filter_precision)
        # (nframes * natoms) * dim * dim
        # env_mat_mul = tf.matmul(env_mat, env_mat, transpose_b=True)

        # (nframes * natoms)
        atype_loc = tf.reshape(atype, [natoms_tot])
        nebd = ebd_type.get_shape().as_list()[1]
        # (nframes * natoms) * nebd
        ebd_type_loc = tf.nn.embedding_lookup(ebd_type, atype_loc)
        # (nframes * natoms) * nnei
        nlist = tf.reshape(nlist, [natoms_tot, nnei])

        # (nframes, 1)
        frame_idx = tf.range(nframes, dtype=tf.int32)
        frame_idx = tf.reshape(frame_idx, [nframes, 1])
        # (nframes, natoms * nnei)
        idx_i = tf.tile(frame_idx * (natoms_loc + 1), (1, natoms_loc * nnei))
        # (nframes * natoms), nnei
        idx_i = tf.reshape(idx_i, [natoms_tot, nnei])

        nlist += idx_i + 1

        for ii, nn in enumerate(self.neuron):
            # nframes, natoms_loc, nebd
            ebd_type_loc_padding = tf.reshape(ebd_type_loc, [nframes, natoms_loc, nebd])
            # concat zero padding
            ebd_type_loc_padding = tf.concat(
                [
                    tf.zeros([nframes, 1, nebd], dtype=self.filter_precision),
                    ebd_type_loc_padding,
                ],
                1,
            )
            # (nframes * (natoms + 1)) * nebd
            ebd_type_loc_padding = tf.reshape(
                ebd_type_loc_padding, [natoms_tot + nframes, nebd]
            )

            # (nframs * natoms) * nnei * nebd
            ebd_type_nei = tf.nn.embedding_lookup(ebd_type_loc_padding, nlist)
            # (nframes * natoms) * nnei * nebd
            ebd_type_nei = tf.reshape(ebd_type_nei, [natoms_tot, nnei, nebd])
            # (nframes * natoms) * nnei * nebd
            ebd_type_loc_reshape = tf.tile(
                tf.reshape(ebd_type_loc, [natoms_tot, 1, nebd]),
                [1, nnei, 1],
            )

            input_ebd_type_nei = tf.concat(
                [ebd_type_loc_reshape, ebd_type_nei, env_mat], axis=2
            )
            input_ebd_type_nei = tf.reshape(
                input_ebd_type_nei, [natoms_tot * nnei, nebd * 2 + dim_env_mat]
            )
            ebd_type_nei = one_layer(
                input_ebd_type_nei,
                nn,
                activation_fn=self.filter_activation_fn,
                precision=self.filter_precision,
                name=f"message_pass_nei_{ii}{suffix}",
                reuse=reuse,
                seed=self.seed,
                trainable=self.trainable,
            )
            ebd_type_nei = tf.reshape(ebd_type_nei, [natoms_tot, nnei, nn])

            # (nframes * natoms) * nebd * nebd
            ebd_type_nei_mat1 = tf.matmul(ebd_type_nei, env_mat, transpose_a=True)
            ebd_type_nei_mat1 /= nnei
            ebd_type_nei_mat = tf.matmul(
                ebd_type_nei_mat1, ebd_type_nei_mat1, transpose_b=True
            )
            ebd_type_nei_mat = tf.reshape(ebd_type_nei_mat, [natoms_tot, nn * nn])

            input = tf.concat([ebd_type_loc, ebd_type_nei_mat], 1)
            input = tf.reshape(input, [natoms_tot, nebd + nn * nn])
            old_ebd_type_loc = ebd_type_loc

            # (nframes * natoms) * nn
            ebd_type_loc = one_layer(
                input,
                nn,
                activation_fn=self.filter_activation_fn,
                precision=self.filter_precision,
                name=f"message_pass_layer_{ii}{suffix}",
                reuse=reuse,
                seed=self.seed,
                trainable=self.trainable,
            )
            shape = old_ebd_type_loc.get_shape().as_list()
            if nn == shape[1]:
                ebd_type_loc += old_ebd_type_loc
            nebd = nn
        # final neighbor ebd
        # nframes, natoms_loc, nebd
        ebd_type_loc_padding = tf.reshape(ebd_type_loc, [nframes, natoms_loc, nebd])
        # concat zero padding
        ebd_type_loc_padding = tf.concat(
            [
                tf.zeros([nframes, 1, nebd], dtype=self.filter_precision),
                ebd_type_loc_padding,
            ],
            1,
        )
        # (nframes * (natoms + 1)) * nebd
        ebd_type_loc_padding = tf.reshape(
            ebd_type_loc_padding, [natoms_tot + nframes, nebd]
        )

        # (nframs * natoms) * nnei * nebd
        ebd_type_nei = tf.nn.embedding_lookup(ebd_type_loc_padding, nlist)
        # (nframes * natoms) * nnei * nebd
        ebd_type_nei = tf.reshape(ebd_type_nei, [natoms_tot, nnei, nebd])

        ebd_type_nei = tf.cast(ebd_type_nei, GLOBAL_TF_FLOAT_PRECISION)
        self.ebd_type_nei = tf.identity(ebd_type_nei, name="t_envtypeebd_nei")
        ebd_type_loc = tf.cast(ebd_type_loc, GLOBAL_TF_FLOAT_PRECISION)
        self.ebd_type = tf.identity(ebd_type_loc, name="t_envtypeebd")
        return ebd_type_loc
