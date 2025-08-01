# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import numpy as np

from deepmd.tf.common import (
    cast_precision,
    get_activation_func,
    get_precision,
)
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.fit.fitting import (
    Fitting,
)
from deepmd.tf.loss.loss import (
    Loss,
)
from deepmd.tf.loss.tensor import (
    TensorLoss,
)
from deepmd.tf.utils.graph import (
    get_fitting_net_variables_from_graph_def,
)
from deepmd.tf.utils.network import (
    one_layer,
    one_layer_rand_seed_shift,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@Fitting.register("dipole")
class DipoleFittingSeA(Fitting):
    r"""Fit the atomic dipole with descriptor se_a.

    Parameters
    ----------
    ntypes
            The ntypes of the descriptor :math:`\mathcal{D}`
    dim_descrpt
            The dimension of the descriptor :math:`\mathcal{D}`
    embedding_width
            The rotation matrix dimension of the descriptor :math:`\mathcal{D}`
    neuron : list[int]
            Number of neurons in each hidden layer of the fitting net
    resnet_dt : bool
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    numb_fparam
            Number of frame parameters
    numb_aparam
            Number of atomic parameters
    dim_case_embd
            Dimension of case specific embedding.
    sel_type : list[int]
            The atom types selected to have an atomic dipole prediction. If is None, all atoms are selected.
    seed : int
            Random seed for initializing the network parameters.
    activation_function : str
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision : str
            The precision of the embedding net parameters. Supported options are |PRECISION|
    uniform_seed
            Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    mixed_types : bool
        If true, use a uniform fitting net for all atom types, otherwise use
        different fitting nets for different atom types.
    type_map: list[str], Optional
            A list of strings. Give the name to each type of atoms.
    trainable : list[bool], Optional
        If the weights of fitting net are trainable.
        Suppose that we have :math:`N_l` hidden layers in the fitting net,
        this list is of length :math:`N_l + 1`, specifying if the hidden layers and the output layer are trainable.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        embedding_width: int,
        neuron: list[int] = [120, 120, 120],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        sel_type: Optional[list[int]] = None,
        seed: Optional[int] = None,
        activation_function: str = "tanh",
        precision: str = "default",
        uniform_seed: bool = False,
        mixed_types: bool = False,
        type_map: Optional[list[str]] = None,  # to be compat with input
        trainable: Optional[list[bool]] = None,
        **kwargs,
    ) -> None:
        """Constructor."""
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.n_neuron = neuron
        self.resnet_dt = resnet_dt
        self.sel_type = sel_type
        if self.sel_type is None:
            self.sel_type = list(range(self.ntypes))
        self.sel_mask = np.array(
            [ii in self.sel_type for ii in range(self.ntypes)], dtype=bool
        )
        self.seed = seed
        self.uniform_seed = uniform_seed
        self.seed_shift = one_layer_rand_seed_shift()
        self.activation_function_name = activation_function
        self.fitting_activation_fn = get_activation_func(activation_function)
        self.fitting_precision = get_precision(precision)
        self.dim_rot_mat_1 = embedding_width
        self.dim_rot_mat = self.dim_rot_mat_1 * 3
        self.useBN = False
        self.fitting_net_variables = None
        self.mixed_prec = None
        self.mixed_types = mixed_types
        self.type_map = type_map
        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam
        self.dim_case_embd = dim_case_embd
        if numb_fparam > 0:
            raise ValueError("numb_fparam is not supported in the dipole fitting")
        if numb_aparam > 0:
            raise ValueError("numb_aparam is not supported in the dipole fitting")
        if dim_case_embd > 0:
            raise ValueError("dim_case_embd is not supported in TensorFlow.")
        self.fparam_avg = None
        self.fparam_std = None
        self.fparam_inv_std = None
        self.aparam_avg = None
        self.aparam_std = None
        self.aparam_inv_std = None
        if trainable is None:
            self.trainable = [True for _ in range(len(self.n_neuron) + 1)]
        elif isinstance(trainable, bool):
            self.trainable = [trainable] * (len(self.n_neuron) + 1)
        else:
            self.trainable = trainable
        assert len(self.trainable) == len(self.n_neuron) + 1, (
            "length of trainable should be that of n_neuron + 1"
        )

    def get_sel_type(self) -> int:
        """Get selected type."""
        return self.sel_type

    def get_out_size(self) -> int:
        """Get the output size. Should be 3."""
        return 3

    @cast_precision
    def _build_lower(self, start_index, natoms, inputs, rot_mat, suffix="", reuse=None):
        # cut-out inputs
        inputs_i = tf.slice(inputs, [0, start_index, 0], [-1, natoms, -1])
        inputs_i = tf.reshape(inputs_i, [-1, self.dim_descrpt])
        rot_mat_i = tf.slice(rot_mat, [0, start_index, 0], [-1, natoms, -1])
        rot_mat_i = tf.reshape(rot_mat_i, [-1, self.dim_rot_mat_1, 3])
        layer = inputs_i
        for ii in range(0, len(self.n_neuron)):
            if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii - 1]:
                layer += one_layer(
                    layer,
                    self.n_neuron[ii],
                    name="layer_" + str(ii) + suffix,
                    reuse=reuse,
                    seed=self.seed,
                    use_timestep=self.resnet_dt,
                    activation_fn=self.fitting_activation_fn,
                    precision=self.fitting_precision,
                    uniform_seed=self.uniform_seed,
                    initial_variables=self.fitting_net_variables,
                    mixed_prec=self.mixed_prec,
                    trainable=self.trainable[ii],
                )
            else:
                layer = one_layer(
                    layer,
                    self.n_neuron[ii],
                    name="layer_" + str(ii) + suffix,
                    reuse=reuse,
                    seed=self.seed,
                    activation_fn=self.fitting_activation_fn,
                    precision=self.fitting_precision,
                    uniform_seed=self.uniform_seed,
                    initial_variables=self.fitting_net_variables,
                    mixed_prec=self.mixed_prec,
                    trainable=self.trainable[ii],
                )
            if (not self.uniform_seed) and (self.seed is not None):
                self.seed += self.seed_shift
        # (nframes x natoms) x naxis
        final_layer = one_layer(
            layer,
            self.dim_rot_mat_1,
            activation_fn=None,
            name="final_layer" + suffix,
            reuse=reuse,
            seed=self.seed,
            precision=self.fitting_precision,
            uniform_seed=self.uniform_seed,
            initial_variables=self.fitting_net_variables,
            mixed_prec=self.mixed_prec,
            final_layer=True,
            trainable=self.trainable[-1],
        )
        if (not self.uniform_seed) and (self.seed is not None):
            self.seed += self.seed_shift
        # (nframes x natoms) x 1 * naxis
        final_layer = tf.reshape(
            final_layer, [tf.shape(inputs)[0] * natoms, 1, self.dim_rot_mat_1]
        )
        # (nframes x natoms) x 1 x 3(coord)
        final_layer = tf.matmul(final_layer, rot_mat_i)
        # nframes x natoms x 3
        final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0], natoms, 3])
        return final_layer

    def build(
        self,
        input_d: tf.Tensor,
        rot_mat: tf.Tensor,
        natoms: tf.Tensor,
        input_dict: Optional[dict] = None,
        reuse: Optional[bool] = None,
        suffix: str = "",
    ) -> tf.Tensor:
        """Build the computational graph for fitting net.

        Parameters
        ----------
        input_d
            The input descriptor
        rot_mat
            The rotation matrix from the descriptor.
        natoms
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        input_dict
            Additional dict for inputs.
        reuse
            The weights in the networks should be reused when get the variable.
        suffix
            Name suffix to identify this descriptor

        Returns
        -------
        dipole
            The atomic dipole.
        """
        if input_dict is None:
            input_dict = {}
        type_embedding = input_dict.get("type_embedding", None)
        atype = input_dict.get("atype", None)
        nframes = input_dict.get("nframes")
        start_index = 0
        inputs = tf.reshape(input_d, [-1, natoms[0], self.dim_descrpt])
        rot_mat = tf.reshape(rot_mat, [-1, natoms[0], self.dim_rot_mat])
        if nframes is None:
            nframes = tf.shape(inputs)[0]

        if self.mixed_types or type_embedding is not None:
            # keep old behavior
            self.mixed_types = True
            nloc_mask = tf.reshape(
                tf.tile(tf.repeat(self.sel_mask, natoms[2:]), [nframes]), [nframes, -1]
            )
            atype_nall = tf.reshape(atype, [-1, natoms[1]])
            # (nframes x nloc_masked)
            self.atype_nloc_masked = tf.reshape(
                tf.slice(atype_nall, [0, 0], [-1, natoms[0]])[nloc_mask], [-1]
            )  ## lammps will make error
            self.nloc_masked = tf.shape(
                tf.reshape(self.atype_nloc_masked, [nframes, -1])
            )[1]

        if type_embedding is not None:
            atype_embed = tf.nn.embedding_lookup(type_embedding, self.atype_nloc_masked)
        else:
            atype_embed = None

        self.atype_embed = atype_embed
        if atype_embed is not None:
            inputs = tf.reshape(
                tf.reshape(inputs, [nframes, natoms[0], self.dim_descrpt])[nloc_mask],
                [-1, self.dim_descrpt],
            )
            rot_mat = tf.reshape(
                tf.reshape(rot_mat, [nframes, natoms[0], self.dim_rot_mat_1 * 3])[
                    nloc_mask
                ],
                [-1, self.dim_rot_mat_1, 3],
            )
            atype_embed = tf.cast(atype_embed, self.fitting_precision)
            type_shape = atype_embed.get_shape().as_list()
            inputs = tf.concat([inputs, atype_embed], axis=1)
            self.dim_descrpt = self.dim_descrpt + type_shape[1]

        if not self.mixed_types:
            count = 0
            outs_list = []
            for type_i in range(self.ntypes):
                if type_i not in self.sel_type:
                    start_index += natoms[2 + type_i]
                    continue
                final_layer = self._build_lower(
                    start_index,
                    natoms[2 + type_i],
                    inputs,
                    rot_mat,
                    suffix="_type_" + str(type_i) + suffix,
                    reuse=reuse,
                )
                start_index += natoms[2 + type_i]
                # concat the results
                outs_list.append(final_layer)
                count += 1
            outs = tf.concat(outs_list, axis=1)
        else:
            inputs = tf.reshape(inputs, [nframes, self.nloc_masked, self.dim_descrpt])
            rot_mat = tf.reshape(
                rot_mat, [nframes, self.nloc_masked, self.dim_rot_mat_1 * 3]
            )
            final_layer = self._build_lower(
                0, self.nloc_masked, inputs, rot_mat, suffix=suffix, reuse=reuse
            )
            # nframes x natoms x 3
            outs = tf.reshape(final_layer, [nframes, self.nloc_masked, 3])

        tf.summary.histogram("fitting_net_output", outs)
        return tf.reshape(outs, [-1])
        # return tf.reshape(outs, [tf.shape(inputs)[0] * natoms[0] * 3 // 3])

    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        suffix: str = "",
    ) -> None:
        """Init the fitting net variables with the given dict.

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str
            suffix to name scope
        """
        self.fitting_net_variables = get_fitting_net_variables_from_graph_def(
            graph_def, suffix=suffix
        )

    def enable_mixed_precision(self, mixed_prec: Optional[dict] = None) -> None:
        """Receive the mixed precision setting.

        Parameters
        ----------
        mixed_prec
            The mixed precision setting used in the embedding net
        """
        self.mixed_prec = mixed_prec
        self.fitting_precision = get_precision(mixed_prec["output_prec"])

    def get_loss(self, loss: dict, lr) -> Loss:
        """Get the loss function.

        Parameters
        ----------
        loss : dict
            the loss dict
        lr : LearningRateExp
            the learning rate

        Returns
        -------
        Loss
            the loss function
        """
        return TensorLoss(
            loss,
            model=self,
            tensor_name="dipole",
            tensor_size=3,
            label_name="dipole",
        )

    def serialize(self, suffix: str) -> dict:
        """Serialize the model.

        Returns
        -------
        dict
            The serialized data
        """
        data = {
            "@class": "Fitting",
            "type": "dipole",
            "@version": 3,
            "ntypes": self.ntypes,
            "dim_descrpt": self.dim_descrpt,
            "embedding_width": self.dim_rot_mat_1,
            "mixed_types": self.mixed_types,
            "dim_out": 3,
            "neuron": self.n_neuron,
            "resnet_dt": self.resnet_dt,
            "numb_fparam": self.numb_fparam,
            "numb_aparam": self.numb_aparam,
            "dim_case_embd": self.dim_case_embd,
            "activation_function": self.activation_function_name,
            "precision": self.fitting_precision.name,
            "exclude_types": [],
            "nets": self.serialize_network(
                ntypes=self.ntypes,
                ndim=0 if self.mixed_types else 1,
                in_dim=self.dim_descrpt,
                out_dim=self.dim_rot_mat_1,
                neuron=self.n_neuron,
                activation_function=self.activation_function_name,
                resnet_dt=self.resnet_dt,
                variables=self.fitting_net_variables,
                trainable=self.trainable,
                suffix=suffix,
            ),
            "type_map": self.type_map,
        }
        return data

    @classmethod
    def deserialize(cls, data: dict, suffix: str):
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
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 3, 1)
        fitting = cls(**data)
        fitting.fitting_net_variables = cls.deserialize_network(
            data["nets"],
            suffix=suffix,
        )
        return fitting

    @property
    def input_requirement(self) -> list[DataRequirementItem]:
        """Return data requirements needed for the model input."""
        data_requirement = []
        if self.numb_fparam > 0:
            data_requirement.append(
                DataRequirementItem(
                    "fparam", self.numb_fparam, atomic=False, must=True, high_prec=False
                )
            )
        if self.numb_aparam > 0:
            data_requirement.append(
                DataRequirementItem(
                    "aparam", self.numb_aparam, atomic=True, must=True, high_prec=False
                )
            )
        return data_requirement

    def get_numb_fparam(self) -> int:
        """Get the number of frame parameters."""
        return self.numb_fparam

    def get_numb_aparam(self) -> int:
        """Get the number of atomic parameters."""
        return self.numb_aparam
