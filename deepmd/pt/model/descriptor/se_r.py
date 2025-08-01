# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Optional,
    Union,
)

import numpy as np
import torch
import torch.nn as nn

from deepmd.dpmodel.utils import EnvMat as DPEnvMat
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.descriptor import (
    prod_env_mat,
)
from deepmd.pt.model.network.mlp import (
    EmbeddingNet,
    NetworkCollection,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.pt.utils.tabulate import (
    DPTabulate,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.env_mat_stat import (
    StatItem,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)

if not hasattr(torch.ops.deepmd, "tabulate_fusion_se_r"):

    def tabulate_fusion_se_r(
        argument0: torch.Tensor,
        argument1: torch.Tensor,
        argument2: torch.Tensor,
        argument3: int,
    ) -> list[torch.Tensor]:
        raise NotImplementedError(
            "tabulate_fusion_se_r is not available since customized PyTorch OP library is not built when freezing the model. "
            "See documentation for model compression for details."
        )

    # Note: this hack cannot actually save a model that can be runned using LAMMPS.
    torch.ops.deepmd.tabulate_fusion_se_r = tabulate_fusion_se_r


@BaseDescriptor.register("se_e2_r")
@BaseDescriptor.register("se_r")
class DescrptSeR(BaseDescriptor, torch.nn.Module):
    def __init__(
        self,
        rcut,
        rcut_smth,
        sel,
        neuron=[25, 50, 100],
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
        type_map: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.rcut = float(rcut)
        self.rcut_smth = float(rcut_smth)
        self.neuron = neuron
        self.filter_neuron = self.neuron
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.resnet_dt = resnet_dt
        self.exclude_types = exclude_types
        self.ntypes = len(sel)
        self.type_map = type_map
        self.seed = seed
        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)
        self.env_protection = env_protection

        self.sel = sel
        self.sec = torch.tensor(
            np.append([0], np.cumsum(self.sel)), dtype=int, device=env.DEVICE
        )
        self.split_sel = self.sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 1

        wanted_shape = (self.ntypes, self.nnei, 1)
        mean = torch.zeros(wanted_shape, dtype=self.prec, device=env.DEVICE)
        stddev = torch.ones(wanted_shape, dtype=self.prec, device=env.DEVICE)
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.filter_layers_old = None
        self.filter_layers = None

        filter_layers = NetworkCollection(
            ndim=1, ntypes=len(sel), network_type="embedding_network"
        )
        # TODO: ndim=2 if type_one_side=False
        for ii in range(self.ntypes):
            filter_layers[(ii,)] = EmbeddingNet(
                1,
                self.filter_neuron,
                activation_function=self.activation_function,
                precision=self.precision,
                resnet_dt=self.resnet_dt,
                seed=child_seed(self.seed, ii),
                trainable=trainable,
            )
        self.filter_layers = filter_layers
        self.stats = None
        # set trainable
        self.trainable = trainable
        for param in self.parameters():
            param.requires_grad = trainable

        # add for compression
        self.compress = False
        self.compress_info = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(0, dtype=self.prec, device="cpu"))
                for _ in range(len(self.filter_layers.networks))
            ]
        )
        self.compress_data = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(0, dtype=self.prec, device=env.DEVICE))
                for _ in range(len(self.filter_layers.networks))
            ]
        )

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.rcut_smth

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        """Returns the number of selected atoms for each type."""
        return self.sel

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_type_map(self) -> list[str]:
        """Get the name to each type of atoms."""
        return self.type_map

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.neuron[-1]

    def get_dim_emb(self) -> int:
        """Returns the output dimension."""
        raise NotImplementedError

    def get_dim_in(self) -> int:
        """Returns the input dimension."""
        return 0

    def mixed_types(self) -> bool:
        """If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return False

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor needs sorted nlist when using `forward_lower`."""
        return False

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.env_protection

    def share_params(self, base_class, shared_level, resume=False) -> None:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert self.__class__ == base_class.__class__, (
            "Only descriptors of the same type can share params!"
        )
        # For SeR descriptors, the user-defined share-level
        # shared_level: 0
        if shared_level == 0:
            # link buffers
            if hasattr(self, "mean") and not resume:
                # in case of change params during resume
                base_env = EnvMatStatSe(base_class)
                base_env.stats = base_class.stats
                for kk in base_class.get_stats():
                    base_env.stats[kk] += self.get_stats()[kk]
                mean, stddev = base_env()
                if not base_class.set_davg_zero:
                    base_class.mean.copy_(
                        torch.tensor(
                            mean, device=env.DEVICE, dtype=base_class.mean.dtype
                        )
                    )
                base_class.stddev.copy_(
                    torch.tensor(
                        stddev, device=env.DEVICE, dtype=base_class.stddev.dtype
                    )
                )
                self.mean = base_class.mean
                self.stddev = base_class.stddev
            # self.load_state_dict(base_class.state_dict()) # this does not work, because it only inits the model
            # the following will successfully link all the params except buffers
            for item in self._modules:
                self._modules[item] = base_class._modules[item]
        # Other shared levels
        else:
            raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        raise NotImplementedError(
            "Descriptor se_e2_r does not support changing for type related params!"
            "This feature is currently not implemented because it would require additional work to support the non-mixed-types case. "
            "We may consider adding this support in the future if there is a clear demand for it."
        )

    def compute_input_stats(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        path: Optional[DPPath] = None,
    ) -> None:
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
        env_mat_stat = EnvMatStatSe(self)
        if path is not None:
            path = path / env_mat_stat.get_hash()
        if path is None or not path.is_dir():
            if callable(merged):
                # only get data for once
                sampled = merged()
            else:
                sampled = merged
        else:
            sampled = []
        env_mat_stat.load_or_compute_stats(sampled, path)
        self.stats = env_mat_stat.stats
        mean, stddev = env_mat_stat()
        if not self.set_davg_zero:
            self.mean.copy_(
                torch.tensor(mean, device=env.DEVICE, dtype=self.mean.dtype)
            )
        self.stddev.copy_(
            torch.tensor(stddev, device=env.DEVICE, dtype=self.stddev.dtype)
        )

    def get_stats(self) -> dict[str, StatItem]:
        """Get the statistics of the descriptor."""
        if self.stats is None:
            raise RuntimeError(
                "The statistics of the descriptor has not been computed."
            )
        return self.stats

    def __setitem__(self, key, value) -> None:
        if key in ("avg", "data_avg", "davg"):
            self.mean = value
        elif key in ("std", "data_std", "dstd"):
            self.stddev = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key in ("avg", "data_avg", "davg"):
            return self.mean
        elif key in ("std", "data_std", "dstd"):
            return self.stddev
        else:
            raise KeyError(key)

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Receive the statisitcs (distance, max_nbor_size and env_mat_range) of the training data.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        """
        if self.compress:
            raise ValueError("Compression is already enabled.")
        data = self.serialize()
        table = DPTabulate(
            self,
            data["neuron"],
            data["type_one_side"],
            data["exclude_types"],
            ActivationFn(data["activation_function"]),
        )
        table_config = [
            table_extrapolate,
            table_stride_1,
            table_stride_2,
            check_frequency,
        ]
        lower, upper = table.build(
            min_nbor_dist, table_extrapolate, table_stride_1, table_stride_2
        )
        table_data = table.data

        for ii, ll in enumerate(self.filter_layers.networks):
            net = "filter_-1_net_" + str(ii)
            info_ii = torch.as_tensor(
                [
                    lower[net],
                    upper[net],
                    upper[net] * table_config[0],
                    table_config[1],
                    table_config[2],
                    table_config[3],
                ],
                dtype=self.prec,
                device="cpu",
            )
            tensor_data_ii = table_data[net].to(device=env.DEVICE, dtype=self.prec)
            self.compress_data[ii] = tensor_data_ii
            self.compress_info[ii] = info_ii

        self.compress = True

    def forward(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ):
        """Compute the descriptor.

        Parameters
        ----------
        coord_ext
            The extended coordinates of atoms. shape: nf x (nallx3)
        atype_ext
            The extended aotm types. shape: nf x nall
        nlist
            The neighbor list. shape: nf x nloc x nnei
        mapping
            The index mapping, not required by this descriptor.
        comm_dict
            The data needed for communication for parallel inference.

        Returns
        -------
        descriptor
            The descriptor. shape: nf x nloc x (ng x axis_neuron)
        gr
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x ng x 3
        g2
            The rotationally invariant pair-partical representation.
            this descriptor returns None
        h2
            The rotationally equivariant pair-partical representation.
            this descriptor returns None
        sw
            The smooth switch function.

        """
        # cast the input to internal precsion
        coord_ext = coord_ext.to(dtype=self.prec)
        del mapping, comm_dict
        nf = nlist.shape[0]
        nloc = nlist.shape[1]
        atype = atype_ext[:, :nloc]
        dmatrix, diff, sw = prod_env_mat(
            coord_ext,
            nlist,
            atype,
            self.mean,
            self.stddev,
            self.rcut,
            self.rcut_smth,
            True,
            protection=self.env_protection,
        )

        assert self.filter_layers is not None
        dmatrix = dmatrix.view(-1, self.nnei, 1)
        nfnl = dmatrix.shape[0]
        # pre-allocate a shape to pass jit
        xyz_scatter = torch.zeros(
            [nfnl, 1, self.filter_neuron[-1]], dtype=self.prec, device=coord_ext.device
        )

        # nfnl x nnei
        exclude_mask = self.emask(nlist, atype_ext).view(nfnl, self.nnei)
        xyz_scatter_total = []
        for ii, (ll, compress_data_ii, compress_info_ii) in enumerate(
            zip(self.filter_layers.networks, self.compress_data, self.compress_info)
        ):
            # nfnl x nt
            mm = exclude_mask[:, self.sec[ii] : self.sec[ii + 1]]
            # nfnl x nt x 1
            ss = dmatrix[:, self.sec[ii] : self.sec[ii + 1], :]
            ss = ss * mm[:, :, None]
            if self.compress:
                ss = ss.squeeze(-1)
                xyz_scatter = torch.ops.deepmd.tabulate_fusion_se_r(
                    compress_data_ii.contiguous(),
                    compress_info_ii.cpu().contiguous(),
                    ss,
                    self.filter_neuron[-1],
                )[0]
                xyz_scatter_total.append(xyz_scatter)
            else:
                # nfnl x nt x ng
                gg = ll.forward(ss)
                gg = torch.mean(gg, dim=1).unsqueeze(1)
                xyz_scatter += gg * (self.sel[ii] / self.nnei)

        res_rescale = 1.0 / 5.0
        if self.compress:
            xyz_scatter = torch.cat(xyz_scatter_total, dim=1)
            result = torch.mean(xyz_scatter, dim=1) * res_rescale
        else:
            result = xyz_scatter * res_rescale
        result = result.view(nf, nloc, self.filter_neuron[-1])
        return (
            result.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            None,
            None,
            None,
            sw.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
        )

    def set_stat_mean_and_stddev(
        self,
        mean: torch.Tensor,
        stddev: torch.Tensor,
    ) -> None:
        """Update mean and stddev for descriptor."""
        self.mean = mean
        self.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get mean and stddev for descriptor."""
        return self.mean, self.stddev

    def serialize(self) -> dict:
        return {
            "@class": "Descriptor",
            "type": "se_r",
            "@version": 2,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
            "neuron": self.neuron,
            "resnet_dt": self.resnet_dt,
            "set_davg_zero": self.set_davg_zero,
            "activation_function": self.activation_function,
            # make deterministic
            "precision": RESERVED_PRECISION_DICT[self.prec],
            "embeddings": self.filter_layers.serialize(),
            "env_mat": DPEnvMat(self.rcut, self.rcut_smth).serialize(),
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "@variables": {
                "davg": self["davg"].detach().cpu().numpy(),
                "dstd": self["dstd"].detach().cpu().numpy(),
            },
            "type_map": self.type_map,
            ## to be updated when the options are supported.
            "trainable": True,
            "type_one_side": True,
            "spin": None,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeR":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        env_mat = data.pop("env_mat")
        obj = cls(**data)

        def t_cvt(xx):
            return torch.tensor(xx, dtype=obj.prec, device=env.DEVICE)

        obj["davg"] = t_cvt(variables["davg"])
        obj["dstd"] = t_cvt(variables["dstd"])
        obj.filter_layers = NetworkCollection.deserialize(embeddings)
        return obj

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
        min_nbor_dist, local_jdata_cpy["sel"] = UpdateSel().update_one_sel(
            train_data, type_map, local_jdata_cpy["rcut"], local_jdata_cpy["sel"], False
        )
        return local_jdata_cpy, min_nbor_dist
