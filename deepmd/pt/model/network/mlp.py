# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    ClassVar,
    Optional,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepmd.pt.utils import (
    env,
)

device = env.DEVICE

from deepmd.dpmodel.utils import (
    NativeLayer,
)
from deepmd.dpmodel.utils import NetworkCollection as DPNetworkCollection
from deepmd.dpmodel.utils import (
    make_embedding_network,
    make_fitting_network,
    make_multilayer_network,
)
from deepmd.pt.model.network.init import (
    kaiming_normal_,
    normal_,
    trunc_normal_,
    xavier_uniform_,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
    get_generator,
    to_numpy_array,
    to_torch_tensor,
)


def empty_t(shape, precision):
    return torch.empty(shape, dtype=precision, device=device)


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        xx: torch.Tensor,
    ) -> torch.Tensor:
        """The Identity operation layer."""
        return xx

    def serialize(self) -> dict:
        return {
            "@class": "Identity",
            "@version": 1,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Identity":
        return Identity()


class MLPLayer(nn.Module):
    def __init__(
        self,
        num_in,
        num_out,
        bias: bool = True,
        use_timestep: bool = False,
        activation_function: Optional[str] = None,
        resnet: bool = False,
        bavg: float = 0.0,
        stddev: float = 1.0,
        precision: str = DEFAULT_PRECISION,
        init: str = "default",
        seed: Optional[Union[int, list[int]]] = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.trainable = trainable
        # only use_timestep when skip connection is established.
        self.use_timestep = use_timestep and (
            num_out == num_in or num_out == num_in * 2
        )
        self.num_in = num_in
        self.num_out = num_out
        self.activate_name = activation_function
        self.activate = ActivationFn(self.activate_name)
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.matrix = nn.Parameter(data=empty_t((num_in, num_out), self.prec))
        random_generator = get_generator(seed)
        if bias:
            self.bias = nn.Parameter(
                data=empty_t([num_out], self.prec),
            )
        else:
            self.bias = None
        if self.use_timestep:
            self.idt = nn.Parameter(data=empty_t([num_out], self.prec))
        else:
            self.idt = None
        self.resnet = resnet
        if init == "default":
            self._default_normal_init(
                bavg=bavg, stddev=stddev, generator=random_generator
            )
        elif init == "trunc_normal":
            self._trunc_normal_init(1.0, generator=random_generator)
        elif init == "relu":
            self._trunc_normal_init(2.0, generator=random_generator)
        elif init == "glorot":
            self._glorot_uniform_init(generator=random_generator)
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "kaiming_normal":
            self._normal_init(generator=random_generator)
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError(f"Unknown initialization method: {init}")

    def check_type_consistency(self) -> None:
        precision = self.precision

        def check_var(var) -> None:
            if var is not None:
                # assertion "float64" == "double" would fail
                assert PRECISION_DICT[var.dtype.name] is PRECISION_DICT[precision]

        check_var(self.matrix)
        check_var(self.bias)
        check_var(self.idt)

    def dim_in(self) -> int:
        return self.matrix.shape[0]

    def dim_out(self) -> int:
        return self.matrix.shape[1]

    def _default_normal_init(
        self,
        bavg: float = 0.0,
        stddev: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        normal_(
            self.matrix.data,
            std=stddev / np.sqrt(self.num_out + self.num_in),
            generator=generator,
        )
        if self.bias is not None:
            normal_(self.bias.data, mean=bavg, std=stddev, generator=generator)
        if self.idt is not None:
            normal_(self.idt.data, mean=0.1, std=0.001, generator=generator)

    def _trunc_normal_init(
        self, scale=1.0, generator: Optional[torch.Generator] = None
    ) -> None:
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.matrix.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        trunc_normal_(self.matrix, mean=0.0, std=std, generator=generator)

    def _glorot_uniform_init(self, generator: Optional[torch.Generator] = None) -> None:
        xavier_uniform_(self.matrix, gain=1, generator=generator)

    def _zero_init(self, use_bias=True) -> None:
        with torch.no_grad():
            self.matrix.fill_(0.0)
            if use_bias and self.bias is not None:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self, generator: Optional[torch.Generator] = None) -> None:
        kaiming_normal_(self.matrix, nonlinearity="linear", generator=generator)

    def forward(
        self,
        xx: torch.Tensor,
    ) -> torch.Tensor:
        """One MLP layer used by DP model.

        Parameters
        ----------
        xx : torch.Tensor
            The input.

        Returns
        -------
        yy: torch.Tensor
            The output.
        """
        ori_prec = xx.dtype
        if not env.DP_DTYPE_PROMOTION_STRICT:
            xx = xx.to(self.prec)
        yy = F.linear(xx, self.matrix.t(), self.bias)
        yy = self.activate(yy)
        yy = yy * self.idt if self.idt is not None else yy
        if self.resnet:
            if xx.shape[-1] == yy.shape[-1]:
                yy = yy + xx
            elif 2 * xx.shape[-1] == yy.shape[-1]:
                yy = yy + torch.concat([xx, xx], dim=-1)
            else:
                yy = yy
        if not env.DP_DTYPE_PROMOTION_STRICT:
            yy = yy.to(ori_prec)
        return yy

    def serialize(self) -> dict:
        """Serialize the layer to a dict.

        Returns
        -------
        dict
            The serialized layer.
        """
        nl = NativeLayer(
            self.matrix.shape[0],
            self.matrix.shape[1],
            bias=self.bias is not None,
            use_timestep=self.idt is not None,
            activation_function=self.activate_name,
            resnet=self.resnet,
            precision=self.precision,
            trainable=self.trainable,
        )
        nl.w, nl.b, nl.idt = (
            to_numpy_array(self.matrix),
            to_numpy_array(self.bias),
            to_numpy_array(self.idt),
        )
        return nl.serialize()

    @classmethod
    def deserialize(cls, data: dict) -> "MLPLayer":
        """Deserialize the layer from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        nl = NativeLayer.deserialize(data)
        obj = cls(
            nl["matrix"].shape[0],
            nl["matrix"].shape[1],
            bias=nl["bias"] is not None,
            use_timestep=nl["idt"] is not None,
            activation_function=nl["activation_function"],
            resnet=nl["resnet"],
            precision=nl["precision"],
            trainable=nl["trainable"],
        )
        prec = PRECISION_DICT[obj.precision]

        def check_load_param(ss):
            return (
                nn.Parameter(data=to_torch_tensor(nl[ss]))
                if nl[ss] is not None
                else None
            )

        obj.matrix = check_load_param("matrix")
        obj.bias = check_load_param("bias")
        obj.idt = check_load_param("idt")
        return obj


MLP_ = make_multilayer_network(MLPLayer, nn.Module)


class MLP(MLP_):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = torch.nn.ModuleList(self.layers)

    forward = MLP_.call


EmbeddingNet = make_embedding_network(MLP, MLPLayer)

FittingNet = make_fitting_network(EmbeddingNet, MLP, MLPLayer)


class NetworkCollection(DPNetworkCollection, nn.Module):
    """PyTorch implementation of NetworkCollection."""

    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "network": MLP,
        "embedding_network": EmbeddingNet,
        "fitting_network": FittingNet,
    }

    def __init__(self, *args, **kwargs) -> None:
        # init both two base classes
        DPNetworkCollection.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        self.networks = self._networks = torch.nn.ModuleList(self._networks)
