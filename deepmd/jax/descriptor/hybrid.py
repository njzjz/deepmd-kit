# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import array_api_compat

from packaging.version import (
    Version,
)

from deepmd.dpmodel.descriptor.hybrid import DescrptHybrid as DescrptHybridDP
from deepmd.dpmodel.utils.nlist import (
    nlist_distinguish_types,
)
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    to_jax_array,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.env import (
    flax_version,
    nnx,
)


@BaseDescriptor.register("hybrid")
@flax_module
class DescrptHybrid(DescrptHybridDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"nlist_cut_idx"}:
            value = [ArrayAPIVariable(to_jax_array(vv)) for vv in value]
            if Version(flax_version) >= Version("0.12.0"):
                value = nnx.List([nnx.data(item) for item in value])
        elif name in {"descrpt_list"}:
            value = [BaseDescriptor.deserialize(vv.serialize()) for vv in value]
            if Version(flax_version) >= Version("0.12.0"):
                value = nnx.List([nnx.data(item) for item in value])

        return super().__setattr__(name, value)

    def call(self, *args: Any, **kwargs: Any) -> tuple[Any, Any | None, Any | None, Any | None, Any | None]:
        if len(args) < 3:
            return super().call(*args, **kwargs)
        if len(args) > 4:
            return super().call(*args, **kwargs)
        if kwargs and set(kwargs) != {"mapping"}:
            return super().call(*args, **kwargs)
        coord_ext, atype_ext, nlist = args[:3]
        mapping = kwargs.pop("mapping", args[3] if len(args) == 4 else None)
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        out_descriptor = []
        out_gr = []
        out_g2 = None
        out_h2 = None
        out_sw = None
        if self.sel_no_mixed_types is not None:
            nl_distinguish_types = nlist_distinguish_types(
                nlist,
                atype_ext,
                self.sel_no_mixed_types,
            )
        else:
            nl_distinguish_types = None
        for descrpt, nci in zip(self.descrpt_list, self.nlist_cut_idx, strict=True):
            nci_value = getattr(nci, "value", nci)
            if self.mixed_types() == descrpt.mixed_types():
                nl = xp.take(nlist, nci_value, axis=2)
            else:
                assert nl_distinguish_types is not None
                nl = nl_distinguish_types[:, :, nci_value]
            odescriptor, gr, g2, h2, sw = descrpt(
                coord_ext, atype_ext, nl, mapping
            )
            out_descriptor.append(odescriptor)
            if gr is not None:
                out_gr.append(gr)

        out_descriptor = xp.concat(out_descriptor, axis=-1)
        out_gr = xp.concat(out_gr, axis=-2) if out_gr else None
        return out_descriptor, out_gr, out_g2, out_h2, out_sw
