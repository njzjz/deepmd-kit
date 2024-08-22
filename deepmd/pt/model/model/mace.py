# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import mace
import mace.modules
import torch
from e3nn import (
    o3,
)
from e3nn.util.jit import (
    script,
)

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.stat import (
    compute_output_stats,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.path import (
    DPPath,
)

PeriodicTable = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "HW": 49,
    "OW": 50,
    "mH": 51,
    "mHe": 52,
    "mLi": 53,
    "mBe": 54,
    "mB": 55,
    "mC": 56,
    "mN": 57,
    "mO": 58,
    "mF": 59,
    "mNe": 60,
    "mNa": 61,
    "mMg": 62,
    "mAl": 63,
    "mSi": 64,
    "mP": 65,
    "mS": 66,
    "mCl": 67,
    "mAr": 68,
    "mK": 69,
    "mCa": 70,
    "mSc": 71,
    "mTi": 72,
    "mV": 73,
    "mCr": 74,
    "mMn": 75,
    "mFe": 76,
    "mCo": 77,
    "mNi": 78,
    "mCu": 79,
    "mZn": 80,
    "mGa": 81,
    "mGe": 82,
    "mAs": 83,
    "mSe": 84,
    "mBr": 85,
    "mKr": 86,
    "mRb": 87,
    "mSr": 88,
    "mY": 89,
    "mZr": 90,
    "mNb": 91,
    "mMo": 92,
    "mTc": 93,
    "mRu": 94,
    "mRh": 95,
    "mPd": 96,
    "mAg": 97,
    "mCd": 98,
}


@BaseModel.register("mace")
class MaceModel(BaseModel):
    """Mace model.

    Parameters
    ----------
    type_map : list[str]
        The name of each type of atoms
    r_max : float, optional
        distance cutoff (in Ang)
    num_radial_basis : int, optional
        number of radial basis functions
    num_cutoff_basis : int, optional
        number of basis functions for smooth cutoff
    max_ell : int, optional
        highest ell of spherical harmonics
    interaction : str, optional
        name of interaction block
    num_interactions : int, optional
        number of interactions
    hidden_irreps : str, optional
        hidden irreps
    pair_repulsion : bool
        use amsgrad variant of optimizer
    distance_transform : str, optional
        distance transform
    correlation : int
        correlation order at each layer
    gate : str, optional
        non linearity for last readout
    MLP_irreps : str, optional
        hidden irreps of the MLP in last readout
    radial_type : str, optional
        type of radial basis functions
    radial_MLP : str, optional
        width of the radial MLP
    std : float, optional
        Standard deviation of force components in the training set
    """

    mm_types: List[int]

    def __init__(
        self,
        type_map: List[str],
        sel: int,
        r_max: float = 5.0,
        num_radial_basis: int = 8,
        num_cutoff_basis: int = 5,
        max_ell: int = 3,
        interaction: str = "RealAgnosticResidualInteractionBlock",
        num_interactions: int = 2,
        hidden_irreps: str = "128x0e + 128x1o",
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        correlation: int = 3,
        gate: str = "silu",
        MLP_irreps: str = "16x0e",
        radial_type: str = "bessel",
        radial_MLP: List[int] = [64, 64, 64],
        std: Optional[float] = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type_map = type_map
        self.ntypes = len(type_map)
        self.rcut = r_max
        atomic_numbers = []
        self.preset_out_bias = {"energy": []}
        self.mm_types = []
        self.sel = sel
        for ii, tt in enumerate(type_map):
            atomic_numbers.append(PeriodicTable[tt])
            if not tt.startswith("m") and tt not in {"HW", "OW"}:
                self.preset_out_bias["energy"].append(None)
            else:
                self.preset_out_bias["energy"].append([0])
                self.mm_types.append(ii)

        self.model = script(
            mace.modules.ScaleShiftMACE(
                r_max=r_max,
                num_bessel=num_radial_basis,
                num_polynomial_cutoff=num_cutoff_basis,
                max_ell=max_ell,
                interaction_cls=mace.modules.interaction_classes[interaction],
                num_interactions=num_interactions,
                num_elements=self.ntypes,
                hidden_irreps=o3.Irreps(hidden_irreps),
                atomic_energies=torch.zeros(self.ntypes),  # pylint: disable=no-explicit-device,no-explicit-dtype
                avg_num_neighbors=sel,
                atomic_numbers=atomic_numbers,
                pair_repulsion=pair_repulsion,
                distance_transform=distance_transform,
                correlation=correlation,
                gate=mace.modules.gate_dict[gate],
                interaction_cls_first=mace.modules.interaction_classes[
                    "RealAgnosticInteractionBlock"
                ],
                MLP_irreps=o3.Irreps(MLP_irreps),
                atomic_inter_scale=std,
                atomic_inter_shift=0.0,
                radial_MLP=radial_MLP,
                radial_type=radial_type,
            )
        )
        self.atomic_numbers = atomic_numbers

    def compute_or_load_stat(
        self,
        sampled_func,
        stat_file_path: Optional[DPPath] = None,
    ):
        """
        Compute or load the statistics parameters of the model,
        such as mean and standard deviation of descriptors or the energy bias of the fitting net.
        When `sampled` is provided, all the statistics parameters will be calculated (or re-calculated for update),
        and saved in the `stat_file_path`(s).
        When `sampled` is not provided, it will check the existence of `stat_file_path`(s)
        and load the calculated statistics parameters.

        Parameters
        ----------
        sampled_func
            The sampled data frames from different data systems.
        stat_file_path
            The path to the statistics files.
        """
        bias_out, _ = compute_output_stats(
            sampled_func,
            self.get_ntypes(),
            keys=["energy"],
            stat_file_path=stat_file_path,
            rcond=None,
            preset_bias=self.preset_out_bias,
        )
        self.model.atomic_energies_fn.atomic_energies = (
            bias_out["energy"]
            .view(self.model.atomic_energies_fn.atomic_energies.shape)
            .to(self.model.atomic_energies_fn.atomic_energies.dtype)
            .to(self.model.atomic_energies_fn.atomic_energies.device)
        )

    @torch.jit.export
    def fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of developer implemented atomic models."""
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy",
                    shape=[1],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                )
            ]
        )

    @torch.jit.export
    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.rcut

    @torch.jit.export
    def get_type_map(self) -> List[str]:
        """Get the type map."""
        return self.type_map

    @torch.jit.export
    def get_sel(self) -> List[int]:
        """Returns the number of selected atoms for each type."""
        return [self.sel]

    @torch.jit.export
    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return 0

    @torch.jit.export
    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return 0

    @torch.jit.export
    def get_sel_type(self) -> List[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return []

    @torch.jit.export
    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False

    @torch.jit.export
    def mixed_types(self) -> bool:
        """If true, the model
        1. assumes total number of atoms aligned across frames;
        2. uses a neighbor list that does not distinguish different atomic types.

        If false, the model
        1. assumes total number of atoms of each atom type aligned across frames;
        2. uses a neighbor list that distinguishes different atomic types.

        """
        return True

    @torch.jit.export
    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return False

    @torch.jit.export
    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        extended_coord, extended_atype, mapping, nlist = (
            extend_input_and_build_neighbor_list(
                coord, atype, self.rcut, self.get_sel(), True, box
            )
        )
        model_ret_lower = self.forward_lower_common(
            extended_coord,
            extended_atype,
            nlist,
            mapping=None,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=None,
        )
        model_ret = communicate_extended_output(
            model_ret_lower,
            ModelOutputDef(self.fitting_output_def()),
            mapping,
            do_atomic_virial,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
        if do_atomic_virial:
            model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-3)
        return model_predict

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        model_ret = self.forward_lower_common(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam,
            aparam,
            do_atomic_virial,
            comm_dict,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
        model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
        if do_atomic_virial:
            model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(-3)
        return model_predict

    def forward_lower_common(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        extended_coord_ = extended_coord
        nf, nall, _ = extended_coord_.shape
        _, nloc, _ = nlist.shape
        assert fparam is None
        assert aparam is None
        assert comm_dict is None
        nlist = nlist.to(torch.int64)
        extended_atype = extended_atype.to(torch.int64)
        nall = extended_coord.shape[1]

        # loop on nf
        energies = []
        forces = []
        virials = []
        atom_energies = []
        for ff in range(nf):
            extended_coord_ff = extended_coord[ff]
            extended_atype_ff = extended_atype[ff]
            nlist_ff = nlist[ff]
            edge_index = torch.ops.deepmd.mace_edge_index(
                nlist_ff,
                extended_atype_ff,
                torch.tensor(self.mm_types, dtype=torch.int64, device="cpu"),
            )
            edge_index = edge_index.T
            nedge = edge_index.shape[1]
            unit_shifts = torch.zeros(
                (nedge, 3), dtype=torch.float64, device=extended_coord_.device
            )
            # to one hot
            indices = extended_atype_ff.unsqueeze(-1)
            oh = torch.zeros(
                (nall, self.ntypes), device=extended_atype.device, dtype=torch.float64
            )
            # scatter_ is the in-place version of scatter
            oh.scatter_(dim=-1, index=indices, value=1)
            one_hot = oh.view((nall, self.ntypes))

            # cast to float32
            extended_coord_ff = extended_coord_ff.to(torch.float32)
            unit_shifts = unit_shifts.to(torch.float32)
            one_hot = one_hot.to(torch.float32)
            # it seems None is not allowed for data
            box = (
                torch.eye(
                    3, dtype=extended_coord_ff.dtype, device=extended_coord_ff.device
                )
                * 1000.0
            )

            ret = self.model.forward(
                {
                    "positions": extended_coord_ff,
                    "unit_shifts": unit_shifts,
                    "cell": box,
                    "edge_index": edge_index,
                    "batch": torch.zeros(
                        [nall], dtype=torch.int64, device=extended_coord_ff.device
                    ),
                    "node_attrs": one_hot,
                    "ptr": torch.tensor(
                        [0, nall], dtype=torch.int64, device=extended_coord_ff.device
                    ),
                    "weight": torch.tensor(
                        [1.0],
                        dtype=extended_coord_ff.dtype,
                        device=extended_coord_ff.device,
                    ),
                },
                compute_virials=True,
                training=self.training,
            )
            energy = ret["energy"]
            assert energy is not None
            energy = energy.view(1, 1).to(extended_coord_.dtype)
            force = ret["forces"]
            assert force is not None
            force = force.view(1, nall, 3).to(extended_coord_.dtype)
            virial = ret["virials"]
            assert virial is not None
            virial = virial.view(1, 9)
            atom_energy = ret["node_energy"]
            assert atom_energy is not None
            atom_energy = atom_energy.view(1, nall).to(extended_coord_.dtype)[:, :nall]
            energies.append(energy)
            forces.append(force)
            virials.append(virial)
            atom_energies.append(atom_energy)
        energies = torch.cat(energies, dim=0)
        forces = torch.cat(forces, dim=0)
        virials = torch.cat(virials, dim=0)
        atom_energies = torch.cat(atom_energies, dim=0)

        return {
            "energy_redu": energies.view(nf, 1),
            "energy_derv_r": forces.view(nf, nall, 1, 3),
            "energy_derv_c_redu": virials.view(nf, 1, 9),
            # take the first nloc atoms to match other models
            "energy": atom_energies.view(nf, nall, 1)[:, :nloc, :],
            # fake atom_virial
            "energy_derv_c": torch.zeros(
                (nf, nall, 1, 9),
                dtype=extended_coord_.dtype,
                device=extended_coord_.device,
            ),
        }

    def serialize(self) -> dict:
        raise NotImplementedError("not implemented")

    @classmethod
    def deserialize(cls, data: dict):
        raise NotImplementedError("not implemented")

    @torch.jit.export
    def get_nnei(self) -> int:
        """Returns the total number of selected neighboring atoms in the cut-off radius."""
        return self.sel

    @torch.jit.export
    def get_nsel(self) -> int:
        """Returns the total number of selected neighboring atoms in the cut-off radius."""
        return self.sel

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: Optional[List[str]],
        local_jdata: dict,
    ) -> Tuple[dict, Optional[float]]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statictics
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
            train_data, type_map, local_jdata_cpy["r_max"], local_jdata_cpy["sel"], True
        )
        local_jdata_cpy["sel"] = sel[0]
        return local_jdata_cpy, min_nbor_dist

    @torch.jit.export
    def model_output_type(self) -> List[str]:
        """Get the output type for the model."""
        return ["energy"]
