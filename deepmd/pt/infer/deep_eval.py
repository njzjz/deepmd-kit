# SPDX-License-Identifier: LGPL-3.0-or-later
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Union,
)

import numpy as np
import torch

from deepmd.dpmodel.common import PRECISION_DICT as NP_PRECISION_DICT
from deepmd.dpmodel.output_def import (
    ModelOutputDef,
    OutputVariableCategory,
    OutputVariableDef,
)
from deepmd.infer.deep_dipole import (
    DeepDipole,
)
from deepmd.infer.deep_dos import (
    DeepDOS,
)
from deepmd.infer.deep_eval import DeepEval as DeepEvalWrapper
from deepmd.infer.deep_eval import (
    DeepEvalBackend,
)
from deepmd.infer.deep_polar import (
    DeepGlobalPolar,
    DeepPolar,
)
from deepmd.infer.deep_pot import (
    DeepPot,
)
from deepmd.infer.deep_property import (
    DeepProperty,
)
from deepmd.infer.deep_wfc import (
    DeepWFC,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNetConsistent,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.pt.utils.env import (
    DEVICE,
    GLOBAL_PT_FLOAT_PRECISION,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.econf_embd import (
    sort_element_type,
)

if TYPE_CHECKING:
    import ase.neighborlist


class DeepEval(DeepEvalBackend):
    """PyTorch backend implementation of DeepEval.

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    output_def : ModelOutputDef
        The output definition of the model.
    *args : list
        Positional arguments.
    auto_batch_size : bool or int or AutomaticBatchSize, default: False
        If True, automatic batch size will be used. If int, it will be used
        as the initial batch size.
    neighbor_list : ase.neighborlist.NewPrimitiveNeighborList, optional
        The ASE neighbor list class to produce the neighbor list. If None, the
        neighbor list will be built natively in the model.
    **kwargs : dict
        Keyword arguments.
    """

    def __init__(
        self,
        model_file: str,
        output_def: ModelOutputDef,
        *args: Any,
        auto_batch_size: Union[bool, int, AutoBatchSize] = True,
        neighbor_list: Optional["ase.neighborlist.NewPrimitiveNeighborList"] = None,
        head: Optional[Union[str, int]] = None,
        no_jit: bool = False,
        **kwargs: Any,
    ) -> None:
        self.output_def = output_def
        self.model_path = model_file
        if str(self.model_path).endswith(".pt"):
            state_dict = torch.load(
                model_file, map_location=env.DEVICE, weights_only=True
            )
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.input_param = state_dict["_extra_state"]["model_params"]
            self.model_def_script = self.input_param
            self.multi_task = "model_dict" in self.input_param
            if self.multi_task:
                model_keys = list(self.input_param["model_dict"].keys())
                if isinstance(head, int):
                    head = model_keys[0]
                assert head is not None, (
                    f"Head must be set for multitask model! Available heads are: {model_keys}"
                )
                assert head in model_keys, (
                    f"No head named {head} in model! Available heads are: {model_keys}"
                )
                self.input_param = self.input_param["model_dict"][head]
                state_dict_head = {"_extra_state": state_dict["_extra_state"]}
                for item in state_dict:
                    if f"model.{head}." in item:
                        state_dict_head[
                            item.replace(f"model.{head}.", "model.Default.")
                        ] = state_dict[item].clone()
                state_dict = state_dict_head
            model = get_model(self.input_param).to(DEVICE)
            if not self.input_param.get("hessian_mode") and not no_jit:
                model = torch.jit.script(model)
            self.dp = ModelWrapper(model)
            self.dp.load_state_dict(state_dict)
        elif str(self.model_path).endswith(".pth"):
            model = torch.jit.load(model_file, map_location=env.DEVICE)
            self.dp = ModelWrapper(model)
            model_def_script = self.dp.model["Default"].get_model_def_script()
            if model_def_script:
                self.model_def_script = json.loads(model_def_script)
            else:
                self.model_def_script = {}
        else:
            raise ValueError("Unknown model file format!")
        self.dp.eval()
        self.rcut = self.dp.model["Default"].get_rcut()
        self.type_map = self.dp.model["Default"].get_type_map()
        if isinstance(auto_batch_size, bool):
            if auto_batch_size:
                self.auto_batch_size = AutoBatchSize()
            else:
                self.auto_batch_size = None
        elif isinstance(auto_batch_size, int):
            self.auto_batch_size = AutoBatchSize(auto_batch_size)
        elif isinstance(auto_batch_size, AutoBatchSize):
            self.auto_batch_size = auto_batch_size
        else:
            raise TypeError("auto_batch_size should be bool, int, or AutoBatchSize")
        self._has_spin = getattr(self.dp.model["Default"], "has_spin", False)
        if callable(self._has_spin):
            self._has_spin = self._has_spin()
        self._has_hessian = self.model_def_script.get("hessian_mode", False)

    def get_rcut(self) -> float:
        """Get the cutoff radius of this model."""
        return self.rcut

    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""
        return len(self.type_map)

    def get_type_map(self) -> list[str]:
        """Get the type map (element name of the atom types) of this model."""
        return self.type_map

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this DP."""
        return self.dp.model["Default"].get_dim_fparam()

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this DP."""
        return self.dp.model["Default"].get_dim_aparam()

    def get_intensive(self) -> bool:
        return self.dp.model["Default"].get_intensive()

    def get_var_name(self) -> str:
        """Get the name of the property."""
        if hasattr(self.dp.model["Default"], "get_var_name") and callable(
            getattr(self.dp.model["Default"], "get_var_name")
        ):
            return self.dp.model["Default"].get_var_name()
        else:
            raise NotImplementedError

    @property
    def model_type(self) -> type["DeepEvalWrapper"]:
        """The the evaluator of the model type."""
        model_output_type = self.dp.model["Default"].model_output_type()
        if "energy" in model_output_type:
            return DeepPot
        elif "dos" in model_output_type:
            return DeepDOS
        elif "dipole" in model_output_type:
            return DeepDipole
        elif "polar" in model_output_type:
            return DeepPolar
        elif "global_polar" in model_output_type:
            return DeepGlobalPolar
        elif "wfc" in model_output_type:
            return DeepWFC
        elif self.get_var_name() in model_output_type:
            return DeepProperty
        else:
            raise RuntimeError("Unknown model type")

    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.dp.model["Default"].get_sel_type()

    def get_numb_dos(self) -> int:
        """Get the number of DOS."""
        return self.dp.model["Default"].get_numb_dos()

    def get_task_dim(self) -> int:
        """Get the output dimension."""
        return self.dp.model["Default"].get_task_dim()

    def get_has_efield(self) -> bool:
        """Check if the model has efield."""
        return False

    def get_ntypes_spin(self) -> int:
        """Get the number of spin atom types of this model. Only used in old implement."""
        return 0

    def get_has_spin(self):
        """Check if the model has spin atom types."""
        return self._has_spin

    def get_has_hessian(self):
        """Check if the model has hessian."""
        return self._has_hessian

    def eval(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: np.ndarray,
        atomic: bool = False,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Evaluate the energy, force and virial by using this DP.

        Parameters
        ----------
        coords
            The coordinates of atoms.
            The array should be of size nframes x natoms x 3
        cells
            The cell of the region.
            If None then non-PBC is assumed, otherwise using PBC.
            The array should be of size nframes x 9
        atom_types
            The atom types
            The list should contain natoms ints
        atomic
            Calculate the atomic energy and virial
        fparam
            The frame parameter.
            The array can be of size :
            - nframes x dim_fparam.
            - dim_fparam. Then all frames are assumed to be provided with the same fparam.
        aparam
            The atomic parameter
            The array can be of size :
            - nframes x natoms x dim_aparam.
            - natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
            - dim_aparam. Then all frames and atoms are provided with the same aparam.
        **kwargs
            Other parameters

        Returns
        -------
        output_dict : dict
            The output of the evaluation. The keys are the names of the output
            variables, and the values are the corresponding output arrays.
        """
        # convert all of the input to numpy array
        atom_types = np.array(atom_types, dtype=np.int32)
        coords = np.array(coords)
        if cells is not None:
            cells = np.array(cells)
        natoms, numb_test = self._get_natoms_and_nframes(
            coords, atom_types, len(atom_types.shape) > 1
        )
        request_defs = self._get_request_defs(atomic)
        if "spin" not in kwargs or kwargs["spin"] is None:
            out = self._eval_func(self._eval_model, numb_test, natoms)(
                coords, cells, atom_types, fparam, aparam, request_defs
            )
        else:
            out = self._eval_func(self._eval_model_spin, numb_test, natoms)(
                coords,
                cells,
                atom_types,
                np.array(kwargs["spin"]),
                fparam,
                aparam,
                request_defs,
            )
        return dict(
            zip(
                [x.name for x in request_defs],
                out,
            )
        )

    def _get_request_defs(self, atomic: bool) -> list[OutputVariableDef]:
        """Get the requested output definitions.

        When atomic is True, all output_def are requested.
        When atomic is False, only energy (tensor), force, and virial
        are requested.

        Parameters
        ----------
        atomic : bool
            Whether to request the atomic output.

        Returns
        -------
        list[OutputVariableDef]
            The requested output definitions.
        """
        if atomic:
            return list(self.output_def.var_defs.values())
        else:
            return [
                x
                for x in self.output_def.var_defs.values()
                if x.category
                in (
                    OutputVariableCategory.OUT,
                    OutputVariableCategory.REDU,
                    OutputVariableCategory.DERV_R,
                    OutputVariableCategory.DERV_C_REDU,
                    OutputVariableCategory.DERV_R_DERV_R,
                )
            ]

    def _eval_func(self, inner_func: Callable, numb_test: int, natoms: int) -> Callable:
        """Wrapper method with auto batch size.

        Parameters
        ----------
        inner_func : Callable
            the method to be wrapped
        numb_test : int
            number of tests
        natoms : int
            number of atoms

        Returns
        -------
        Callable
            the wrapper
        """
        if self.auto_batch_size is not None:

            def eval_func(*args, **kwargs):
                return self.auto_batch_size.execute_all(
                    inner_func, numb_test, natoms, *args, **kwargs
                )

        else:
            eval_func = inner_func
        return eval_func

    def _get_natoms_and_nframes(
        self,
        coords: np.ndarray,
        atom_types: np.ndarray,
        mixed_type: bool = False,
    ) -> tuple[int, int]:
        if mixed_type:
            natoms = len(atom_types[0])
        else:
            natoms = len(atom_types)
        if natoms == 0:
            assert coords.size == 0
        else:
            coords = np.reshape(np.array(coords), [-1, natoms * 3])
        nframes = coords.shape[0]
        return natoms, nframes

    def _eval_model(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: np.ndarray,
        fparam: Optional[np.ndarray],
        aparam: Optional[np.ndarray],
        request_defs: list[OutputVariableDef],
    ):
        model = self.dp.to(DEVICE)
        prec = NP_PRECISION_DICT[RESERVED_PRECISION_DICT[GLOBAL_PT_FLOAT_PRECISION]]

        nframes = coords.shape[0]
        if len(atom_types.shape) == 1:
            natoms = len(atom_types)
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
        else:
            natoms = len(atom_types[0])

        coord_input = torch.tensor(
            coords.reshape([nframes, natoms, 3]).astype(prec),
            dtype=GLOBAL_PT_FLOAT_PRECISION,
            device=DEVICE,
        )
        type_input = torch.tensor(
            atom_types.astype(NP_PRECISION_DICT[RESERVED_PRECISION_DICT[torch.long]]),
            dtype=torch.long,
            device=DEVICE,
        )
        if cells is not None:
            box_input = torch.tensor(
                cells.reshape([nframes, 3, 3]).astype(prec),
                dtype=GLOBAL_PT_FLOAT_PRECISION,
                device=DEVICE,
            )
        else:
            box_input = None
        if fparam is not None:
            fparam_input = to_torch_tensor(
                fparam.reshape(nframes, self.get_dim_fparam())
            )
        else:
            fparam_input = None
        if aparam is not None:
            aparam_input = to_torch_tensor(
                aparam.reshape(nframes, natoms, self.get_dim_aparam())
            )
        else:
            aparam_input = None
        do_atomic_virial = any(
            x.category == OutputVariableCategory.DERV_C for x in request_defs
        )
        batch_output = model(
            coord_input,
            type_input,
            box=box_input,
            do_atomic_virial=do_atomic_virial,
            fparam=fparam_input,
            aparam=aparam_input,
        )
        if isinstance(batch_output, tuple):
            batch_output = batch_output[0]

        results = []
        for odef in request_defs:
            pt_name = self._OUTDEF_DP2BACKEND[odef.name]
            if pt_name in batch_output:
                shape = self._get_output_shape(odef, nframes, natoms)
                out = batch_output[pt_name].reshape(shape).detach().cpu().numpy()
                results.append(out)
            else:
                shape = self._get_output_shape(odef, nframes, natoms)
                results.append(
                    np.full(np.abs(shape), np.nan, dtype=prec)
                )  # this is kinda hacky
        return tuple(results)

    def _eval_model_spin(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: np.ndarray,
        spins: np.ndarray,
        fparam: Optional[np.ndarray],
        aparam: Optional[np.ndarray],
        request_defs: list[OutputVariableDef],
    ):
        model = self.dp.to(DEVICE)

        nframes = coords.shape[0]
        if len(atom_types.shape) == 1:
            natoms = len(atom_types)
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
        else:
            natoms = len(atom_types[0])

        coord_input = torch.tensor(
            coords.reshape([nframes, natoms, 3]),
            dtype=GLOBAL_PT_FLOAT_PRECISION,
            device=DEVICE,
        )
        type_input = torch.tensor(atom_types, dtype=torch.long, device=DEVICE)
        spin_input = torch.tensor(
            spins.reshape([nframes, natoms, 3]),
            dtype=GLOBAL_PT_FLOAT_PRECISION,
            device=DEVICE,
        )
        if cells is not None:
            box_input = torch.tensor(
                cells.reshape([nframes, 3, 3]),
                dtype=GLOBAL_PT_FLOAT_PRECISION,
                device=DEVICE,
            )
        else:
            box_input = None
        if fparam is not None:
            fparam_input = to_torch_tensor(
                fparam.reshape(nframes, self.get_dim_fparam())
            )
        else:
            fparam_input = None
        if aparam is not None:
            aparam_input = to_torch_tensor(
                aparam.reshape(nframes, natoms, self.get_dim_aparam())
            )
        else:
            aparam_input = None

        do_atomic_virial = any(
            x.category == OutputVariableCategory.DERV_C_REDU for x in request_defs
        )
        batch_output = model(
            coord_input,
            type_input,
            spin=spin_input,
            box=box_input,
            do_atomic_virial=do_atomic_virial,
            fparam=fparam_input,
            aparam=aparam_input,
        )
        if isinstance(batch_output, tuple):
            batch_output = batch_output[0]

        results = []
        for odef in request_defs:
            pt_name = self._OUTDEF_DP2BACKEND[odef.name]
            if pt_name in batch_output:
                shape = self._get_output_shape(odef, nframes, natoms)
                out = batch_output[pt_name].reshape(shape).detach().cpu().numpy()
                results.append(out)
            else:
                shape = self._get_output_shape(odef, nframes, natoms)
                results.append(
                    np.full(
                        np.abs(shape),
                        np.nan,
                        dtype=NP_PRECISION_DICT[
                            RESERVED_PRECISION_DICT[GLOBAL_PT_FLOAT_PRECISION]
                        ],
                    )
                )  # this is kinda hacky
        return tuple(results)

    def _get_output_shape(self, odef, nframes, natoms):
        if odef.category == OutputVariableCategory.DERV_C_REDU:
            # virial
            return [nframes, *odef.shape[:-1], 9]
        elif odef.category == OutputVariableCategory.REDU:
            # energy
            return [nframes, *odef.shape, 1]
        elif odef.category == OutputVariableCategory.DERV_C:
            # atom_virial
            return [nframes, *odef.shape[:-1], natoms, 9]
        elif odef.category == OutputVariableCategory.DERV_R:
            # force
            return [nframes, *odef.shape[:-1], natoms, 3]
        elif odef.category == OutputVariableCategory.OUT:
            # atom_energy, atom_tensor
            # Something wrong here?
            # return [nframes, *shape, natoms, 1]
            return [nframes, natoms, *odef.shape, 1]
        elif odef.category == OutputVariableCategory.DERV_R_DERV_R:
            return [nframes, 3 * natoms, 3 * natoms]
            # return [nframes, *odef.shape, 3 * natoms, 3 * natoms]
        else:
            raise RuntimeError("unknown category")

    def eval_typeebd(self) -> np.ndarray:
        """Evaluate output of type embedding network by using this model.

        Returns
        -------
        np.ndarray
            The output of type embedding network. The shape is [ntypes, o_size] or [ntypes + 1, o_size],
            where ntypes is the number of types, and o_size is the number of nodes
            in the output layer. If there are multiple type embedding networks,
            these outputs will be concatenated along the second axis.

        Raises
        ------
        KeyError
            If the model does not enable type embedding.

        See Also
        --------
        deepmd.pt.model.network.network.TypeEmbedNetConsistent :
            The type embedding network.
        """
        out = []
        for mm in self.dp.model["Default"].modules():
            if mm.original_name == TypeEmbedNetConsistent.__name__:
                out.append(mm(DEVICE))
        if not out:
            raise KeyError("The model has no type embedding networks.")
        typeebd = torch.cat(out, dim=1)
        return to_numpy_array(typeebd)

    def get_model_def_script(self) -> str:
        """Get model definition script."""
        return self.model_def_script

    def get_model_size(self) -> dict:
        """Get model parameter count.

        Returns
        -------
        dict
            A dictionary containing the number of parameters in the model.
            The keys are 'descriptor', 'fitting_net', and 'total'.
        """
        params_dict = dict(self.dp.named_parameters())
        sum_param_des = sum(
            params_dict[k].numel() for k in params_dict.keys() if "descriptor" in k
        )
        sum_param_fit = sum(
            params_dict[k].numel()
            for k in params_dict.keys()
            if "fitting" in k and "_networks" not in k
        )
        return {
            "descriptor": sum_param_des,
            "fitting-net": sum_param_fit,
            "total": sum_param_des + sum_param_fit,
        }

    def get_observed_types(self) -> dict:
        """Get observed types (elements) of the model during data statistics.

        Returns
        -------
        dict
            A dictionary containing the information of observed type in the model:
            - 'type_num': the total number of observed types in this model.
            - 'observed_type': a list of the observed types in this model.
        """
        observed_type_list = self.dp.model["Default"].get_observed_type_list()
        return {
            "type_num": len(observed_type_list),
            "observed_type": sort_element_type(observed_type_list),
        }

    def eval_descriptor(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: np.ndarray,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Evaluate descriptors by using this DP.

        Parameters
        ----------
        coords
            The coordinates of atoms.
            The array should be of size nframes x natoms x 3
        cells
            The cell of the region.
            If None then non-PBC is assumed, otherwise using PBC.
            The array should be of size nframes x 9
        atom_types
            The atom types
            The list should contain natoms ints
        fparam
            The frame parameter.
            The array can be of size :
            - nframes x dim_fparam.
            - dim_fparam. Then all frames are assumed to be provided with the same fparam.
        aparam
            The atomic parameter
            The array can be of size :
            - nframes x natoms x dim_aparam.
            - natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
            - dim_aparam. Then all frames and atoms are provided with the same aparam.

        Returns
        -------
        descriptor
            Descriptors.
        """
        model = self.dp.model["Default"]
        model.set_eval_descriptor_hook(True)
        self.eval(
            coords,
            cells,
            atom_types,
            atomic=False,
            fparam=fparam,
            aparam=aparam,
            **kwargs,
        )
        descriptor = model.eval_descriptor()
        model.set_eval_descriptor_hook(False)
        return to_numpy_array(descriptor)

    def eval_fitting_last_layer(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: np.ndarray,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Evaluate fitting before last layer by using this DP.

        Parameters
        ----------
        coords
            The coordinates of atoms.
            The array should be of size nframes x natoms x 3
        cells
            The cell of the region.
            If None then non-PBC is assumed, otherwise using PBC.
            The array should be of size nframes x 9
        atom_types
            The atom types
            The list should contain natoms ints
        fparam
            The frame parameter.
            The array can be of size :
            - nframes x dim_fparam.
            - dim_fparam. Then all frames are assumed to be provided with the same fparam.
        aparam
            The atomic parameter
            The array can be of size :
            - nframes x natoms x dim_aparam.
            - natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
            - dim_aparam. Then all frames and atoms are provided with the same aparam.

        Returns
        -------
        fitting
            Fitting output before last layer.
        """
        model = self.dp.model["Default"]
        model.set_eval_fitting_last_layer_hook(True)
        self.eval(
            coords,
            cells,
            atom_types,
            atomic=False,
            fparam=fparam,
            aparam=aparam,
            **kwargs,
        )
        fitting_net = model.eval_fitting_last_layer()
        model.set_eval_fitting_last_layer_hook(False)
        return to_numpy_array(fitting_net)
