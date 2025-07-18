# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Optional,
    Union,
)

import numpy as np

from deepmd.backend.backend import (
    Backend,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
)
from deepmd.utils.batch_size import (
    AutoBatchSize,
)

if TYPE_CHECKING:
    import ase.neighborlist


class DeepEvalBackend(ABC):
    """Low-level Deep Evaluator interface.

    Backends should inherbit implement this interface. High-level interface
    will be built on top of this.

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    *args : list
        Positional arguments.
    auto_batch_size : bool or int or AutoBatchSize, default: True
        If True, automatic batch size will be used. If int, it will be used
        as the initial batch size.
    neighbor_list : ase.neighborlist.NewPrimitiveNeighborList, optional
        The ASE neighbor list class to produce the neighbor list. If None, the
        neighbor list will be built natively in the model.
    **kwargs : dict
        Keyword arguments.
    """

    _OUTDEF_DP2BACKEND: ClassVar[dict] = {
        "energy": "atom_energy",
        "energy_redu": "energy",
        "energy_derv_r": "force",
        "energy_derv_r_mag": "force_mag",
        "energy_derv_c": "atom_virial",
        "energy_derv_c_mag": "atom_virial_mag",
        "energy_derv_c_redu": "virial",
        "polar": "polar",
        "polar_redu": "global_polar",
        "polar_derv_r": "force",
        "polar_derv_c": "atom_virial",
        "polar_derv_c_redu": "virial",
        "dipole": "dipole",
        "dipole_redu": "global_dipole",
        "dipole_derv_r": "force",
        "dipole_derv_c": "atom_virial",
        "dipole_derv_c_redu": "virial",
        "dos": "atom_dos",
        "dos_redu": "dos",
        "mask_mag": "mask_mag",
        "mask": "mask",
        # old models in v1
        "global_polar": "global_polar",
        "wfc": "wfc",
        "energy_derv_r_derv_r": "hessian",
    }

    @abstractmethod
    def __init__(
        self,
        model_file: str,
        output_def: ModelOutputDef,
        *args: Any,
        auto_batch_size: Union[bool, int, AutoBatchSize] = True,
        neighbor_list: Optional["ase.neighborlist.NewPrimitiveNeighborList"] = None,
        **kwargs: Any,
    ) -> None:
        pass

    def __new__(cls, model_file: str, *args, **kwargs):
        if cls is DeepEvalBackend:
            backend = Backend.detect_backend_by_model(model_file)
            return super().__new__(backend().deep_eval)
        return super().__new__(cls)

    @abstractmethod
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

    @abstractmethod
    def get_rcut(self) -> float:
        """Get the cutoff radius of this model."""

    @abstractmethod
    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""

    @abstractmethod
    def get_type_map(self) -> list[str]:
        """Get the type map (element name of the atom types) of this model."""

    @abstractmethod
    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this DP."""

    @abstractmethod
    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this DP."""

    def eval_descriptor(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: np.ndarray,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        efield: Optional[np.ndarray] = None,
        mixed_type: bool = False,
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
        efield
            The external field on atoms.
            The array should be of size nframes x natoms x 3
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.

        Returns
        -------
        descriptor
            Descriptors.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def eval_typeebd(self) -> np.ndarray:
        """Evaluate output of type embedding network by using this model.

        Returns
        -------
        np.ndarray
            The output of type embedding network. The shape is [ntypes, o_size],
            where ntypes is the number of types, and o_size is the number of nodes
            in the output layer.

        Raises
        ------
        KeyError
            If the model does not enable type embedding.
        """
        raise NotImplementedError

    def _check_mixed_types(self, atom_types: np.ndarray) -> bool:
        """Check if atom types of all frames are the same.

        Traditional descriptors like se_e2_a requires all the frames to
        have the same atom types.

        Parameters
        ----------
        atom_types : np.ndarray
            The atom types of all frames, in shape nframes * natoms.
        """
        if np.count_nonzero(atom_types[0] == -1) > 0:
            # assume mixed_types if there are virtual types, even when
            # the atom types of all frames are the same
            return False
        return np.all(np.equal(atom_types, atom_types[0])).item()

    @property
    @abstractmethod
    def model_type(self) -> type["DeepEval"]:
        """The the evaluator of the model type."""

    @abstractmethod
    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """

    def get_numb_dos(self) -> int:
        """Get the number of DOS."""
        raise NotImplementedError

    def get_has_efield(self) -> bool:
        """Check if the model has efield."""
        return False

    def get_has_spin(self) -> bool:
        """Check if the model has spin atom types."""
        return False

    def get_has_hessian(self):
        """Check if the model has hessian."""
        return False

    def get_var_name(self) -> str:
        """Get the name of the fitting property."""
        raise NotImplementedError

    @abstractmethod
    def get_ntypes_spin(self) -> int:
        """Get the number of spin atom types of this model. Only used in old implement."""

    def get_model_def_script(self) -> dict:
        """Get model definition script."""
        raise NotImplementedError("Not implemented in this backend.")

    def get_model_size(self) -> dict:
        """Get model parameter count."""
        raise NotImplementedError("Not implemented in this backend.")

    def get_observed_types(self) -> dict:
        """Get observed types (elements) of the model during data statistics."""
        raise NotImplementedError("Not implemented in this backend.")


class DeepEval(ABC):
    """High-level Deep Evaluator interface.

    The specific DeepEval, such as DeepPot and DeepTensor, should inherit
    from this class. This class provides a high-level interface on the top
    of the low-level interface.

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    *args : list
        Positional arguments.
    auto_batch_size : bool or int or AutoBatchSize, default: True
        If True, automatic batch size will be used. If int, it will be used
        as the initial batch size.
    neighbor_list : ase.neighborlist.NewPrimitiveNeighborList, optional
        The ASE neighbor list class to produce the neighbor list. If None, the
        neighbor list will be built natively in the model.
    **kwargs : dict
        Keyword arguments.
    """

    def __new__(cls, model_file: str, *args, **kwargs):
        if cls is DeepEval:
            deep_eval = DeepEvalBackend(
                model_file,
                ModelOutputDef(FittingOutputDef([])),
                *args,
                **kwargs,
            )
            return super().__new__(deep_eval.model_type)
        return super().__new__(cls)

    def __init__(
        self,
        model_file: str,
        *args: Any,
        auto_batch_size: Union[bool, int, AutoBatchSize] = True,
        neighbor_list: Optional["ase.neighborlist.NewPrimitiveNeighborList"] = None,
        **kwargs: Any,
    ) -> None:
        self.deep_eval = DeepEvalBackend(
            model_file,
            self.output_def,
            *args,
            auto_batch_size=auto_batch_size,
            neighbor_list=neighbor_list,
            **kwargs,
        )
        if self.deep_eval.get_has_spin() and hasattr(self, "output_def_mag"):
            self.deep_eval.output_def = self.output_def_mag

    @property
    @abstractmethod
    def output_def(self) -> ModelOutputDef:
        """Returns the output variable definitions."""

    def get_rcut(self) -> float:
        """Get the cutoff radius of this model."""
        return self.deep_eval.get_rcut()

    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""
        return self.deep_eval.get_ntypes()

    def get_type_map(self) -> list[str]:
        """Get the type map (element name of the atom types) of this model."""
        return self.deep_eval.get_type_map()

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this DP."""
        return self.deep_eval.get_dim_fparam()

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this DP."""
        return self.deep_eval.get_dim_aparam()

    def _get_natoms_and_nframes(
        self,
        coords: np.ndarray,
        atom_types: np.ndarray,
        mixed_type: bool = False,
    ) -> tuple[int, int]:
        if mixed_type or atom_types.ndim > 1:
            natoms = len(atom_types[0])
        else:
            natoms = len(atom_types)
        if natoms == 0:
            assert coords.size == 0
        else:
            coords = np.reshape(np.array(coords), [-1, natoms * 3])
        nframes = coords.shape[0]
        return natoms, nframes

    def _expande_atype(self, atype: np.ndarray, nframes: int, mixed_type: bool):
        if not mixed_type:
            atype = np.tile(atype.reshape(1, -1), (nframes, 1))
        return atype

    def eval_descriptor(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: np.ndarray,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        mixed_type: bool = False,
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
        efield
            The external field on atoms.
            The array should be of size nframes x natoms x 3
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.

        Returns
        -------
        descriptor
            Descriptors.
        """
        (
            coords,
            cells,
            atom_types,
            fparam,
            aparam,
            nframes,
            natoms,
        ) = self._standard_input(coords, cells, atom_types, fparam, aparam, mixed_type)
        descriptor = self.deep_eval.eval_descriptor(
            coords,
            cells,
            atom_types,
            fparam=fparam,
            aparam=aparam,
            **kwargs,
        )
        return descriptor

    def eval_fitting_last_layer(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: np.ndarray,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        mixed_type: bool = False,
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
        efield
            The external field on atoms.
            The array should be of size nframes x natoms x 3
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.

        Returns
        -------
        fitting
            Fitting output before last layer.
        """
        (
            coords,
            cells,
            atom_types,
            fparam,
            aparam,
            nframes,
            natoms,
        ) = self._standard_input(coords, cells, atom_types, fparam, aparam, mixed_type)
        fitting = self.deep_eval.eval_fitting_last_layer(
            coords,
            cells,
            atom_types,
            fparam=fparam,
            aparam=aparam,
            **kwargs,
        )
        return fitting

    def eval_typeebd(self) -> np.ndarray:
        """Evaluate output of type embedding network by using this model.

        Returns
        -------
        np.ndarray
            The output of type embedding network. The shape is [ntypes, o_size],
            where ntypes is the number of types, and o_size is the number of nodes
            in the output layer.

        Raises
        ------
        KeyError
            If the model does not enable type embedding.

        See Also
        --------
        deepmd.tf.utils.type_embed.TypeEmbedNet : The type embedding network.

        Examples
        --------
        Get the output of type embedding network of `graph.pb`:

        >>> from deepmd.infer import DeepPotential
        >>> dp = DeepPotential("graph.pb")
        >>> dp.eval_typeebd()
        """
        return self.deep_eval.eval_typeebd()

    def _standard_input(self, coords, cells, atom_types, fparam, aparam, mixed_type):
        coords = np.array(coords)
        if cells is not None:
            cells = np.array(cells)
        atom_types = np.array(atom_types, dtype=np.int32)
        if fparam is not None:
            fparam = np.array(fparam)
        if aparam is not None:
            aparam = np.array(aparam)
        natoms, nframes = self._get_natoms_and_nframes(coords, atom_types, mixed_type)
        atom_types = self._expande_atype(atom_types, nframes, mixed_type)
        coords = coords.reshape(nframes, natoms, 3)
        if cells is not None:
            cells = cells.reshape(nframes, 3, 3)
        if fparam is not None:
            fdim = self.get_dim_fparam()
            if fparam.size == nframes * fdim:
                fparam = np.reshape(fparam, [nframes, fdim])
            elif fparam.size == fdim:
                fparam = np.tile(fparam.reshape([-1]), [nframes, 1])
            else:
                raise RuntimeError(
                    f"got wrong size of frame param, should be either {nframes} x {fdim} or {fdim}"
                )
        if aparam is not None:
            fdim = self.get_dim_aparam()
            if aparam.size == nframes * natoms * fdim:
                aparam = np.reshape(aparam, [nframes, natoms * fdim])
            elif aparam.size == natoms * fdim:
                aparam = np.tile(aparam.reshape([-1]), [nframes, 1])
            elif aparam.size == fdim:
                aparam = np.tile(aparam.reshape([-1]), [nframes, natoms])
            else:
                raise RuntimeError(
                    f"got wrong size of frame param, should be either {nframes} x {natoms} x {fdim} or {natoms} x {fdim} or {fdim}"
                )
        return coords, cells, atom_types, fparam, aparam, nframes, natoms

    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.deep_eval.get_sel_type()

    def _get_sel_natoms(self, atype) -> int:
        return np.sum(np.isin(atype, self.get_sel_type()).astype(int))

    @property
    def has_efield(self) -> bool:
        """Check if the model has efield."""
        return self.deep_eval.get_has_efield()

    @property
    def has_spin(self) -> bool:
        """Check if the model has spin."""
        return self.deep_eval.get_has_spin()

    @property
    def has_hessian(self) -> bool:
        """Check if the model has hessian."""
        return self.deep_eval.get_has_hessian()

    def get_ntypes_spin(self) -> int:
        """Get the number of spin atom types of this model. Only used in old implement."""
        return self.deep_eval.get_ntypes_spin()

    def get_model_def_script(self) -> dict:
        """Get model definition script."""
        return self.deep_eval.get_model_def_script()

    def get_model_size(self) -> dict:
        """Get model parameter count."""
        return self.deep_eval.get_model_size()

    def get_observed_types(self) -> dict:
        """Get observed types (elements) of the model during data statistics."""
        return self.deep_eval.get_observed_types()
