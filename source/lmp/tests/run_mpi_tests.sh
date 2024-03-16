#!/bin/bash
# For unclear reason, we can only start one LAMMPS instance at a time,
# otherwise, "stack smashing detected".
SCRIPT_PATH=$(dirname $(realpath -s $0))
export DP_TEST_REUSE_MODELS=1

# Below two throw Segmentation fault.
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_deeptensor.py::test_compute_deeptensor_atom
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_deeptensor.py::test_compute_deeptensor_atom_si
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_dplr.py::test_pair_deepmd_sr
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_dplr.py::test_pair_deepmd_sr_virial
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_dplr.py::test_pair_deepmd_lr
# AttributeError: 'list' object has no attribute 'strip'
# perhaps an upstream bug
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_dplr.py::test_pair_deepmd_lr_efield_constant
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_dplr.py::test_pair_deepmd_lr_efield_variable
# below hangs for ever
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_dplr.py::test_min_dplr
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_dplr.py::test_pair_deepmd_lr_type_map
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_dplr.py::test_pair_deepmd_lr_si
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps.py::test_pair_deepmd
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps.py::test_pair_deepmd_virial
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps.py::test_pair_deepmd_model_devi
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps.py::test_pair_deepmd_model_devi_virial
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps.py::test_pair_deepmd_model_devi_atomic_relative
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps.py::test_pair_deepmd_model_devi_atomic_relative_v
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps.py::test_pair_deepmd_type_map
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps.py::test_pair_deepmd_real
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps.py::test_pair_deepmd_virial_real
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps.py::test_pair_deepmd_model_devi_real
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps.py::test_pair_deepmd_model_devi_virial_real
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps.py::test_pair_deepmd_model_devi_atomic_relative_real
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps.py::test_pair_deepmd_model_devi_atomic_relative_v_real
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps.py::test_pair_deepmd_si
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_3types.py::test_pair_deepmd
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_3types.py::test_pair_deepmd_virial
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_3types.py::test_pair_deepmd_model_devi
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_3types.py::test_pair_deepmd_model_devi_virial
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_3types.py::test_pair_deepmd_model_devi_atomic_relative
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_3types.py::test_pair_deepmd_model_devi_atomic_relative_v
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_3types.py::test_pair_deepmd_type_map
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_faparam.py::test_pair_deepmd
mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_faparam.py::test_pair_deepmd_virial
# All pt throws errors:
# terminate called after throwing an instance of 'c10::Error'
# what():  torch.cat(): expected a non-empty list of Tensors
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_pt.py::test_pair_deepmd
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_pt.py::test_pair_deepmd_virial
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_pt.py::test_pair_deepmd_model_devi
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_pt.py::test_pair_deepmd_model_devi_virial
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_pt.py::test_pair_deepmd_model_devi_atomic_relative
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_pt.py::test_pair_deepmd_model_devi_atomic_relative_v
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_pt.py::test_pair_deepmd_type_map
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_pt.py::test_pair_deepmd_real
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_pt.py::test_pair_deepmd_virial_real
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_pt.py::test_pair_deepmd_model_devi_real
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_pt.py::test_pair_deepmd_model_devi_virial_real
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_pt.py::test_pair_deepmd_model_devi_atomic_relative_real
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_pt.py::test_pair_deepmd_model_devi_atomic_relative_v_real
# mpirun -n 2 pytest -q ${SCRIPT_PATH}/test_lammps_pt.py::test_pair_deepmd_si
