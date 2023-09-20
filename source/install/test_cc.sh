set -e

if [ "$DP_VARIANT" = "cuda" ]; then
	CUDA_ARGS="-DUSE_CUDA_TOOLKIT=TRUE"
elif [ "$DP_VARIANT" = "rocm" ]; then
	CUDA_ARGS="-DUSE_ROCM_TOOLKIT=TRUE"
fi
#------------------

SCRIPT_PATH=$(dirname $(realpath -s $0))
NPROC=$(nproc --all)

#------------------

INSTALL_PREFIX=${SCRIPT_PATH}/../../dp_test
BUILD_TMP_DIR=${SCRIPT_PATH}/../build_tests
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
cmake -DINSTALL_TENSORFLOW=TRUE -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DTENSORFLOW_ROOT=${INSTALL_PREFIX} -DBUILD_TESTING:BOOL=TRUE -DLAMMPS_VERSION=stable_2Aug2023 ${CUDA_ARGS} ..
cmake --build . -j${NPROC}
cmake --install .

#------------------
# go to a subdirectory...
# TODO: detect directory of graph files
mkdir -p ${BUILD_TMP_DIR}/exec_tests
cd ${BUILD_TMP_DIR}/exec_tests

${INSTALL_PREFIX}/bin/runUnitTests_lib
${INSTALL_PREFIX}/bin/runUnitTests_cc
${INSTALL_PREFIX}/bin/runUnitTests_c
