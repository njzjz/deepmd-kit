set -e

# You need to first run ./build_cc.sh
#------------------

SCRIPT_PATH=$(dirname $(realpath -s $0))
if [ -z "$INSTALL_PREFIX" ]
then
  INSTALL_PREFIX=$(realpath -s ${SCRIPT_PATH}/../../dp)
fi
mkdir -p ${INSTALL_PREFIX}
echo "Installing LAMMPS to ${INSTALL_PREFIX}"
NPROC=$(nproc --all)

#------------------
# copy lammps plugin
BUILD_TMP_DIR2=${SCRIPT_PATH}/../build
cd ${BUILD_TMP_DIR2}
make lammps

#------------------

BUILD_TMP_DIR=${SCRIPT_PATH}/../build_lammps
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
# download LAMMMPS
LAMMPS_VERSION=stable_29Oct2020
if [ ! -d "lammps-${LAMMPS_VERSION}" ]
then
	curl -L -o lammps.tar.gz https://github.com/lammps/lammps/archive/refs/tags/${LAMMPS_VERSION}.tar.gz
	tar vxzf lammps.tar.gz
fi
mkdir -p ${BUILD_TMP_DIR}/lammps-${LAMMPS_VERSION}/src/USER-DEEPMD
cp -r ${BUILD_TMP_DIR2}/USER-DEEPMD/* ${BUILD_TMP_DIR}/lammps-${LAMMPS_VERSION}/src/USER-DEEPMD

mkdir -p ${BUILD_TMP_DIR}/lammps-${LAMMPS_VERSION}/build
cd ${BUILD_TMP_DIR}/lammps-${LAMMPS_VERSION}/build

cmake -C ../cmake/presets/all_off.cmake -C ../src/USER-DEEPMD/deepmd.cmake -D PKG_KSPACE=ON -D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} ../cmake

make -j${NPROC}
make install

#------------------
echo "Congratulations! LAMMPS has been installed at ${INSTALL_PREFIX}"

