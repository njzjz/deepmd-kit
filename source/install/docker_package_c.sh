set -e

SCRIPT_PATH=$(dirname $(realpath -s $0))

wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip -O libtorch.zip
unzip libtorch.zip
rm -f libtorch.zip

docker run --rm -v ${SCRIPT_PATH}/../..:/root/deepmd-kit -w /root/deepmd-kit \
   -v libtorch:/root/libtorch \
	tensorflow/build:${TENSORFLOW_BUILD_VERSION:-2.16}-python3.11 \
	/bin/sh -c "pip install \"tensorflow${TENSORFLOW_VERSION}\" cmake \
            && git config --global --add safe.directory /root/deepmd-kit \
            && cd /root/deepmd-kit/source/install \
            && CC=/dt9/usr/bin/gcc \
               CXX=/dt9/usr/bin/g++ \
               CMAKE_PREFIX_PATH=/root/libtorch \
               /bin/sh package_c.sh"
