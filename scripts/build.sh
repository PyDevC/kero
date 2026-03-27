#!/usr/bin/env bash

if [[ -z "$THIRDPARTY_LLVM_DIR" ]]; then
    echo "THIRDPARTY_LLVM_DIR variable not set"
    echo 'export THIRDPARTY_LLVM_DIR="$PWD/external/llvm-project"'
    echo "write above command to pass-in the env var"
    exit 0
fi

BUILD_SYSTEM=Ninja
BUILD_TAG=ninja
BUILD_DIR=$(pwd)/build

mkdir -p $BUILD_DIR

pushd $BUILD_DIR

cmake .. -G $BUILD_SYSTEM \
      -DCMAKE_CXX_COMPILER="$(which clang++)" \
      -DCMAKE_C_COMPILER="$(which clang)" \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -DLLVM_LOCAL_RPATH=$INSTALL_DIR/lib \
      -DLLVM_PARALLEL_COMPILE_JOBS=16 \
      -DLLVM_PARALLEL_LINK_JOBS=5 \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_CCACHE_BUILD=ON \
      -DTHIRDPARTY_LLVM_DIR=$THIRDPARTY_LLVM_DIR \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \

sudo cmake --build . -j $(nproc)

popd
