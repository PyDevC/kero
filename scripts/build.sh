#!/usr/bin/env bash

if [[ -z "$THIRDPARTY_LLVM_DIR" ]]; then
    echo "THIRDPARTY_LLVM_DIR variable not set"
    echo 'export THIRDPARTY_LLVM_DIR="$PWD/external/llvm-project"'
    echo "write above command to pass-in the env var"
    exit 0
fi

export MLIR_DIR="$THIRDPARTY_LLVM_DIR/build/lib/cmake/mlir"
export LLVM_DIR="$THIRDPARTY_LLVM_DIR/build/lib/cmake/llvm"

BUILD_SYSTEM=Ninja
BUILD_TAG=ninja
BUILD_DIR=$(pwd)/build
BUILD_TYPE=Release

if [[ "$1" == "--build-type" ]]; then
    BUILD_TYPE=$2
fi

mkdir -p $BUILD_DIR

pushd $BUILD_DIR

cmake .. -G $BUILD_SYSTEM \
      -DCMAKE_CXX_COMPILER="$(which clang++)" \
      -DCMAKE_C_COMPILER="$(which clang)" \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=_keroEngine \
      -DLLVM_PARALLEL_COMPILE_JOBS=$(nproc) \
      -DLLVM_PARALLEL_LINK_JOBS=5 \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_CCACHE_BUILD=ON \
      -DTHIRDPARTY_LLVM_DIR=$THIRDPARTY_LLVM_DIR \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \

ninja -j $(nproc)

popd
