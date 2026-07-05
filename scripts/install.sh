#!/usr/bin/env bash

if [[ -z "$THIRDPARTY_LLVM_DIR" ]]; then
    echo "THIRDPARTY_LLVM_DIR variable not set"
    echo 'export THIRDPARTY_LLVM_DIR="$PWD/external/llvm-project"'
    echo "write above command to pass-in the env var"
    exit 0
fi

export MLIR_DIR="$THIRDPARTY_LLVM_DIR/build/lib/cmake/mlir"
export LLVM_DIR="$THIRDPARTY_LLVM_DIR/build/lib/cmake/llvm"

if [[ $# -gt 0 ]]; then 
    if [[ $1 == "install" ]]; then
        pip install . -v
    elif [[ $1 == "wheel" ]]; then
        python3 -m build --wheel
    fi
else
    pip install . -v
fi
