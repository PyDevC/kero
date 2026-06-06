#!/usr/bin/env bash

if [[ -z "$THIRDPARTY_LLVM_DIR" ]]; then
    echo "THIRDPARTY_LLVM_DIR variable not set"
    echo 'export THIRDPARTY_LLVM_DIR="$PWD/external/llvm-project"'
    echo "write above command to pass-in the env var"
    exit 0
fi

export LLVM_DIR="$THIRDPARTY_LLVM_DIR/build/lib/cmake/llvm"

python3 setup.py bdist_wheel
pip uninstall kero -y
pip install dist/kero-0.1.0-cp312-cp312-linux_x86_64.whl
