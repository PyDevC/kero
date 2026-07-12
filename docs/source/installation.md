# Installation

## Prerequisites

- CMake >= 3.20
- Ninja build system >= 1.13
- LLVM / MLIR built from source (LLVM 22.1.8)
- Python >= 3.9

### Install System Dependencies

```bash
sudo apt install \
    cmake make ninja-build llvm lld clang ccache build-essential git -y
```

## Build LLVM from Source

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

mkdir -p build
cmake -G Ninja -Bbuild llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang \
    -DLLVM_CCACHE_BUILD=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_PARALLEL_LINK_JOBS=3 \
    -DLLVM_PARALLEL_TABLEGEN_JOBS=5 \
    -DLLVM_TARGETS_TO_BUILD="Native" \
    -DLLVM_USE_LINKER=lld \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DPython_FIND_VIRTUALENV=ONLY

cmake --build build
```

```{tip}
LLVM compilation is resource-intensive. Reduce `DLLVM_PARALLEL_LINK_JOBS` and `DLLVM_PARALLEL_TABLEGEN_JOBS` if your system has limited RAM.
```

## Install Kero-Sine

```bash
pip install -r requirements.txt
export THIRDPARTY_LLVM_DIR="/path/to/llvm-project"

bash scripts/py_build.sh
```

Set `THIRDPARTY_LLVM_DIR` to the path where you cloned `llvm-project`.
