# Kero-Sine

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PyDevC/kero)
![Status](https://img.shields.io/badge/status-pre--release-orange)

Kero-Sine is a SQL Query Engine built using MLIR, that has the capability to 
run and optimize queries on CPU (and on GPU in future). It uses `SQLGlot` to 
handle syntax correctness, it can handle different kinds of queries from 
different `dialects`. We use pyarrow to interact with existing databases and 
apply ETL on them, our main mode of data transfer is numpy arrays which can be 
easily gathered from pyarrow tables.

> [!NOTE] Dependencies, full API documentation, and additional docs will be updated at the first release.

## Prerequisites

- CMake (Min. version 3.20)
- Ninja build system (1.13)
- LLVM / MLIR (built from source — see instructions below) (Primarily from LLVM version 22.1.8)
- Python 3.11

## Build Instructions

### Build LLVM from Source

First install Dependencies
```bash
sudo apt install \
    cmake make ninja-build llvm lld clang ccache build-essential git -y
```

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

> [!TIP]: LLVM compilation is resource-intensive. If your system has limited RAM, reduce `DLLVM_PARALLEL_{COMPILE,LINK}_JOBS` to `3` or `5` to avoid system crash.

### Install/Build Kero-Sine from source

```bash
pip install -r requirements.txt
export THIRDPARTY_LLVM_DIR="/path/to/llvm-project" # path where you cloned the repo

bash scripts/py_build.sh
```

## Example query
```python
from kero.arrow.samples import all_number_dataset
from kero.engine import Parser, codegen
from kero.engine.execution import KeroEngine

dataset = all_number_dataset(size=10_000)
parser = Parser(dataset)

query = "SELECT * FROM employee WHERE age > 20"
query_ast = parser.parse(query)
irgen = codegen.IRGen("get_aged_employee", query_ast)
irgen.emit_ir()
codegen.db_to_llvm_lowering(irgen.module, irgen.context)

exe = KeroEngine(irgen.module, irgen.context)
exe.configure_outputs([0, 1, 2])
print(irgen.func_result_num)
results = exe.execute("get_aged_employee", dataset, ["employee"])
numpy_results = exe.results_to_numpy(results)
for id, array in numpy_results.items():
    print(array)
```
