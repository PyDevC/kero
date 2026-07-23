# Kero-Sine

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PyDevC/kero)
![Status](https://img.shields.io/badge/status-pre--release-orange)

Kero-Sine is a SQL compiler, designed to run SQL query from Apache Arrow Table
and help user train ML models with minimal code spent on infra than on actual
ML code. It adopts the tensor computation model from the Tensor Query Processor
(TQP), utilizing MLIR Infrastructure to allow compiled code to run on different
architectures. Gaining both performance, usability, and device portability.

> [!NOTE] Dependencies, full API documentation, and additional docs will be updated at the first release.

## Motivation

SQL driven ML pipelines, high amount of code is spent on maintaining
infrastructure, converting data to different formats to use different libraries
for analysis and transformations, most of these pipelines are slow and error
prone. Having a easy to work pipeline allows developers to spend more time on
model training rather than on initial pipeline setup. 

Some of the database queries are required to be run on GPU, rather than on CPU,
for large amount of datasets, for these systems we lack the heterogeneity of
devices we can run code on limiting usage of such applications on wide variety
of devices. Our Library aims to provide both of these worlds where usability,
performance, and device portability is not compromised.

## Dialect Design Motivation

Kero-Sine uses custom dialect named db dialect to represent relational queries
into simple operations that can executed in order to mimic results from a SQL
compiler like PostgreSQL. We aim to lower db dialect to Tensor and Linalg mainly
allowing for further optimizations such as tiling, directly from MLIR existing
MLIR passes, without the need to maintain custom MLIR passes.

## Existing Work
There are several applications designed to run SQL queries, some adopt row by
row execution, some adopt column by column, Some use MLIR under the hood, some
use CUDA as it's backend.

In this wide variety of applications few are worth nothing which intersects with
the scope of Kero-Sine.

### Lingo-DB
Lingo-DB is a data processing compiler developed by Technical University of 
Munich. It uses MLIR infrastructure to develop highly optimized pipelines, which
can perfomant code easily on different device architectures, Lingo-DB allows for
Cross-domin optimizations, JIT compilation, etc.

Lingo-DB works on the model of column by column execution, with the aim to
directly adapt Apache Arrow memory format. It provides multi layer IR design for
investigating heterogenous hardware for data processing.

### Heavy DB
Heavy DB is a SQL query engine build using rapids API from CUDA and designed to
run SQL query fast on GPU. Running queries as GPU kernels allow easy transfer
for different applications such as ML model training, query large databases with
high number of complex queries.

Heavy DB aims to provide both performance and usability but fails at the part of
device heterogeneity, making it unsuitable to run same code on different devices.

### Tensor Query Processor (TQP)
TQP is a research tool that runs the SQL query using already available PyTorch
infrastructure. It uses tensor computation model, treating each column as tensor
and applying already available PyTorch Operations to mimic SQL query execution.
Written purely using PyTorch allows for hardware portability and performant code
when utilized with PyTorch Compiler Stack, allowing for optimized GPU kernels,
Operator fusion, etc. While this project passes all the tests that kero aims to
achieve, it is proprietary research project which is not available for public use.

## Dialect Design

Kero-Sine uses MLIR infrastructure, to compile query from it's custom dialect to
tensor and linalg dialect. Most of the heavy lifting is done by MLIR, allowing
for better code maintenance, and far more available optimizations out of the box.

### DB dialect
DB dialect tries to adapt the approaches of Heavy DB and TQP, and treat Table
columns as 1-D tensors, and generate separate code for filter execution and for
applying mask generated (mask is a tensor that tells which row in tensor are not
available and which one are).

#### DB Types
- DBTable: shows the flow of data for each operation.
- DBColumn: shows the computation for each operation.

#### Operations
- ScanOp: acts as a placeholder for the table that is inputed.
- FilterOp: Applies where clause to on the table.
- OutputOp: extracts the selected columns from the table and create a new table.

#### Filter Op in detail
DB Filter operation provides a region which is used to apply WHERE clause from
sql query and generate a new table out of it. DB Filter Operation consists of
operations such as `db.cmp` which is used to apply comparision between a column
and a constant for query like `salary > 100`, these operations directly maps to
arith operations making it easier to generate IR out of it.

Lowering DB Filter to Linalg generic to create a mask:

Linalg generic provides interface to apply element-wise ops for given tensors
and generate new tensors out of it, making them ideal choice for generating mask
which can be applied on different whole table.

Lowering DB Filter to Linalg generic allows for lowering to different forms such
as `--convert-linalg-to-parallel-loops`, `--convert-scf-to-openmp`, getting us
free performance out of already optimized passes.

## Compilation Pipeline

- Parsing: Kero uses SQLGlot to parse SQL queries, and generates a new DB ast with type resolved nodes identical to db dialect operations and types.
- Codegen: Uses MLIR Python bindings to add correct operation to module, from db dialect.
- MLIR IR: Codegen produces a IR from the db dialect. It is a custom MLIR dialect which lies close to SQL semantics.
- Lowering to tensor, Linalg and scf: kero contains a custom lowering pass to lower db dialect to tensor, linalg, and scf dialect.
- Lowering to LLVM: Since all the operations are converted to upstream MLIR dialects, we can progressively lower it to LLVM dialect using passes such as one-shot-bufferize, convert-linalg-to-loops, etc. other standard passes.

## Current Limitations

- Limited Query Scope (db.join, db.sum, db.count, etc), (in future we can add support for nested queries but requires it requires analysis to merge, filter regions).
- No GPU execution.
- Inefficient apply mask application (can be applied using Prefix sum algorithm to generate new tensors).
- No support for operation chaining.
- Limited to int and float (can be done Limited to int and float adopted HeavyDB String serialization techiniques).
- Column to literal comparison only (can support column with column comparision without much code change since it will just take in input a column row element instead of a constant).

## Future Scope

Kero-Sine currently has several limitations, which can be harmful for performance, memory usage, as well as API UX. In future, it is aimed to overcome those limitations by introducing various new features.

- Adding new custom dialect, kero, for representing computations such as implementing String dictionaries to n-bit integers (depending on the uniqueness and num rows), adding custom C++ libs for date manipulation or string manipulation that can be called via shared_lib functions inside the JIT Engine.
- Adding more operations like JOIN, Aggregation, or nested queries, etc.
- Accept whole database as input instead of just single table and extract table from dataset using scan operation (it’s decided to get the dataset from external function call instead of passing it directly to the jit arguments).
- Allow for Operation chaining if possible and apply canonicalization to group up different operations such as two filter operations mixed to one to produce single linalg generic op instead of two.
- Apply prefix-sum algorithm for applying mask to the columns.
- Add GPU lowering to dialects such as NVVM or ROCDL.

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
This is example bare minimum code required for computations.
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
exe.configure_outputs([i for i in range(irgen.func_result_num)])
print(irgen.func_result_num)
results = exe.execute("get_aged_employee", dataset, ["employee"])
numpy_results = exe.results_to_numpy(results)
for id, array in numpy_results.items():
    print(array)
```

The Above Query will give us IR form:
```mlir
!table = !db.table<3, 10000 : [<"age", i32, 10000>, <"salary", i32, 10000>, <"spendings", i32, 10000>]>
!new_table = !db.table<3, -1 : [<"age", i32, -1>, <"salary", i32, -1>, <"spendings", i32, -1>]>

module {
  func.func @get_aged_employee(%arg0: !table) -> !new_table attributes {llvm.emit_c_interface} {
    %0 = db.scan %arg0 : !table -> !table
    %1 = db.filter %0 : !table {
    ^bb0(%arg1: !db.column<i32>, %arg2: !db.column<i32>, %arg3: !db.column<i32>):
      %c20_i32 = arith.constant 20 : i32
      %3 = db.cmp gt, %arg1, %c20_i32 : (<i32>, i32) -> <i1>
      db.filter_yield %3 : <i1>
    } -> (!new_table)
    %2 = db.output {select = ["age", "salary", "spendings"]} %1 : !new_table -> !new_table
    return %2 : !new_table
  }
}
```

The Above IR will get lowered to linalg, tensor and STD dialects:
```mlir
#map = affine_map<(d0) -> (d0)>
module {
  func.func @get_aged_employee(%arg0: tensor<10000xi32>, %arg1: tensor<10000xi32>, %arg2: tensor<10000xi32>) -> (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<10000xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2 : tensor<10000xi32>, tensor<10000xi32>, tensor<10000xi32>) outs(%0 : tensor<10000xi1>) {
    ^bb0(%in: i32, %in_2: i32, %in_3: i32, %out: i1):
      %c20_i32 = arith.constant 20 : i32
      %7 = arith.cmpi sgt, %in, %c20_i32 : i32
      linalg.yield %7 : i1
    } -> tensor<10000xi1>
    %c0 = arith.constant 0 : index
    %c10000 = arith.constant 10000 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %2 = scf.for %arg3 = %c0 to %c10000 step %c1 iter_args(%arg4 = %c0_0) -> (index) {
      %extracted = tensor.extract %1[%arg3] : tensor<10000xi1>
      %7 = scf.if %extracted -> (index) {
        %8 = arith.addi %arg4, %c1 : index
        scf.yield %8 : index
      } else {
        scf.yield %arg4 : index
      }
      scf.yield %7 : index
    }
    %3 = tensor.empty(%2) : tensor<?xi32>
    %4 = tensor.empty(%2) : tensor<?xi32>
    %5 = tensor.empty(%2) : tensor<?xi32>
    %c0_1 = arith.constant 0 : index
    %6:4 = scf.for %arg3 = %c0 to %c10000 step %c1 iter_args(%arg4 = %3, %arg5 = %4, %arg6 = %5, %arg7 = %c0_1) -> (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, index) {
      %extracted = tensor.extract %1[%arg3] : tensor<10000xi1>
      %7:4 = scf.if %extracted -> (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, index) {
        %extracted_2 = tensor.extract %arg0[%arg3] : tensor<10000xi32>
        %inserted = tensor.insert %extracted_2 into %arg4[%arg7] : tensor<?xi32>
        %extracted_3 = tensor.extract %arg1[%arg3] : tensor<10000xi32>
        %inserted_4 = tensor.insert %extracted_3 into %arg5[%arg7] : tensor<?xi32>
        %extracted_5 = tensor.extract %arg2[%arg3] : tensor<10000xi32>
        %inserted_6 = tensor.insert %extracted_5 into %arg6[%arg7] : tensor<?xi32>
        %8 = arith.addi %arg7, %c1 : index
        scf.yield %inserted, %inserted_4, %inserted_6, %8 : tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, index
      } else {
        scf.yield %arg4, %arg5, %arg6, %arg7 : tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, index
      }
      scf.yield %7#0, %7#1, %7#2, %7#3 : tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, index
    }
    return %6#0, %6#1, %6#2 : tensor<?xi32>, tensor<?xi32>, tensor<?xi32>
  }
}
```
