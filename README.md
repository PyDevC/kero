# Kero-Sine

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PyDevC/kero)
![Status](https://img.shields.io/badge/status-pre--release-orange)

Kero-Sine is a SQL Query Engine built using MLIR, that has the capability to 
run and optimize queries on the CPU (and on the GPU in the future). It uses `SQLGlot` to 
handle syntax correctness and can handle different kinds of queries from 
different `dialects`. We use pyarrow to interact with existing databases and 
apply ETL on them, our main mode of data transfer is NumPy arrays which can be 
easily gathered from pyarrow tables.

> [!NOTE] Dependencies, full API documentation, and additional docs will be updated at the first release.

## Motivation

For my 4th-semester university database project, I decided to create a SQL Query 
Engine inspired by the Tensor Query Processor (TQP) which was written using PyTorch.
I was able to get the filtering system working. It was GPU-accelearted for CUDA and ROCm,
it worked with torch.compile and I had all the free resources from torch compile such
as operator fusion, jit compilation, Hardware acceleartion etc. But something was not 
right, I didn't write the full compiler by myself so I was wary that this might not work
out in the future. Fast forward to my 6th semester where we had to make a minor project. I 
decided to re-create it but with MLIR (because I wanted to get started with AI compilers).

As we all know AI compilers are mostly made using MLIR, so MLIR was my first choice 
when working on this project, It was hard and something that have only been done once
as Lingo-DB. But the idea came to mind that what if I created a SQL Query Engine which
executes data as if they were tensors, of course, there I knew I would face challenges with expensive
computations of data types such as strings, dates, floating point, etc.

The project finally came into existence when I made my first end-to-end compiler in MLIR.
I knew I would have to deal with tensors, so I decided to use Apache Arrow since it deals with
databases as columns instead of classic row based representation. 

Along the way I made so many mistakes and fixed them while learning better ways of doing things.

## Design Decisions
### DB Type
DB Types involve two types: DBTable, DBColumn
- **Table** types are used to transfer information about the movement of data.
- **Column** is for representing computation for a given operation inside filter.

Table type has an array of column attributes which helps in filtering, applying
filter on particular operation as well as knowing the data type of each column.
Since all the columns were needed to be converted to tensors in lowering pass, I decided 
to make it so that a single type shows the data movement and later that type can be expanded to
tensors without needing complex analysis.

#### Why is there no need for number of rows in Column type?
This is because the column doesn't need to know where it needs to apply the operation, this is 
handled in lowering by linalg, the pattern closely matches to arith way of comparing values,
making it easier to lower.

### Operations

- **Scan Operation:** is needed in the case when we create a new complex data type infuture, where we
    will pass whole database in the function argument allowing us to compute while retaining the complex
    relations between different tables.

- **Filter Operation:** is used to apply the where clause in the query and output a unknown size filtered table
- **Output Operation:** is pretty straight forward to apply, it just takes in a table and outputs only the column
    which are selected, making lowering much easier by only outputing the filtered tensors, that were selected.

## Current Limitations
- No JOINs, aggregates (COUNT, SUM), GROUP BY, ORDER BY, LIMIT
- Restricted support to Integer datatype
- No query optimization passes
- Sequential execution only
- Single-table queries only
- No Query Chaining

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
      %3 = db.cmpi gt, %arg1, %c20_i32 : (<i32>, i32) -> <i1>
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

## What's Next
- Float and String column type support.
- Output selects `*` directly, making IR cleaner and having to reducing large output overhead.
- Support Chaining Different operations together.
- Query Optimizations, combine different  queries
- Add float and string support to Execution Engine, and direct reading from pyarrow.
- Support Nested Query as well as complex query operations such as JOIN, AGG, etc.
- GPU execution path
