# Getting Started

## Creating a Dataset

Kero-Sine uses PyArrow tables as the data source. You can create a dataset manually or use the built-in sample generators:

```python
from kero.arrow.samples import all_number_dataset

dataset = all_number_dataset(size=10_000)
```

## Running a Query

The query pipeline has three steps: **parse**, **compile**, and **execute**.

```python
from kero.engine import Parser, codegen
from kero.engine.execution import KeroEngine

# 1. Parse the SQL query into an internal AST
parser = Parser(dataset)
query_ast = parser.parse("SELECT * FROM employee WHERE age > 20")

# 2. Compile the AST to MLIR IR and lower to LLVM
irgen = codegen.IRGen("get_aged_employee", query_ast)
irgen.emit_ir()
codegen.db_to_llvm_lowering(irgen.module, irgen.context)

# 3. Execute the compiled query
exe = KeroEngine(irgen.module, irgen.context)
exe.configure_outputs([0, 1, 2])
results = exe.execute("get_aged_employee", dataset, ["employee"])
```

## Extracting Results

Results are returned as numpy arrays keyed by output index:

```python
numpy_results = exe.results_to_numpy(results)
for id, array in numpy_results.items():
    print(array)
```

## Using Tensor Abstractions

Kero-Sine provides typed tensor classes for representing relational columns as PyTorch tensors (requires `torch`):

```python
from kero import NumTensor, StrTensor, DateTensor
```

These are useful when integrating query results into ML pipelines.
