import pyarrow as pa
import torch
from kero.engine.parser import Parser
from kero.engine.compiler import Compiler

def test_run_kero_end_to_end():
    # 1. Dataset Generation: Create synthetic Apache Arrow data
    data = {
        "user_id": [101, 102, 103, 104, 105],
        "age": [25, 42, 18, 36, 29],
        "score": [88.5, 92.0, 75.2, 81.0, 95.5]
    }
    table = pa.Table.from_pydict(data)

    # 2. Setup: Initialize MLIR compiler and SQL parser
    compiler = Compiler()
    parser = Parser()

    parser.attach_module(compiler.module)
    parser.registry.register("users", table.schema)
    sql_query = "SELECT age, score FROM users WHERE age > 21"
    meta = parser.parse(sql_query)

    compiled_query = compiler.compile(meta)

    # 4. Inspection: Output compilation results
    print(f"Target Table: {compiled_query.table_name}")
    print(f"Referenced Columns (Contract Order): {compiled_query.referenced_columns}")
    
    print("\n--- Generated db-dialect MLIR ---")
    print(compiled_query.ir_text)

    print("\n--- Lowered LLVM Dialect IR ---")
    print(compiler.lower_to_llvm_ir())
