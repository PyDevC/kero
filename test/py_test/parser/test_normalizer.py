"""example.py - Demo file for testing the kero module."""

import torch
import pyarrow as pa

from kero.tensors import TableTensor, NumTensor, StrTensor
from kero.engine.parser import Parser
from kero.engine.compiler import Compiler
from kero.engine.executor import Executor


def create_sample_data():
    """Create a sample TableTensor for testing."""
    columns = {
        "id": NumTensor(torch.tensor([1, 2, 3, 4, 5]), "id"),
        "name": StrTensor(["alice", "bob", "carol", "dave", "eve"], "name"),
        "age": NumTensor(torch.tensor([25, 30, 35, 40, 45]), "age"),
        "salary": NumTensor(torch.tensor([50000.0, 60000.0, 70000.0, 80000.0, 90000.0]), "salary"),
    }
    return TableTensor(columns, name="employees")


def test_tensor_creation():
    """Test basic tensor creation."""
    print("=== Test: Tensor Creation ===")
    table = create_sample_data()
    print(f"Table: {table.name}")
    print(f"Columns: {list(table.columns.keys())}")
    print(f"Rows: {len(table.columns['id'].tensor)}")
    print()


def test_parser_compiler():
    """Test parsing and compiling a SQL query."""
    print("=== Test: Parser + Compiler ===")

    table = create_sample_data()

    try:
        compiler = Compiler()
        parser = Parser()
        parser.attach_module(compiler.module)

        meta = parser.parse("SELECT id, name, age FROM employees WHERE age > 30")
        print(f"Parsed query, table: {meta.table_name}")
        print(f"Referenced columns: {meta.referenced_columns}")

        compiled = compiler.compile(meta)
        print(f"Compiled query: {compiled}")
        print(f"Output schema: {compiled.output_schema}")
    except RuntimeError as e:
        print(f"Expected (C++ extension not built): {e}")
    print()


def test_executor():
    """Test the Executor with tensor operations."""
    print("=== Test: Executor ===")

    table = create_sample_data()

    try:
        executor = Executor(table)

        kquery = {
            "operations": [
                ("filter", "age", ">", 30),
            ],
            "columns": ["id", "name", "age"],
        }
        result = executor.execute(kquery)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    print()


def test_arrow_integration():
    """Test with actual Arrow table."""
    print("=== Test: Arrow Integration ===")

    data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["alice", "bob", "carol", "dave", "eve"],
        "age": [25, 30, 35, 40, 45],
    }
    table = pa.table(data)

    print(f"Arrow table:\n{table}")
    print(f"Schema: {table.schema}")
    print()


if __name__ == "__main__":
    test_tensor_creation()
    test_parser_compiler()
    test_executor()
    test_arrow_integration()
    print("Done!")
