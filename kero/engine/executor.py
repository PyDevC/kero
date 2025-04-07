import torch
from typing import Dict, Any
from kero.tensors import TableTensor
from kero.compiler import KeroCompiler
from kero.queryparser import KeroQueryParser


class Executor:
    """
    Executes queries represented in Intermediate Representation (IR).
    This class integrates query parsing, compilation, and execution.
    """

    def __init__(self, table: TableTensor):
        """
        Initialize the executor with a TableTensor.

        Args:
            table (TableTensor): The tensor-based representation of a relational table.
        """
        self.table = table
        self.parser = KeroQueryParser()
        self.compiler = KeroCompiler(table)

    def execute_query(self, query: str) -> torch.Tensor:
        """
        Parse, compile, and execute a query.

        Args:
            query (str): SQL-like query string.

        Returns:
            torch.Tensor: Resulting tensor after executing the query.
        """
        # Step 1: Parse the query into Intermediate Representation (IR)
        print(f"Parsing query: {query}")
        ir = self.parser.parse(query)
        print(f"Intermediate Representation: {ir}")

        # Step 2: Compile and execute the IR using the compiler
        print("Compiling and executing the query...")
        result = self.compiler.compile(ir)

        print("Execution complete.")
        return result
