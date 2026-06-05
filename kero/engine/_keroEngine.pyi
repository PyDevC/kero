from mlir.ir import MlirContext, Module

from typing import Dict
import pyarrow as pa

def register_dialect(context: MlirContext, load: bool=True) -> None:
    """Register All dialects including DB dialect
    """
    pass

def compile_and_execute(module: Module, inputs: pa.table) -> pa.table: ...
