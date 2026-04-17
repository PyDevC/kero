import torch
import ctypes
import pyarrow as pa
from kero._arrow_bridge import ArrowTableTensorFormat
from typing import Dict, List, Any

class Executor:
    def __init__(self, table: pa.Table):
        self.table = table
        self.bridge = ArrowTableTensorFormat()
        self.result = None

    def execute(self, compiled_query: Any) -> Dict[str, Any]:
        num_rows = len(self.table)
        
        # 1. Maintain object persistence to prevent pointer invalidation
        descriptors = []
        addresses = []

        # 2. Add Table Descriptor (Arg 0)
        # Note: In MLIR db-dialect, Arg 0 is the table itself converted to a tensor
        table_tensor = self.bridge.arrow_to_tensor(self.table, compiled_query.referenced_columns[0], [num_rows, compiled_query.n_cols])
        table_desc = self.bridge.tensor_to_memref(table_tensor)
        descriptors.append(table_desc)
        addresses.append(ctypes.addressof(table_desc))

        # 3. Add Column Descriptors (Args 1..N)
        for col_name in compiled_query.referenced_columns:
            tensor_ptr = self.bridge.arrow_to_tensor(self.table, col_name, [num_rows, 1])
            if tensor_ptr is None:
                raise RuntimeError(f"Column {col_name} resolution failed")
            
            col_desc = self.bridge.tensor_to_memref(tensor_ptr)
            descriptors.append(col_desc)
            addresses.append(ctypes.addressof(col_desc))

        # 4. Invoke JIT
        # The list 'descriptors' keeps the C-structs alive in memory during this call
        compiled_query.call_jit(addresses)
        
        return {"status": "success", "table": compiled_query.table_name}
