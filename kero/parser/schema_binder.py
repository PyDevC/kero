import sqlglot.expressions as exp
import pyarrow as pa

from typing import Dict

from .ast_nodes import *
from .exec import *

_Scope = Dict[str, pa.Schema]

class SchemaBinder:
    def __init__(self, registry):
        self._registry = registry

    def bind_op(self, op: DBOperation) -> DBOperation:
        scope: _Scope = {}

        if isinstance(op, ScanOp):
            return self._bind_scan_op(op, scope)

    def _bind_scan_op(self, op: DBOperation, scope):
