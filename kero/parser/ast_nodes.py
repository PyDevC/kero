"""Dataclasses for representing all nodes related to db-dialect

These nodes are either types or operations mentioned in DB.td
and DBOps.td
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional
import pyarrow as pa

@dataclass
class ColumnRef:
    table_name: str
    column_name: str
    dtype: Optional[pa.DataType] = None

@dataclass
class Literal:
    value: int | float # For now you can't keep it as str
    dtype: Optional[pa.DataType] = None

@dataclass
class BinaryExpr:
    op: str
    left: "Expr"
    right: "Expr"
    result_dtype: Optional[pa.DataType] = None

Expr = ColumnRef | Literal | BinaryExpr

@dataclass
class ScanOp:
    table_name: str
    schema: pa.schema = None# necessary

@dataclass
class ProjectOp:
    source_node: "DBOperation"
    columns: List[ColumnRef]

@dataclass
class GetColOp:
    source_node: "DBOperation"
    column: ColumnRef

@dataclass
class FilterOp:
    source_node: "DBOperation"
    predicate: "Expr"

DBOperation = ScanOp | ProjectOp | GetColOp | FilterOp
