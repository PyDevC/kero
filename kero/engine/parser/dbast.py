from typing import List, Union

class Metadata:
    """
    Metadata holds necessary information about a type of which it's part of.

    For Tables metadata has: 
        - ncols: number of columns
        - nrows: number of rows
        - columns: names of columns

    You can add or remove any information from the metadata and just need to check 
    if it exists, this is very good for writing assert statements making it easier 
    to verify the correctness of semantics before you generate DB IR from this.
    """
    def __init__(self) -> None:
        self.metadata = {}

## Types
class DBLiteral:
    def __init__(self, constant: Union[int, float]):
        self.constant = constant

    def __repr__(self) -> str:
        return f'{self.constant}'

class DBTable:
    def __init__(self, metadata: Metadata, columns: List["DBColumnAttr"]) -> None:
        self.metadata = metadata
        self.columns = columns

    def __repr__(self) -> str:
        return f'!db.table<{self.metadata.metadata["ncols"]}, {self.metadata.metadata["nrows"]} : [{", ".join((repr(col) for col in self.columns))}]>'

class DBColumnAttr:
    def __init__(self, metadata: Metadata):
        self.metadata = metadata

    def __repr__(self) -> str:
        return f'#db.column<{self.metadata.metadata["name"]}, {self.metadata.metadata["dtype"]} : [{self.metadata.metadata["nrows"]}]>'

class DBColumn:
    def __init__(self, dtype: str, name: str = ""):
        self.dtype = dtype
        self.name = name

    def __repr__(self) -> str:
        return f'!db.column<{self.dtype}>'

class DBCmpIPredicate:
    def __init__(self, predicate: str):
        self.predicate = predicate

    def __repr__(self) -> str:
        return f'{self.predicate}'

class DBRegion:
    def __init__(self, operations: "CmpIOp", f_yield: "FilterYieldOp"):
        self.operations = operations
        self.f_yield = f_yield

    def __repr__(self) -> str:
        return f'{self.operations}\n{self.f_yield}\n'

## Statements
class ScanOp:
    def __init__(self, table: DBTable) -> None:
        self.table = table

    def __repr__(self) -> str:
        return f'db.scan %{self.table.metadata.metadata["name"]} : {self.table} -> {self.table}\n'

class OutputOp:
    def __init__(self, input: DBTable, output: DBTable, select: List[str]) -> None:
        self.input = input
        self.output = output
        self.select = select

    def __repr__(self) -> str:
        return f'db.output {{ {self.select} }} %{self.input.metadata.metadata["name"]} : {self.input} -> {self.output}\n'

class FilterOp:
    def __init__(self, input: DBTable, output: DBTable, region: DBRegion) -> None:
        self.input = input
        self.output = output
        self.region = region

    def __repr__(self) -> str:
        return f'db.filter %{self.input.metadata.metadata["name"]} : {self.input} {{ \n {self.region} }} -> {self.output}\n'

class FilterYieldOp:
    def __init__(self, mask: DBColumn) -> None:
        self.mask = mask

    def __repr__(self) -> str:
        return f'db.filter_yield %yield_var : {self.mask}\n'

class CmpIOp:
    def __init__(self, lhs: DBColumn, rhs: DBLiteral, predicate: str, output) -> None:
        self.lhs = lhs
        self.rhs = rhs
        self.predicate = predicate
        self.output = output

    def __repr__(self) -> str:
        return f'db.cmpi {self.predicate}, %column, %constant : ({self.lhs}, {self.rhs}) -> {self.output}\n'
