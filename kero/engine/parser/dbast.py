from typing import List, Union

class Metadata:
    """
    Metadata holds necessary information about a type of which it's part of.

    For Tables metadata has: 
        - num_cols: number of columns
        - num_rows: number of rows
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
        self.map = dict(zip(self.metadata.metadata["column_names"], self.metadata.metadata["column_dtypes"]))

    def get_column_dtype(self, column_name):
        return self.map[column_name]

    def __repr__(self) -> str:
        return f'!db.table<{self.metadata.metadata["num_cols"]}, {self.metadata.metadata["num_rows"]} : [{", ".join((repr(col) for col in self.columns))}]>'

class DBColumnAttr:
    def __init__(self, metadata: Metadata):
        self.metadata = metadata

    def __repr__(self) -> str:
        return f'#db.column<{self.metadata.metadata["column_name"]}, {self.metadata.metadata["column_dtype"]}, {self.metadata.metadata["num_rows"]}>'

class DBColumn:
    def __init__(self, dtype, name: str = ""):
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
    def __init__(self, block_args: List[DBColumn], operations: "CmpIOp", f_yield: "FilterYieldOp"):
        self.block_args = block_args
        self.operations = operations
        self.f_yield = f_yield

    def __repr__(self) -> str:
        return f'args: ({self.block_args}):\n{self.operations}\n{self.f_yield}\n'

## Statements
class ScanOp:
    def __init__(self, input: DBTable) -> None:
        self.input = input

    def __repr__(self) -> str:
        return f'db.scan %{self.input.metadata.metadata["name"]} : {self.input} -> {self.input}\n'

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
