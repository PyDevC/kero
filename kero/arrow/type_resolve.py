## TODO(PyDevC): Here create a mapping of types from pyarrow to db dialect

import pyarrow as pa

import typing as t

class DBTypes: ...
    
class Column:
    def __init__(self, name: str, dtype: DBTypes, nrows: int):
        self.name = name
        self.dtype = dtype
        self.nrows = nrows

    def __repr__(self):
        return f"#db.column<'{self.name}', {self.dtype}, {self.nrows}>"

class Table:
    def __init__(self, name: str, columns: t.Dict[str, Column], ncols: int, nrows: int):
        self.name = name
        self.columns = columns
        self.ncols = ncols
        self.nrows = nrows

    def __repr__(self):
        return f"!db.table<'{self.name}', {self.columns.values()}, {self.ncols}, {self.nrows}>"


def resolve_table(table: pa.Table, node):
    """Resolve table with some columns and ncols and nrows"""
    schema = table.schema
    nrows = table.num_columns
    ncols = table.num_rows
    columns = {}
    for field in schema:
        columns[field.name] = Column(field.name, field.type, nrows)

    node.columns = columns
    node.nrows = nrows
    node.ncols = ncols

    return node
