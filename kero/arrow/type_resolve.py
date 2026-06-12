import pyarrow as pa

import typing as t

import kero._engine._kero.ir as ir


PYARROW_TO_DB_TYPES = {
    # Integer Types
    pa.int8(): "i8",
    pa.int16(): "i16",
    pa.int32(): "i32",
    pa.int64(): "i64",

    # Floating Types
    pa.float16(): "f16",
    pa.float32(): "f32",
    pa.float64(): "f64",

    # String Types
    pa.string(): ir.StringAttr,
}

class DBOps:
    def __init__(
            self, 
            id, 
            inputs: t.Optional[t.List['DBOps']] = None, 
            outputs: t.Optional[t.List['DBOps']] = None, 
            attribute=None
    ):
        self.id = id
        self.inputs = inputs or []
        self.attribute = attribute
        self.outputs = outputs or []

class DBTypes: ...
    
class Column(DBTypes):
    def __init__(self, name: str, dtype: DBTypes, nrows: int):
        self.name = name
        self.dtype = dtype
        self.nrows = nrows

    def __repr__(self):
        return f'#db.column<"{self.name}", {self.dtype}, {self.nrows}>'

class Table(DBTypes):
    def __init__(self, name: str, columns: t.Dict[str, Column], ncols: int, nrows: int):
        self.name = name
        self.columns = columns
        self.ncols = ncols
        self.nrows = nrows

    def __repr__(self):
        return f'!db.table<{self.ncols}, {self.nrows} : [{", ".join([repr(column) for column in self.columns.values()])}]>'


def resolve_table(table: pa.Table, node):
    """Resolve table with some columns and ncols and nrows"""
    schema = table.schema
    ncols = table.num_columns
    nrows = table.num_rows
    columns = {}
    for field in schema:
        columns[field.name] = Column(field.name, field.type, nrows)

    node.columns = columns
    node.nrows = nrows
    node.ncols = ncols

    return node
