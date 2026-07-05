from . import exce as arrowexce

import pyarrow as pa

import typing as t

class Dataset:
    def __init__(self, tables: t.Dict[str, pa.Table] | None = None):
        if tables is not None:
            self.tables = tables
        else:
            self.tables: t.Dict[str, pa.Table] = {}

    def __len__(self):
        return len(self.tables)

    def get_table(self, name: str) -> pa.Table:
        if name in self.tables:
            return self.tables[name]

        raise arrowexce.TableNotFoundException(
            f"Table not found in Dataset: {self.tables.keys()}"
        )

    def add_table(self, name: str, table: pa.Table) -> None:
        self.tables[name] = table

    def get_table_as_arrays(self, name: str):
        if name in self.tables:
            table = self.tables[name]
            return (col.to_numpy() for col in table.columns)

        raise arrowexce.TableNotFoundException(
            f"Table not found in Dataset: {self.tables.keys()}"
        )
