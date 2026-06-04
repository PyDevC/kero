from . import exce as arrowexce

import pyarrow as pa

import typing as t

class Dataset:
    def __init__(self, tables: t.Dict[str, pa.Table] | None = None):
        if tables is not None:
            self.tables = tables
        else:
            self.tables: t.Dict[str, pa.Table] = dict()

    def __len__(self):
        return len(self.tables)

    def get_table(self, name: str) -> pa.Table:
        if name in self.tables:
            return self.tables[name]

        # TODO(PyDevC): Maybe we need to be a little bit less verbose
        raise arrowexce.TableNotFoundException(
            f"Table not found in Dataset: {self.tables}"
        )

    def add_table(self, name: str, table: pa.Table) -> None:
        self.tables[name] = table
