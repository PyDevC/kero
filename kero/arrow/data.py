from . import exce as arrowexce

import pyarrow as pa
import numpy as np

from typing import Any, Generator, Iterator

class Dataset:
    def __init__(self, tables: dict[str, pa.Table] | None = None):
        if tables is not None:
            self.tables = tables
        else:
            self.tables: dict[str, pa.Table] = {}

        self.metadatas = {}

    def __len__(self) -> int:
        return len(self.tables)

    def _get_table(self, name: str) -> pa.Table:
        if name in self.tables:
            return self.tables[name]

        raise arrowexce.TableNotFoundException(
            f"Table not found in Dataset: {self.tables.keys()}"
        )

    def __getitem__(self, name) -> pa.Table:
        return self._get_table(name)

    def __setitem__(self, name: str, table: pa.Table) -> None:
        self.metadatas.pop(name)
        self.tables[name] = table

    def __delitem__(self, name: str) -> None:
        self.metadatas.pop(name)
        self.tables.pop(name)

    def __contains__(self, name: str) -> bool:
        return name in self.tables

    def __iter__(self) -> Iterator:
        return iter(self.tables)

    def get_table(self, name: str) -> pa.Table:
        return self.__getitem__(name)

    def remove_table(self, name: str) -> None:
        self.__delitem__(name)

    def add_table(self, name: str, table: pa.Table) -> None:
        if name in self.tables:
            raise arrowexce.TableAlreadyInDataset(
                f"Table {name} already present in the dataset\n"
                "To overwrite do dataset[name] = table"
            )

        self.__setitem__(name, table)

    def get_table_as_arrays(self, name: str) -> Generator[np.ndarray]:
        return (col.to_numpy() for col in self._get_table(name).columns)

    def get_table_metadata(self, name: str) -> dict[str, Any]:
        if name in self.metadatas:
            return self.metadatas[name]

        self.metadatas[name] = self._get_metadata(name)
        return self.metadatas[name]

    def _get_metadata(self, name: str) -> dict[str, Any]:
        table = self._get_table(name)
        num_rows = table.num_rows
        num_cols = table.num_columns
        schema = table.schema
        column_dtypes = [field.type for field in schema]
        column_names= [field.name for field in schema]

        return {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "column_dtypes": column_dtypes,
            "column_names": column_names,
        }

    @property
    def table_names(self) -> list[str]:
        return list(self.tables.keys())
