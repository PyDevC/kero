import pyarrow as pa

from typing import Dict


class SchemaRegistry:
    def __init__(self, multiple_query: bool):
        self.schemas: Dict[str, pa.Schema] = {}
        self.multiple_query = multiple_query

    def register(self, table_name: str, table: pa.Schema):
        self.schemas[table_name] = table

    def reset(self):
        self.schemas.clear()

    def on_parse_hook(self):
        if self.multiple_query:
            self.reset()

    def registed(self, table_name: str):
        return table_name in self.schemas

    def get(self, table_name: str) -> pa.Schema:
        if table_name not in self.schemas:
            # TODO: Raise some error
            raise Exception
        return self.schemas[table_name]

    def __len__(self):
        return len(self.schemas)

    def __repr__(self):
        keys = self.schemas.keys
        return f"SchemaRegistry({keys}, {self.multiple_query=})"

