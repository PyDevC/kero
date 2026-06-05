"""Parser for SQLGlot AST to DB AST"""
# NOTE: It is incomplete as of now, we need to add support for operations

import sqlglot
import sqlglot.expressions as expr

from ..arrow import type_resolve as arrow_ast
from ..arrow.data import Dataset

import typing as t


class ScanOp:
    def __init__(self, tables: t.List[arrow_ast.Table]):
        self.scan_tables: t.List[arrow_ast.Table] = tables

    def __repr__(self) -> str:
        return f"db.scan [{self.scan_tables}]"

class ProjectOp:
    def __init__(self, projected):
        self.projected: t.List[arrow_ast.Column] = projected

    def __repr__(self) -> str:
        return f"db.project [{self.projected}]"


class ParserExecption(Exception): ...


class Parser:
    def __init__(self, dataset: Dataset):
        self.ast = {}
        self.glot_ast: sqlglot.Expr
        self.dataset = dataset

    def parse(self, query: str):
        self.glot_ast = sqlglot.parse_one(query)

        root = self.glot_ast.find(expr.Select)
        if not root:
            raise ParserExecption("Only Select statements are supported")


        root_tables = self._get_tables(root)
        tables = []

        for table in root_tables:
            table_node = arrow_ast.Table(table.name, {}, 0, 0)
            dataset_table = self.dataset.get_table(table.name)
            arrow_ast.resolve_table(dataset_table, table_node)
            tables.append(table_node)

        selected_columns = self._get_columns(root)
        columns = []

        for column in selected_columns:
            # having table_node unbounded is not a problem 
            # will get error before we even reach here
            column_node = table_node.columns.get(column.name)
            if column_node:
                columns.append(column_node)

        self.add_node(ScanOp(tables), "db.scan")
        self.add_node(ProjectOp(columns), "db.project")

        return self.ast

    def walk(self) -> t.Iterator:
        """return the iterator for walking the ast nodes in order"""
        return iter(self.ast)

    def _get_columns(self, node: expr.Select) -> t.List[expr.Column]:
        selected_columns = node.find_all(expr.Column)
        return list(selected_columns)

    def _get_tables(self, node: expr.Select) -> t.List[expr.Table]:
        tables = node.find_all(expr.Table)
        return list(tables)

    def add_node(self, node, key):
        self.ast[key] = node

    def __repr__(self) -> str:
        return str(self.ast)
