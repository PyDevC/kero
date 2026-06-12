"""Parser for SQLGlot AST to DB AST"""

import sqlglot
import sqlglot.expressions as expr

from ..arrow import type_resolve as resolve
from ..arrow.data import Dataset

import typing as t

# TODO(PyDevC): Format the code to make it more readable, everything is cluttered right now
class ScanOp(resolve.DBOps):
    def __init__(self, input_table: resolve.Table, outputs=None, attribute=None):
        super().__init__("scan", outputs=outputs, attribute=attribute)
        self.input_table: resolve.Table = input_table

    def __repr__(self) -> str:
        # Keep the input table and output table same
        if self.attribute is not None:
            return f'db.scan {{ {self.attribute["table_name"]} }} : {self.input_table} -> {self.input_table}'

        return f'db.scan : {self.input_table} -> {self.input_table}'

class OutputOp(resolve.DBOps):
    def __init__(self, input_table: resolve.Table, output_table: resolve.Table, attribute=None):
        super().__init__("output", attribute=attribute)
        self.output_table = output_table
        self.input_table = input_table

    def __repr__(self) -> str:
        if self.attribute is not None:
            return f'db.output {{ select = [{self.attribute["select"]}] }} : {self.input_table} -> {self.output_table}'
        else:
            # TODO(PyDevC): raise Exception that no column was selected
            raise Exception()


class ParserExecption(Exception): ...
class ParserNodeNotFound(ParserExecption): ...


class Parser:
    def __init__(self, dataset: Dataset):
        self.operations = []
        self.glot_ast: sqlglot.Expr
        self.dataset = dataset

    def parse(self, query: str):
        self.glot_ast = sqlglot.parse_one(query)

        root = self.glot_ast.find(expr.Select)
        if not root:
            raise ParserExecption(
                f"Only Select statements are supported\n",
                f"Glot AST Trace: ",
                repr(self.glot_ast)
            )

        root_tables = self._get_tables(root)
        tables = []

        for table in root_tables:
            table_node = resolve.Table(table.name, {}, 0, 0)
            dataset_table = self.dataset.get_table(table.name)
            resolve.resolve_table(dataset_table, table_node)
            tables.append(table_node)

        dag_node_scan = ScanOp(tables[0], attribute=None)
        self.operations.append(dag_node_scan)

        selected_columns = self._get_columns(root)
        columns = {}

        for column in selected_columns:
            # having table_node unbounded is not a problem 
            # will get error before we even reach here
            column_node = table_node.columns.get(column.name)
            if column_node:
                columns[column.name] = column_node


        selected_table = resolve.Table("temp", columns, len(columns), table_node.nrows)

        dag_node_output = OutputOp(input_table=dag_node_scan.input_table, output_table=selected_table, attribute={"select": list(columns.values())})
        self.operations.append(dag_node_output)

        return self.operations

    def _get_columns(self, node: expr.Select) -> t.List[expr.Column]:
        # TODO (PyDevC): This is will not work for some complex cases
        # Change how to extract the columns from the node
        selected_columns = node.args.get("expressions")
        return list(selected_columns)

    def _get_tables(self, node: expr.Select) -> t.List[expr.Table]:
        from_node = node.args.get("from_")
        if from_node is None:
            raise ParserNodeNotFound(
                "Glot does not have FROM node",
                repr(node)
            )
        tables = from_node.this

        return [tables]

    def _get_where(self, node: expr.Select):
        where_node = node.args.get("where")
        # TODO (PyDevC): Complete this when working with filter op
        pass
