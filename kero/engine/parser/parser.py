import typing as t
from types import MethodType

import sqlglot
import sqlglot.expressions as exp

from .dbast import *
import kero.arrow.type_resolve as resolve
from kero.arrow.data import Dataset 


DB_Ast = t.Union[tuple[ScanOp, OutputOp, FilterOp], tuple[ScanOp, OutputOp]]

class Parser:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def parse(self, query: str) -> DB_Ast:
        converter = GlotToDB(query)
        return converter.convert()

## Parser and Converter Exceptions
class BaseParserException(Exception): ...
class BaseGlotToDBException(Exception): ...
class GlotConversionNotPossible(BaseGlotToDBException):
    def __init__(self, node):
        super().__init__()
        self.node = node

    def __str__(self):
        return f'Could not convert {type(self.node)} to any of the possible db ops'

class NodeNotFound(BaseParserException):
    def __init__(self, node):
        super().__init__()
        self.node = node

    def __str__(self):
        return f'{type(self.node)} does not exists in the query'

class NodeNotImplemented(BaseGlotToDBException):
    def __str__(self):
        return 'Node not implemented'

class GlotToDB:
    def __init__(self, query: str) -> None:
        self.parsed_query = sqlglot.parse_one(query)
        self.op_map = self._create_op_map()

    def convert(self) -> DB_Ast:
        self.root = self.parsed_query.find(exp.Select)
        if not self.root:
            raise NodeNotFound(exp.Select)

        select_op = self._parse_scan_op(self.root)
        output_op = self._parse_output_op(select_op.table)

        where_clause = self.root.find(exp.Where)
        if where_clause:
            where_operations = self._parse_where_clause(where_clause)
            filter_op = self._parse_filter_op(select_op.table, output_op.output, where_operations)
            return (select_op, output_op, filter_op)

        return (select_op, output_op)

    def _parse_table(self, glotnode: exp.Select, table_columns: t.List[DBColumnAttr]):
        table = glotnode.find(exp.From)
        if table:
            table_metadata = Metadata()
            table_metadata.metadata["name"] = table.name
            table_metadata.metadata["nrows"] = 0
            table_metadata.metadata["ncols"] = 0
            table_node = DBTable(table_metadata, table_columns)
            return table_node

        raise NodeNotFound(exp.From)

    def _parse_columns(self, glotnode: exp.Select):
        columns = glotnode.expressions
        if len(columns) == 0:
            raise NodeNotFound(exp.Column)

        if isinstance(columns[0], exp.Star):
            metadata = Metadata()
            metadata.metadata["is_star"] = True
            metadata.metadata["name"] = None
            metadata.metadata["dtype"] = None
            metadata.metadata["nrows"] = None
            star_column = DBColumnAttr(metadata)
            return [star_column]
        
        column_nodes = []
        for column in columns:
            metadata = Metadata()
            metadata.metadata["is_star"] = False
            metadata.metadata["name"] = column.this.name
            metadata.metadata["dtype"] = None
            metadata.metadata["nrows"] = None
            column_nodes.append(DBColumnAttr(metadata))

        return column_nodes

    def _parse_where_clause(self, glotnode: exp.Where) -> CmpIOp:
        where_operation = glotnode.this
        operations = self._parse_expression(where_operation)
        if isinstance(operations, DBColumn) or isinstance(operations, DBLiteral):
            raise GlotConversionNotPossible(where_operation)

        return operations
            
    def _parse_expression(self, glotnode: exp.Expr) -> t.Union[CmpIOp, DBColumn, DBLiteral]:
        if type(glotnode) not in self.op_map:
            raise NodeNotImplemented()

        return self.op_map[type(glotnode)](glotnode)

    def _create_op_map(self) -> t.Dict[t.Any, MethodType]:
        return {
            exp.And: self._not_implemeted,
            exp.Or: self._not_implemeted,
            exp.Not: self._not_implemeted,
            exp.LT: self._parse_exp_cmp_lt,
            exp.LTE: self._parse_exp_cmp_lte,
            exp.EQ: self._parse_exp_cmp_eq,
            exp.NEQ: self._parse_exp_cmp_neq,
            exp.GT: self._parse_exp_cmp_gt,
            exp.GTE: self._parse_exp_cmp_gte,
            exp.Literal: self._parse_exp_literal,
            exp.Column: self._parse_exp_column,
        }

    def _not_implemeted(self, glotnode: exp.Expr):
        raise NodeNotImplemented()

    # Parse specific expressions
    def _parse_exp_cmp(self, glotnode: exp.Expr, predicate) -> CmpIOp:
        lhs = self._parse_expression(glotnode.this)
        rhs = self._parse_expression(glotnode.expression)
        output = DBColumn("bool")

        if not isinstance(lhs, DBColumn):
            raise GlotConversionNotPossible(type(lhs))

        if not isinstance(rhs, DBLiteral):
            raise GlotConversionNotPossible(type(rhs))

        return CmpIOp(lhs, rhs, predicate, output)

    def _parse_exp_cmp_lt(self, glotnode: exp.LT) -> CmpIOp:
        return self._parse_exp_cmp(glotnode, "lt")

    def _parse_exp_cmp_lte(self, glotnode: exp.LTE) -> CmpIOp:
        return self._parse_exp_cmp(glotnode, "lte")

    def _parse_exp_cmp_eq(self, glotnode: exp.EQ) -> CmpIOp:
        return self._parse_exp_cmp(glotnode, "eq")

    def _parse_exp_cmp_neq(self, glotnode: exp.NEQ) -> CmpIOp:
        return self._parse_exp_cmp(glotnode, "neq")

    def _parse_exp_cmp_gt(self, glotnode: exp.GT) -> CmpIOp:
        return self._parse_exp_cmp(glotnode, "gt")

    def _parse_exp_cmp_gte(self, glotnode: exp.GTE) -> CmpIOp:
        return self._parse_exp_cmp(glotnode, "gte")

    def _parse_exp_literal(self, glotnode: exp.Literal) -> DBLiteral:
        return DBLiteral(glotnode.this)

    def _parse_exp_column(self, glotnode: exp.Column) -> DBColumn:
        return DBColumn("None")

    # Parse Operations
    def _parse_scan_op(self, glotnode: exp.Select) -> ScanOp:
        table_columns = self._parse_columns(glotnode)
        table_node = self._parse_table(glotnode, table_columns)
        return ScanOp(table_node)

    def _parse_filter_op(
            self, 
            input: DBTable, 
            output: DBTable, 
            operations: CmpIOp
        ) -> FilterOp:

        mask = DBColumn("bool")
        f_yield = FilterYieldOp(mask)
        region = DBRegion(operations, f_yield)
        return FilterOp(input, output, region)


    def _parse_output_op(self, input: DBTable) -> OutputOp:
        select = [column.metadata.metadata["name"] for column in input.columns]
        # We will create output table same as input since we have not type resolved it
        metadata = Metadata()
        metadata.metadata = input.metadata.metadata.copy()
        output = DBTable(metadata, input.columns.copy())
        return OutputOp(input, output, select)
