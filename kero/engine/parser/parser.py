import typing as t
from types import MethodType

import sqlglot
import sqlglot.expressions as exp

from .dbast import *
from kero.arrow.data import Dataset 


DB_Ast: t.TypeAlias = List[ScanOp | OutputOp | FilterOp]

class Parser:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def parse(self, query: str) -> DB_Ast:
        converter = GlotToDB(query, self.dataset)
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
    def __init__(self, node):
        super().__init__()
        self.node = node

    def __str__(self):
        return f'{self.node} node not implemented'

class ColumnNotInTable(BaseGlotToDBException):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __str__(self):
        return f'Column(s) {self.name} not present in Table'

class GlotToDB:
    def __init__(self, query: str, dataset: Dataset) -> None:
        self.sqlglot_ast = sqlglot.parse_one(query)
        self.dataset = dataset
        self.op_map = self._get_operation_map()

    def convert(self) -> DB_Ast:
        operations = []

        self.root = self.sqlglot_ast.find(exp.Select)
        if not self.root:
            raise NodeNotFound(exp.Select)

        select_op = self._parse_scan_op(self.root)
        operations.append(select_op)

        self.input_table = select_op.input

        where_clause = self.root.find(exp.Where)
        if where_clause:
            # Get filter output table
            metadata = Metadata()
            metadata.metadata = select_op.input.metadata.metadata.copy()
            metadata.metadata["num_rows"] = -1
            filter_column_attr = self._create_column_attr_node(metadata.metadata)
            filter_output_table = DBTable(metadata, filter_column_attr)

            where_operations, block_args = self._parse_where_clause(where_clause, self.input_table)
            filter_op = self._parse_filter_op(self.input_table, filter_output_table, block_args, where_operations)
            operations.append(filter_op)

            self.input_table = filter_op.output

        output_op = self._parse_output_op(self.root, self.input_table)
        operations.append(output_op)

        return operations

    def _parse_table(self, glotnode: exp.Select):
        table = glotnode.find(exp.From)
        if table:
            metadata = self.dataset.get_table_metadata(table.name)
            table_metadata = Metadata()
            table_metadata.metadata = metadata
            table_column_attr = self._create_column_attr_node(metadata)
            table_node = DBTable(table_metadata, table_column_attr)
            return table_node

        raise NodeNotFound(exp.From)

    def _create_column_attr_node(self, metadata):
        column_names = metadata["column_names"]
        column_dtypes = metadata["column_dtypes"]
        num_rows = metadata["num_rows"]
        column_attr = []
        for name, dtype in zip(column_names, column_dtypes):
            column_metadata = Metadata()
            column_metadata.metadata = { 
                "column_name": name,
                "num_rows": num_rows,
                "column_dtype": dtype
            }

            column_attr.append(DBColumnAttr(column_metadata))

        return column_attr

    def _parse_selected_columns(self, glotnode: exp.Select):
        columns = glotnode.expressions
        if len(columns) == 0:
            raise NodeNotFound(exp.Column)

        if isinstance(columns[0], exp.Star):
            return ["*"]
        
        return [column.this.name for column in columns]

    def _parse_where_clause(self, glotnode: exp.Where, input: DBTable):
        where_operation = glotnode.this
        operations = self._parse_expression(where_operation)

        if isinstance(operations, DBColumn) or isinstance(operations, DBLiteral):
            raise GlotConversionNotPossible(where_operation)

        glot_columns = where_operation.find_all(exp.Column)
        seen_columns = set()
        block_args = []
        for col in glot_columns:
            if col.name not in seen_columns:
                seen_columns.add(col.name)
                block_args.append(self._parse_exp_column(col))

        return operations, block_args
            
    def _parse_expression(self, glotnode: exp.Expr) -> t.Union[LogicalOp, CmpIOp, DBColumn, DBLiteral]:
        if type(glotnode) not in self.op_map:
            raise NodeNotImplemented(glotnode)

        return self.op_map[type(glotnode)](glotnode)

    def _get_operation_map(self) -> t.Dict[t.Any, MethodType]:
        return {
            exp.And: self._parse_exp_and,
            exp.Or: self._parse_exp_or,
            exp.Not: self._parse_exp_not,
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
        raise NodeNotImplemented(glotnode)

    # Parse specific expressions
    def _parse_exp_logical(self, glotnode: exp.Expr):
        lhs = self._parse_expression(glotnode.this)
        rhs = self._parse_expression(glotnode.expression)
        output = DBColumn("bool")

        if isinstance(lhs, DBColumn):
            raise GlotConversionNotPossible(type(lhs))
        elif isinstance(rhs, DBColumn):
            raise GlotConversionNotPossible(type(lhs))

        if isinstance(lhs, DBLiteral):
            raise GlotConversionNotPossible(type(rhs))
        elif isinstance(rhs, DBLiteral):
            raise GlotConversionNotPossible(type(lhs))

        return lhs, rhs, output

    def _parse_exp_and(self, glotnode: exp.And):
        lhs, rhs, output = self._parse_exp_logical(glotnode)
        return AndOp(lhs, rhs, output)

    def _parse_exp_or(self, glotnode: exp.Or):
        lhs, rhs, output = self._parse_exp_logical(glotnode)
        return OrOp(lhs, rhs, output)

    def _parse_exp_not(self, glotnode: exp.Not):
        rhs = self._parse_expression(glotnode.this)
        output = DBColumn("bool")

        if isinstance(rhs, DBColumn):
            raise GlotConversionNotPossible(type(rhs))
        elif isinstance(rhs, DBLiteral):
            raise GlotConversionNotPossible(type(rhs))

        return NotOp(rhs, output)

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
        try: 
            number = int(glotnode.this)
        except ValueError:
            number = float(glotnode.this)

        return DBLiteral(number)

    def _parse_exp_column(self, glotnode: exp.Column) -> DBColumn:
        dtype = self.input_table.get_column_dtype(glotnode.name)
        return DBColumn(dtype, glotnode.name)

    # Parse Operations
    def _parse_scan_op(self, glotnode: exp.Select) -> ScanOp:
        table_node = self._parse_table(glotnode)
        return ScanOp(table_node)

    def _parse_filter_op(
        self,
        input: DBTable,
        output: DBTable,
        block_args: List[DBColumn],
        operations: Union[CmpIOp, LogicalOp]
    ) -> FilterOp:

        mask = DBColumn("bool")
        f_yield = FilterYieldOp(mask)
        region = DBRegion(block_args, operations, f_yield)
        return FilterOp(input, output, region)


    def _parse_output_op(self, glotnode: exp.Select, input: DBTable) -> OutputOp:
        selected = self._parse_selected_columns(glotnode)
        column_names = input.metadata.metadata["column_names"]
        
        if selected[0] != "*":
            extra_columns = set(selected) - set(column_names)
            if extra_columns:
                raise ColumnNotInTable(list(extra_columns))
        else:
            selected = column_names

        selected_dtypes = [input.get_column_dtype(col) for col in selected]

        metadata = Metadata()
        metadata.metadata = {
            "name": input.metadata.metadata["name"],
            "num_rows": input.metadata.metadata["num_rows"],
            "num_cols": len(selected),
            "column_dtypes": selected_dtypes,
            "column_names": selected,
        }

        column_attr = self._create_column_attr_node(metadata.metadata)
        output = DBTable(metadata, column_attr)
        return OutputOp(input, output, selected)
