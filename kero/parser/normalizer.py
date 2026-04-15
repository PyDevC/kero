import sqlglot.expressions as exp

from .ast_nodes import *
from .exec import *

class Normalizer:
    def normalize(self, ast: exp.Select) -> DBOperation:
        source = self._extract_source(ast)
        filter = self._extract_filter(source, ast)
        projected = self._extract_projected(filter, ast)
        return projected

    def _extract_source(self, ast: exp.Select) -> DBOperation:
        from_clause = ast.args.get("FROM")
        if from_clause is None:
            raise NodeNotFoundError("'FROM' node not found in AST.")

        table_name = from_clause.this.name
        return ScanOp(table_name=table_name)

    def _extract_filter(self, source: DBOperation, ast: exp.Select) -> DBOperation:
        where_clause = ast.args.get("WHERE")
        if where_clause is None:
            # There is no Where clause used so we need to project all rows
            raise NodeNotFoundError("This is not an excpetion but for now we want where clause in query")

        return FilterOp(source_node=source, predicate=where_clause)

    def _extract_projected(self, filter: DBOperation, ast: exp.Select) -> DBOperation:
        columns = []
        for col in ast.expressions:
            col_name = col.alias_or_name
            if col_name:
                columns.append(col_name)
            else:
                raise NotImplemented("Complex select queries not implemented.")

        return ProjectOp(source_node=filter, columns=columns)
