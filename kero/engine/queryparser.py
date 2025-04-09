import sqlglot
import sqlglot.expressions as exp

class Parser:
    def parse(self, query: str) -> dict:
        self.query = query
        kquery = self._generate_kquery()
        return kquery

    def _generate_kquery(self) -> dict:
        parsed = sqlglot.parse_one(self.query)
        query_dict = {
            'columns': [],
            'Table': None,
            'operations': [],
            'op_pattern': []
        }

        # Extract table name
        for node in parsed.find_all(exp.Table):
            query_dict['Table'] = node.name

        # Extract WHERE conditions
        where_clause = parsed.find(exp.Where)
        if where_clause:
            self._extract_conditions(where_clause.this, query_dict)

        self._extract_columns(parsed, query_dict['columns'])

        return query_dict


    def _extract_columns(self, parsed, columns):
        for expression in sqlglot.parse_one(self.query).find(exp.Select).args["expressions"]:
            if isinstance(expression, exp.Alias):
                columns.append(expression.text("alias"))
            elif isinstance(expression, exp.Column):
                columns.append(expression.text("this"))

    def _extract_conditions(self, condition, query_dict):
        """Recursively extract conditions from WHERE clause"""
        if isinstance(condition, exp.And) or isinstance(condition, exp.Or):
            self._extract_conditions(condition.left, query_dict)
            self._extract_conditions(condition.right, query_dict)
        elif isinstance(condition, exp.Binary):
            operator = self._get_operator_symbol(condition)
            left = condition.left.name if isinstance(condition.left, exp.Column) else condition.left.this
            right = condition.right.this if isinstance(condition.right, exp.Literal) else condition.right.name
            
            query_dict["operations"].append({
                "operator": operator,
                "left": left,
                "right": right
            })
            query_dict["op_pattern"].append((left, right, operator))

    def _get_operator_symbol(self, condition):
        """Map condition types to operator symbols"""
        if isinstance(condition, exp.EQ):
            return '='
        elif isinstance(condition, exp.NEQ):
            return '!='
        elif isinstance(condition, exp.GT):
            return '>'
        elif isinstance(condition, exp.GTE):
            return '>='
        elif isinstance(condition, exp.LT):
            return '<'
        elif isinstance(condition, exp.LTE):
            return '<='
        elif isinstance(condition, exp.Like):
            return 'LIKE'
        return 'UNKNOWN'
