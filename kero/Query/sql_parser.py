import sqlglot
from sqlglot import parse_one, expressions as exp
from .exec import ( ParserException )
from .catalog import get_result_type

from dataclasses import dataclass

class SelectNode():
    def __init__(self, node_type, specific_type, query_expression):
        self.node_type = node_type
        self.specific_type = specific_type
        self.query_expression = query_expression
        self.from_clause = self._create_from_clause_dict(query_expression)
        self.select_list = self._create_select_list(query_expression)
        self.where_clause = self._create_where_clause_dict(query_expression)

    def _create_select_list(self, node: exp.Expression):
        select_list = []
        for expression in node.expressions:
            if isinstance(expression, exp.Column):
                select_list.append(self._create_column_node_dict(expression.name, self.from_clause['table_name']))
            elif isinstance(expression, exp.Star):
                select_list.append({
                    "expression_class": "SELECT_STAR",
                    "table_reference": self.from_clause['table_name'],
                    "result_type": get_result_type(),
                })

        return select_list

    def _create_column_node_dict(self, column_name, table_reference):
        """Returns a dictionary for column node from the main query node
        """
        return {
            "expression_class": "COLUMN",
            "column_name": column_name,
            "table_reference": table_reference,
            "result_type": get_result_type(),
        }

    def _create_from_clause_dict(self, query_expression: sqlglot.Expression):
        for node in query_expression.find_all(exp.Table):
                return {
                    "node_type": "TABLE",
                    "specific_type": "TABLE",
                    "table_name": node.name 
                }

        return {}

    def _create_where_clause_dict(self, query_expression):
        where_clause = query_expression.args.get("where")  
        if where_clause:  
            return self._convert_expression(where_clause)  
        return {}  
  
    def _convert_expression(self, expression: exp.Expression):
        if isinstance(expression, exp.Column):  
            return self._create_column_node_dict(expression.name, expression.table)  
          
        elif isinstance(expression, exp.Literal):  
            return {  
                "expression_class": "CONSTANT",  
                "value": expression.this,  
                "result_type": get_result_type()
            }  
          
        elif isinstance(expression, exp.EQ):  
            return self._create_comparison_dict("COMPARE_EQUAL", expression)  
          
        elif isinstance(expression, exp.GT):  
            return self._create_comparison_dict("COMPARE_GREATERTHAN", expression)  
          
        elif isinstance(expression, exp.LT):  
            return self._create_comparison_dict("COMPARE_LESSTHAN", expression)  
          
        elif isinstance(expression, exp.GTE):  
            return self._create_comparison_dict("COMPARE_GREATERTHANOREQUALTO", expression)  
          
        elif isinstance(expression, exp.LTE):  
            return self._create_comparison_dict("COMPARE_LESSTHANOREQUALTO", expression)  
          
        elif isinstance(expression, exp.And):  
            return self._create_conjunction_dict("CONJUNCTION_AND", expression)  
          
        elif isinstance(expression, exp.Or):  
            return self._create_conjunction_dict("CONJUNCTION_OR", expression)  
          
        else:  
            return {  
                "expression_class": "UNKNOWN",  
                "expression_type": str(expression.__class__.__name__),  
                "result_type": get_result_type()
            }  
  
    def _create_comparison_dict(self, comp_type: str, expression: exp.Operator):
        left = self._convert_expression(expression.left)  
        right = self._convert_expression(expression.right)  
          
        return {  
            "expression_class": "BOUND_COMPARISON",  
            "expression_type": comp_type,  
            "children": [left, right],  
            "result_type": {"type_id": "BOOLEAN", "is_nullable": False}  
        }  
  
    def _create_conjunction_dict(self, conj_type: str, expression: exp.Connector):
        children = [self._convert_expression(expr) for expr in expression.flatten()]  
          
        return {  
            "expression_class": "CONJUNCTION",  
            "expression_type": conj_type,  
            "children": children,  
            "result_type": get_result_type()
        }  
  
    def to_dict(self):
        return {  
            'node_type': self.node_type,  
            'specific_type': self.specific_type,  
            'select_list': self.select_list,  
            'from_clause': self.from_clause,  
            'where_clause': self.where_clause  
        }  

class Parser:
    def _generate_query_dict(self, query_expression: sqlglot.Expression):
        if isinstance(query_expression, exp.Select):
            select_node = SelectNode("QUERY", "SELECT", query_expression)
            return select_node.to_dict()


    @classmethod
    def Capture(cls, query):
        try:
            query_expression = parse_one(query)
            query_dict = cls()._generate_query_dict(query_expression)
        except Exception as e:
            raise ValueError(f"{e}")
        return query_dict
