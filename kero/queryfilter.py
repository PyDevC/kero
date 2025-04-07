import sqlparse
from typing import Dict, Any

class QueryFilter:
    """
    A class responsible for parsing SQL queries and converting them into Intermediate Representation (IR).
    """

    def __init__(self):
        pass

    def filter(self, query: str) -> tuple[str, Dict[str, Any]]:
        """
        Parse an SQL query and convert it into PostgreSQL query and Intermediate Representation (IR).

        Args:
            query (str): SQL query string.

        Returns:
            tuple[str, Dict[str, Any]]: Tuple containing the PostgreSQL query and IR.
        """
        parsed_query = sqlparse.parse(query)[0]

        # Extract SELECT clause
        columns = self._extract_columns(parsed_query)

        # Extract FROM clause (table name)
        table_name = self._extract_table_name(parsed_query)

        # Extract WHERE clause
        where_clause = self._extract_where_clause(parsed_query)

        # Construct PostgreSQL query
        pg_query = f"SELECT {', '.join(columns)} FROM {table_name}"

        # Construct Intermediate Representation (IR)
        ir = {
            "columns": columns,
            "table": table_name,
            "where": where_clause
        }

        return pg_query, ir

    def _extract_columns(self, parsed_query) -> list:
        """Extract column names from the SELECT clause."""
        for token in parsed_query.tokens:
            if token.ttype is sqlparse.tokens.DML and token.value.upper() == "SELECT":
                next_token = token.get_next_sibling()
                if next_token:
                    return [col.strip() for col in str(next_token).split(",")]
        raise ValueError("No SELECT clause found in query.")

    def _extract_table_name(self, parsed_query) -> str:
        """Extract table name from the FROM clause."""
        for token in parsed_query.tokens:
            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "FROM":
                next_token = token.get_next_sibling()
                if next_token:
                    return str(next_token).strip()
        raise ValueError("No FROM clause found in query.")

    def _extract_where_clause(self, parsed_query) -> Dict[str, Any]:
        """Extract WHERE clause conditions."""
        for token in parsed_query.tokens:
            if isinstance(token, sqlparse.sql.Where):
                condition_str = str(token).replace("WHERE", "").strip()
                return self._parse_condition(condition_str)
        return None

    def _parse_condition(self, condition_str: str) -> Dict[str, Any]:
        """Parse a single condition into structured format."""
        if " AND " in condition_str:
            left, right = condition_str.split(" AND ", 1)
            return {
                "operator": "AND",
                "left": self._parse_condition(left),
                "right": self._parse_condition(right)
            }
        
        if " OR " in condition_str:
            left, right = condition_str.split(" OR ", 1)
            return {
                "operator": "OR",
                "left": self._parse_condition(left),
                "right": self._parse_condition(right)
            }
        
        if "=" in condition_str:
            left, right = condition_str.split("=", 1)
            return {"operator": "=", "left": left.strip(), "right": right.strip()}
        
        if "<" in condition_str:
            left, right = condition_str.split("<", 1)
            return {"operator": "<", "left": left.strip(), "right": right.strip()}
        
        if ">" in condition_str:
            left, right = condition_str.split(">", 1)
            return {"operator": ">", "left": left.strip(), "right": right.strip()}
        
        raise ValueError(f"Unsupported condition format: {condition_str}")
