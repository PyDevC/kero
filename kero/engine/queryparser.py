import sqlparse

class Parser:
    """
    A parser that converts SQL-like queries into intermediate representations (kquery).
    """

    def parse(self, query: str) -> Dict:
        """
        Parse an SQL-like query into an intermediate representation.

        Args:
            query (str): SQL-like query string.

        Returns:
            Dict: Intermediate representation (kquery).
        """
        parsed_query = sqlparse.parse(query)[0]
        
        # Extract SELECT clause
        columns = self._extract_columns(parsed_query)

        # Extract FROM clause (table name is ignored here as we work with TableTensor)
        
        # Extract WHERE clause
        where_clause = self._extract_where(parsed_query)

        return {
            "columns": columns,
            "where": where_clause,
        }

    def _extract_columns(self, parsed_query) -> List[str]:
        """Extract column names from the SELECT clause."""
        select_token = next(token for token in parsed_query.tokens if token.ttype is sqlparse.tokens.DML and token.value.upper() == "SELECT")
