from typing import Dict, Any, List
import psycopg2
from psycopg2 import sql

class DatabaseAPI:
    """
    A Database API for connecting to PostgreSQL, validating queries,
    executing queries, and converting results into Intermediate Representation (IR).
    """

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.connection = None

    def connector(self):
        """Establishes a connection to the database."""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            print("Successfully connected to the database.")
            return self.connection
        except psycopg2.OperationalError as e:
            print(f"Error connecting to the database: {e}")
            return None

    def query_validator(self, query: str) -> bool:
        """Validates the SQL query."""
        valid_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE']
        return any(query.strip().upper().startswith(keyword) for keyword in valid_keywords)

    def query_executor(self, query: str, params: tuple = None) -> Any:
        """Executes the validated SQL query."""
        if not self.query_validator(query):
            raise ValueError("Invalid SQL query")

        if not self.connection:
            self.connector()

        try:
            with self.connection.cursor() as cursor:
                if params:
                    cursor.execute(sql.SQL(query), params)
                else:
                    cursor.execute(sql.SQL(query))

                if query.strip().upper().startswith('SELECT'):
                    # Fetch column names
                    column_names = [desc[0] for desc in cursor.description]
                    # Fetch all results
                    raw_results = cursor.fetchall()
                    # Convert results to IR
                    return self.result_to_ir(query, raw_results, column_names)  # Pass column names
                else:
                    self.connection.commit()
                    return cursor.rowcount
        except psycopg2.Error as e:
            print(f"Error executing query: {e}")
            self.connection.rollback()
            return None

    def result_to_ir(self, query: str, raw_results: List[tuple], column_names: List[str]) -> Dict[str, Any]:
        """
        Converts raw results from PostgreSQL into Intermediate Representation (IR).
        Args:
            query (str): The original SQL query.
            raw_results (List[tuple]): Raw results fetched from PostgreSQL.
            column_names (List[str]): List of column names from the query.
        Returns:
            Dict[str, Any]: Intermediate representation of the query results.
        """
        # Structure IR
        ir = {
            "columns": column_names,
            "table": self._parse_table_name(query),
            "where": self._parse_where_clause(query),
            "data": {col: [row[i] for row in raw_results] for i, col in enumerate(column_names)}
        }

        return ir

    def _parse_select_clause(self, query: str) -> List[str]:
        """Extract column names from the SELECT clause."""
        select_start = query.upper().find("SELECT") + len("SELECT")
        from_start = query.upper().find("FROM")

        columns_str = query[select_start:from_start].strip()
        return [col.strip() for col in columns_str.split(",")]

    def _parse_table_name(self, query: str) -> str:
        """Extract table name from the FROM clause."""
        from_start = query.upper().find("FROM") + len("FROM")
        where_start = query.upper().find("WHERE")

        if where_start == -1:
            table_name = query[from_start:].strip()
        else:
            table_name = query[from_start:where_start].strip()

        return table_name.split()[0]  # Handle cases like "FROM table AS alias"

    def _parse_where_clause(self, query: str) -> Dict[str, Any]:
        """Extract WHERE clause conditions."""
        where_start = query.upper().find("WHERE")

        if where_start == -1:
            return None

        condition_str = query[where_start + len("WHERE"):].strip()

        # Simple parsing for basic conditions (e.g., "column = value AND column < value")
        conditions = []

        if "AND" in condition_str or "OR" in condition_str:
            operators = ["AND", "OR"]

            for operator in operators:
                if operator in condition_str:
                    left, right = condition_str.split(operator, 1)
                    conditions.append({
                        "operator": operator,
                        "left": self._parse_condition(left.strip()),
                        "right": self._parse_condition(right.strip())
                    })
                    break
        else:
            conditions.append(self._parse_condition(condition_str))

        return {"operator": "AND", "operands": conditions}  # Default to AND for simplicity

    def _parse_condition(self, condition_str: str) -> Dict[str, Any]:
        """Parse individual condition into structured format."""
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

    def close_connection(self):
        """Closes the database connection."""
        if self.connection:
            self.connection.close()
            print("Database connection closed.")
