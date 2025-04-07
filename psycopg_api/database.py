import psycopg2
from psycopg2 import sql
from typing import Dict, Any

class DatabaseAPI:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config

    def connector(self):
        """Establishes a connection to the database."""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            print("Successfully connected to the database.")
            return self.connection
        except psycopg2.OperationalError as e:
            raise ConnectionError(
                f"Error connecting to the database: {e}"
            )

    def query_validator(self, query: str) -> bool:
        """Validates the SQL query."""
        # Basic validation - check if the query starts with a valid SQL keyword
        valid_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE']
        return any(query.strip().upper().startswith(keyword) for keyword in valid_keywords)

    def query_executor(self, query: str, params: tuple=()) -> Any:
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
                    return cursor.fetchall()
                else:
                    self.connection.commit()
                    return cursor.rowcount
        except psycopg2.Error as e:
            print(f"Error executing query: {e}")
            self.connection.rollback()
            return None

    def close_connection(self):
        """Closes the database connection."""
        if self.connection:
            self.connection.close()
            print("Database connection closed.")
