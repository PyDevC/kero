import kero
import torch
from kero import QueryFilter
from kero.engine import Executor, KeroCompiler, Parser, operators
from psycopg_api.database import DatabaseAPI
from kero import TableTensor, NumTensor, StrTensor

queries = [
"SELECT employee_id, first_name, last_name, department, hire_date FROM employees WHERE department = 'Sales' AND hire_date < DATE_SUB(CURDATE(), INTERVAL 5 YEAR);",
"SELECT employee_id, first_name, last_name, department, hire_date FROM employees WHERE department = 'Sales' OR hire_date < DATE_SUB(CURDATE(), INTERVAL 10 YEAR);",
"SELECT employee_id, first_name, last_name, department FROM employees WHERE NOT department = 'HR';",
"SELECT SUM(salary) AS total_salary_expenditure FROM employees WHERE department = 'Marketing';",
"SELECT employee_id, first_name, last_name, salary, salary * 1.10 AS salary_with_bonus FROM employees WHERE salary > 50000;",
"SELECT employee_id, first_name, last_name, birth_date, (100 - TIMESTAMPDIFF(YEAR, birth_date, CURDATE())) AS years_to_100 FROM employees WHERE TIMESTAMPDIFF(YEAR, birth_date, CURDATE()) >= 30;",
"SELECT employee_id, first_name, last_name, salary, salary * 0.05 AS tax_deduction FROM employees WHERE salary BETWEEN 40000 AND 60000;",
"SELECT employee_id, first_name, last_name, salary, (salary * 1.15) AS new_salary FROM employees WHERE salary > 60000 AND TIMESTAMPDIFF(YEAR, birth_date, CURDATE()) BETWEEN 30 AND 40;",
]

db_config = {
    "dbname": "your_db_name",
    "user": "your_username",
    "password": "your_password",
    "host": "your_host",
    "port": "your_port"
}

# Step 2: Initialize the Database API
db_api = DatabaseAPI(db_config)

# Step 3: Connect to the database
connection = db_api.connector()

# Step 5: Validate the query
for query in queries:
    if not db_api.query_validator(query):
        raise ValueError("Invalid SQL query")

    raw_results = db_api.query_executor(query)
    print(f"Raw Results from Database: {raw_results}")
    columns = {
        "employee_id": NumTensor(torch.tensor([row[0] for row in raw_results]), name="employee_id"),
        "department": StrTensor([row[1] for row in raw_results], name="department")
    }

    table_tensor = TableTensor(columns=columns, name="employees")
    executor = Executor(table_tensor)

    kquery = {
        "columns": ["employee_id"],
        "where": {
            "operator": "=",
            "left": "department",
            "right": "'Sales'"
        }
    }

    result_tensor = executor.execute_query(kquery)
    print(f"Processed Results as Tensor:\n{result_tensor}")

db_api.close_connection()
