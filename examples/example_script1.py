from kero.tensor import TableTensor, NumTensor

# Create a sample TableTensor
table_data = {
    "employee_id": NumTensor(torch.tensor([1, 2, 3]), name="employee_id"),
    "department": NumTensor(torch.tensor([0, 1, 0]), name="department"),  # Encoded as 0=Sales, 1=HR
}
table_tensor = TableTensor(columns=table_data, name="employees")

# Initialize executor
executor = KeroExecutor(table_tensor)

# Query to execute
query = """
SELECT employee_id 
FROM employees 
WHERE department = 0;
"""

# Execute query
result_tensor = executor.execute_query(query)
print(result_tensor)
