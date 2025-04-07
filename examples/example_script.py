from kero import queryfilter
import torch
from kero import TableTensor
import psycopg_api

database = psycopg_api.connector(db="dork.tar") # create a connection

# This is an example query
query = "SELECT employee_id, first_name, last_name, department, hire_date FROM employees WHERE department = 'Sales' AND hire_date < DATE_SUB(CURDATE(), INTERVAL 5 YEAR);"

# Constraint: the query should only be used for information retrival
# A query should mention at least a SELECT statement and FROM Table_name statement

# filter query that will be loaded in postgresql database
psquery, kquery = queryfilter.filter(query)

table = database.execute(psquery) # returns a TableTensor based on the query
# table is of kero.TableTensor type

device = 'cuda' if torch.cuda.is_available() else 'cpu'
result = table.execute(kquery, device=device)

# how does kquery work:
# kquery creates a map of operators and oprands that has to be performed in specific order
# execute only works for TableTensor
# TableTensor.execute calls the functions from query engine
# result is either a tensor class out of kero.tensors

print(result)
