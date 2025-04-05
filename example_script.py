from kero import all
import psycopg_api

database = psycopg_api.connector(db="dork.tar") # create a connection

# This is an example query
query = "SELECT employee_id, first_name, last_name, department, hire_date FROM employees WHERE department = 'Sales' AND hire_date < DATE_SUB(CURDATE(), INTERVAL 5 YEAR);"

# filter query that will be loaded in postgresql database
psquery, kquery = kero.queryfilter(query)

table = database.execute(psquery) # returns a tensor based on the query

result = table.execute(kquery)

print(result)
