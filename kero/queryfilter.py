import re

def filter(query:str)->tuple[str, str]:
    """filter to get postgresql query for retriving data from database
    and getting kquery for performing tensor operations on kquery
    """
    table = get_table_name(query)
    column_names = get_column_names(query)
    pgquery = generate_pgquery(table, column_names)
    kquery = generate_kquery(query)
    return pgquery, kquery

def generate_pgquery(table:str, column_names:list[str])->str:
    """generates a pgquery from column names and table name
    """
    column_names_str = ", ".join(column_names)
    query = f"SELECT {column_names_str} FROM {table}"
    return query

def generate_kquery(query:str)->str:
    """depends on kerosine tensor strucutre
    """
    return query # will develop in future

def get_table_name(query):
    if " WHERE " in query:
        match = r"(?<=FROM)(.*)(?=WHERE)"
    else:
        match = r"(?<=FROM)(.*)(?=WHERE|$)"
    table_name = re.match(match, query)
    table_name = table_name.group(0)
    return table_name

def get_column_names(query:str)->list[str]:
    match = r"(?<=SELECT)(.*)(?=FROM)"
    query = re.search(match, query)
    columns = query.group(0)
    columns = columns.split()
    return columns
