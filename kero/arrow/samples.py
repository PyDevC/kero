import numpy as np
import pyarrow as pa

from . import data

def toy_school_dataset() -> data.Dataset:
    """
    Toy Dataset with Person table
    """
    name = ["prime", "teej", "begin", "bot", "source"]
    age = np.array([10, 20, 30, 40, 50], dtype=np.int8)
    
    person_dict = {
        "name": name,
        "age": age,
    }
    
    person = pa.Table.from_pydict(person_dict)
    
    tables = {
        "person": person
    }
    
    return data.Dataset(tables)

def all_number_dataset() -> data.Dataset:
    age = np.random.randint(10, 80, size=(100,), dtype=np.int32)
    salary = np.random.randint(10000, 80000, size=(100,), dtype=np.int32)
    spendings = np.random.randint(10000, 90000, size=(100,), dtype=np.int32)

    emp_dict = {
        "age": age,
        "salary": salary,
        "spendings": spendings,
    }

    emp = pa.Table.from_pydict(emp_dict)

    tables = {"employee": emp}

    return data.Dataset(tables)
