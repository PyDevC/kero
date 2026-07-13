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

def all_number_dataset(size=100) -> data.Dataset:
    age = np.random.randint(10, 80, size=(size,), dtype=np.int32)
    salary = np.random.randint(10000, 80000, size=(size,), dtype=np.int32)
    spendings = np.random.randint(10000, 90000, size=(size,), dtype=np.int32)

    emp_dict = {
        "age": age,
        "salary": salary,
        "spendings": spendings,
    }

    emp = pa.Table.from_pydict(emp_dict)

    tables = {"employee": emp}

    return data.Dataset(tables)

def all_number_dataset(size=100) -> data.Dataset:
    age = np.random.randint(10, 80, size=(size,), dtype=np.int32)
    salary = np.random.randint(10000, 80000, size=(size,), dtype=np.int32)
    spendings = np.random.randint(10000, 90000, size=(size,), dtype=np.int32)
    department_id = np.random.randint(1, 10, size=(size,), dtype=np.int32)
    experience_years = np.random.randint(0, 40, size=(size,), dtype=np.int32)
    performance_rating = np.random.randint(1, 6, size=(size,), dtype=np.int32)
    is_active = np.random.randint(0, 2, size=(size,), dtype=np.int32)
    hire_year = np.random.randint(2000, 2027, size=(size,), dtype=np.int32)
    bonus_eligible = np.random.randint(0, 2, size=(size,), dtype=np.int32)
    manager_id = np.random.randint(100, 150, size=(size,), dtype=np.int32)
    region_id = np.random.randint(1, 6, size=(size,), dtype=np.int32)
    termination_year = np.random.choice([0, 2024, 2025, 2026], size=(size,), p=[0.8, 0.05, 0.1, 0.05]).astype(np.int32)
    position_level = np.random.randint(1, 6, size=(size,), dtype=np.int32)
    certification_count = np.random.randint(0, 5, size=(size,), dtype=np.int32)
    
    performance_score = np.random.uniform(0.0, 5.0, size=(size,)).astype(np.float32)
    commission_pct = np.random.uniform(0.0, 0.3, size=(size,)).astype(np.float32)
    hourly_rate = np.random.uniform(15.0, 150.0, size=(size,)).astype(np.float32)
    satisfaction_index = np.random.uniform(0.0, 1.0, size=(size,)).astype(np.float32)
    tax_rate = np.random.uniform(0.1, 0.4, size=(size,)).astype(np.float32)
    overtime_hours = np.random.uniform(0.0, 40.0, size=(size,)).astype(np.float32)

    emp_dict = {
        "age": age,
        "salary": salary,
        "spendings": spendings,
        "department_id": department_id,
        "experience_years": experience_years,
        "performance_rating": performance_rating,
        "is_active": is_active,
        "hire_year": hire_year,
        "bonus_eligible": bonus_eligible,
        "manager_id": manager_id,
        "region_id": region_id,
        "termination_year": termination_year,
        "position_level": position_level,
        "certification_count": certification_count,
        "performance_score": performance_score,
        "commission_pct": commission_pct,
        "hourly_rate": hourly_rate,
        "satisfaction_index": satisfaction_index,
        "tax_rate": tax_rate,
        "overtime_hours": overtime_hours,
    }

    emp = pa.Table.from_pydict(emp_dict)
    tables = {"employee": emp}

    return data.Dataset(tables)
