# Creating a product DataFrame
from typing import Any
from .schemas import Employee


employee_data: dict[str, Any] = {
    Employee.id: [1, 2, 3, 4, 5],
    Employee.name: ["Alice", "Bob", "Charlie", "Diana", "Evan"],
    Employee.age: [28, 35, 42, 31, 25],
    Employee.salary: [75000.0, 85000.0, 120000.0, 92000.0, 65000.0],
    Employee.department: [
        "Engineering",
        "Marketing",
        "Finance",
        "Engineering",
        "HR",
    ],
    Employee.start_date: [
        "2020-01-15",
        "2018-05-20",
        "2015-11-10",
        "2019-08-05",
        "2021-03-22",
    ],
    Employee.is_manager: [False, True, True, False, False],
}


employee_data_with_str: dict[str, Any] = {
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "Diana", "Evan"],
    "age": [28, 35, 42, 31, 25],
    "salary": [75000.0, 85000.0, 120000.0, 92000.0, 65000.0],
    "department": [
        "Engineering",
        "Marketing",
        "Finance",
        "Engineering",
        "HR",
    ],
    "start_date": [
        "2020-01-15",
        "2018-05-20",
        "2015-11-10",
        "2019-08-05",
        "2021-03-22",
    ],
    "is_manager": [False, True, True, False, False],
}
