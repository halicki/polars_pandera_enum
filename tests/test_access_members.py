import polars as pl

from polars_pandera_enum.analytics import (
    DepartmentSalary,
    get_avg_salary_by_department,
)
from polars_pandera_enum.data import employee_data
from polars_pandera_enum.schemas import Employee
from polars_pandera_enum import PolarsDataFrame


def test_can_access_member():
    df = PolarsDataFrame[Employee](employee_data)
    result = get_avg_salary_by_department(df)
    assert result.shape == (4, 2)
    assert result.columns == [DepartmentSalary.department, DepartmentSalary.avg_salary]
    assert result[DepartmentSalary.department].dtype == pl.Utf8
    assert result[DepartmentSalary.avg_salary].dtype == pl.Float64
    
    # Convert to dict for easy comparison
    result_dict = {}
    for dept in result[DepartmentSalary.department].to_list():
        salary = result.filter(pl.col(DepartmentSalary.department) == dept)[DepartmentSalary.avg_salary][0]
        result_dict[dept] = [float(salary)]
    
    assert result_dict == {
        "Engineering": [83500.0],
        "Finance": [120000.0],
        "HR": [65000.0],
        "Marketing": [85000.0],
    }
