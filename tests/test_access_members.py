import polars as pl
from pandera.typing.polars import DataFrame

from polars_pandera_enum.analytics import (
    DepartmentSalary,
    get_avg_salary_by_department,
)
from polars_pandera_enum.data import employee_data
from polars_pandera_enum.schemas import Employee


def test_can_access_member():
    df = DataFrame[Employee](employee_data)
    result = get_avg_salary_by_department(df)
    assert result.shape == (4, 2)
    assert result.columns == [DepartmentSalary.department, DepartmentSalary.avg_salary]
    assert result[DepartmentSalary.department].dtype == pl.Utf8
    assert result[DepartmentSalary.avg_salary].dtype == pl.Float64
    assert result.transpose(column_names=DepartmentSalary.department).to_dict(
        as_series=False
    ) == {
        "Engineering": [83500.0],
        "Finance": [120000.0],
        "HR": [65000.0],
        "Marketing": [85000.0],
    }
