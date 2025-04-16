import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame, Series

from .schemas import Employee


class DepartmentSalary(pa.DataFrameModel):
    department: Series[str] = pa.Field(
        isin=["Engineering", "Marketing", "HR", "Finance"]
    )
    avg_salary: Series[float] = pa.Field()


def get_avg_salary_by_department(
    df: DataFrame[Employee],
) -> DataFrame[DepartmentSalary]:
    """Calculate average salary by department using schema attributes."""
    return DataFrame[DepartmentSalary](
        df.sort(Employee.department)
        .group_by(Employee.department)
        .agg(pl.col(Employee.salary).mean().alias(DepartmentSalary.avg_salary))
    )
