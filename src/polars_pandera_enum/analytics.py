import pandera.polars as pa
import polars as pl
from pandera.typing.polars import Series

from .schemas import Employee
from .type_integration import PolarsDataFrame


class DepartmentSalary(pa.DataFrameModel):
    """Schema for department salary aggregation results."""
    department: Series[str] = pa.Field(
        isin=["Engineering", "Marketing", "HR", "Finance"]
    )
    avg_salary: Series[float] = pa.Field()


def get_avg_salary_by_department(
    df: PolarsDataFrame[Employee],
) -> PolarsDataFrame[DepartmentSalary]:
    """Calculate average salary by department using schema attributes."""
    # Extract the inner DataFrame if we have a PolarsDataFrame
    inner_df = df.df if hasattr(df, "df") else df
    
    result_df = (
        inner_df.sort(Employee.department)
        .group_by(Employee.department)
        .agg(pl.col(Employee.salary).mean().alias(DepartmentSalary.avg_salary))
    )
    
    return PolarsDataFrame[DepartmentSalary](result_df)
