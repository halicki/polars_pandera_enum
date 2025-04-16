"""Input model for using Pandera polars DataFrames with Pydantic."""
from pydantic import BaseModel

from .schemas import Employee, Product
from .type_integration import PolarsDataFrame


class Inputs(BaseModel):
    """Input model for validating polars DataFrames with Pandera schemas."""
    employee_data: PolarsDataFrame[Employee]
    product_data: PolarsDataFrame[Product]