import pandera.typing.polars as pa
from pydantic import BaseModel

from .schemas import Employee, Product


class Inputs(BaseModel):
    employee_data: pa.DataFrame[Employee]
    product_data: pa.DataFrame[Product]
