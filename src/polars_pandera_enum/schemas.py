import datetime as dt

import pandera.polars as pa
from pandera.typing.polars import Series


class Employee(pa.DataFrameModel):
    """Schema for employee data with direct field access."""

    # Define fields using normal Pandera syntax
    id: Series[int] = pa.Field(ge=1)
    name: Series[str] = pa.Field(description="Employee full name")
    age: Series[int] = pa.Field(ge=18, le=100)
    salary: Series[float] = pa.Field(ge=0)
    department: Series[str] = pa.Field(
        isin=["Engineering", "Marketing", "HR", "Finance"]
    )
    start_date: Series[dt.date] = pa.Field()
    is_manager: Series[bool] = pa.Field()

    class Config:
        coerce = True


class Product(pa.DataFrameModel):
    """Schema for product data."""
    
    id: Series[int] = pa.Field(ge=1)
    name: Series[str] = pa.Field()
    price: Series[float] = pa.Field(ge=0)
    category: Series[str] = pa.Field()
    in_stock: Series[bool] = pa.Field()
    
    class Config:
        coerce = True
