"""Input model for using Pandera polars DataFrames with Pydantic."""
from pydantic import BaseModel
from pandera.typing.polars import DataFrame

from .schemas import Employee, Product
from .pydantic_integration import pandera_polars_model


@pandera_polars_model
class Inputs(BaseModel):
    """Input model using the decorator for validating polars DataFrames with Pandera schemas."""
    employee_data: DataFrame[Employee]
    product_data: DataFrame[Product]
    
    model_config = {"arbitrary_types_allowed": True}