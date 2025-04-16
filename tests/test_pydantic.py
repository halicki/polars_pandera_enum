# https://pandera.readthedocs.io/en/latest/pydantic_integration.html#pydantic-integration


from typing import Any
import pytest
import pydantic


valid_dict: dict[str, Any] = {"x": 1, "df": {"str_col": ["hello", "world"]}}
invalid_dict: dict[str, Any] = {"x": 1, "df": {"str_col": ["hello", "hello"]}}


def test_work_with_pandas():
    import pandera as pa
    from pandera.typing import DataFrame, Series

    class SimpleSchema(pa.DataFrameModel):
        str_col: Series[str] = pa.Field(unique=True)

    class PydanticModel(pydantic.BaseModel):
        x: int
        df: DataFrame[SimpleSchema]

    PydanticModel.model_validate(valid_dict)

    with pytest.raises(pydantic.ValidationError):
        PydanticModel.model_validate(invalid_dict)


def test_work_with_polars():
    import polars as pl
    import pandera.polars as pa
    from pandera.typing.polars import DataFrame, Series
    from typing import Any
    from pandera.errors import SchemaError

    class SimpleSchema(pa.DataFrameModel):
        str_col: Series[str] = pa.Field(unique=True)

    class PydanticModel(pydantic.BaseModel):
        x: int
        df: Any  # Use Any instead of DataFrame[SimpleSchema]
        
        model_config = {"arbitrary_types_allowed": True}
        
        @pydantic.field_validator("df", mode="before")
        @classmethod
        def validate_df(cls, value: Any) -> Any:
            if isinstance(value, dict):
                df = pl.DataFrame(value)
                try:
                    # Validate with SimpleSchema
                    return DataFrame[SimpleSchema](df)
                except SchemaError as e:
                    # Convert Pandera error to Pydantic error
                    raise ValueError(f"Pandera validation error: {str(e)}")
            return value

    model = PydanticModel.model_validate(valid_dict)
    assert isinstance(model.df, pl.DataFrame)
    
    # This should raise a ValidationError because the pandera validator catches the duplicate
    with pytest.raises(pydantic.ValidationError):
        PydanticModel.model_validate(invalid_dict)
