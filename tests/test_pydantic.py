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


@pytest.mark.xfail(reason="Native DataFrame[Schema] without validation will fail")
def test_work_with_polars_original():
    """Original test that tries to use DataFrame[Schema] directly without validation - will fail."""
    import polars as pl
    import pandera.polars as pa
    from pandera.typing.polars import DataFrame, Series

    class SimpleSchema(pa.DataFrameModel):
        str_col: Series[str] = pa.Field(unique=True)

    class PydanticModel(pydantic.BaseModel):
        x: int
        df: DataFrame[SimpleSchema]  # This will fail without validation
        
        model_config = {"arbitrary_types_allowed": True}

    # These will fail because there's no proper validation/conversion
    PydanticModel.model_validate(valid_dict)
    
    with pytest.raises(pydantic.ValidationError):
        PydanticModel.model_validate(invalid_dict)


def test_work_with_polars_manual():
    """Test the direct approach with Annotated and BeforeValidator."""
    import polars as pl
    import pandera.polars as pa
    from pandera.typing.polars import DataFrame, Series
    from typing import Any, Annotated
    from pandera.errors import SchemaError

    class SimpleSchema(pa.DataFrameModel):
        str_col: Series[str] = pa.Field(unique=True)

    # Define a custom validator for DataFrame[SimpleSchema]
    def validate_pandera_df(value: Any) -> DataFrame[SimpleSchema]:
        if isinstance(value, dict):
            df = pl.DataFrame(value)
            try:
                return DataFrame[SimpleSchema](df)
            except SchemaError as e:
                raise ValueError(f"Pandera validation error: {str(e)}")
        elif isinstance(value, pl.DataFrame):
            try:
                return DataFrame[SimpleSchema](value)
            except SchemaError as e:
                raise ValueError(f"Pandera validation error: {str(e)}")
        return value

    # Use Annotated to attach validator to the type
    PanderaDataFrame = Annotated[
        DataFrame[SimpleSchema], pydantic.BeforeValidator(validate_pandera_df)
    ]

    class PydanticModel(pydantic.BaseModel):
        x: int
        df: PanderaDataFrame

        model_config = {"arbitrary_types_allowed": True}

    model = PydanticModel.model_validate(valid_dict)
    assert isinstance(model.df, pl.DataFrame)

    # This should raise a ValidationError because the pandera validator catches the duplicate
    with pytest.raises(pydantic.ValidationError):
        PydanticModel.model_validate(invalid_dict)


def test_work_with_polars_field():
    """Test using our PanderaPolarsField for cleaner integration."""
    import polars as pl
    import pandera.polars as pa
    from pandera.typing.polars import Series
    from polars_pandera_enum.pydantic_integration import PanderaPolarsField

    class SimpleSchema(pa.DataFrameModel):
        str_col: Series[str] = pa.Field(unique=True)

    class PydanticModel(pydantic.BaseModel):
        x: int
        df: PanderaPolarsField[SimpleSchema]  # Use our field type

        model_config = {"arbitrary_types_allowed": True}

    model = PydanticModel.model_validate(valid_dict)
    assert isinstance(model.df, pl.DataFrame)

    with pytest.raises(pydantic.ValidationError):
        PydanticModel.model_validate(invalid_dict)


def test_multi_dataframe_model():
    """Test a model with multiple dataframes of different schemas."""
    import polars as pl
    import pandera.polars as pa
    from pandera.typing.polars import Series
    from polars_pandera_enum.pydantic_integration import PanderaPolarsField

    class UserSchema(pa.DataFrameModel):
        user_id: Series[int] = pa.Field(ge=1)
        username: Series[str] = pa.Field(unique=True)

    class OrderSchema(pa.DataFrameModel):
        order_id: Series[int] = pa.Field(ge=1000)
        user_id: Series[int] = pa.Field(ge=1)
        amount: Series[float] = pa.Field(ge=0)

    class AppData(pydantic.BaseModel):
        users: PanderaPolarsField[UserSchema]
        orders: PanderaPolarsField[OrderSchema]
        app_version: str

        model_config = {"arbitrary_types_allowed": True}

    # Valid data for both dataframes
    valid_app_data = {
        "users": {"user_id": [1, 2, 3], "username": ["alice", "bob", "carol"]},
        "orders": {
            "order_id": [1001, 1002, 1003],
            "user_id": [1, 2, 1],
            "amount": [10.5, 20.0, 15.75],
        },
        "app_version": "1.0.0",
    }

    # Test valid data works
    model = AppData.model_validate(valid_app_data)
    assert isinstance(model.users, pl.DataFrame)
    assert isinstance(model.orders, pl.DataFrame)

    # Test with invalid data (duplicate username)
    invalid_app_data = {
        "users": {
            "user_id": [1, 2, 3],
            "username": ["alice", "bob", "bob"],
        },  # duplicate username
        "orders": {
            "order_id": [1001, 1002, 1003],
            "user_id": [1, 2, 1],
            "amount": [10.5, 20.0, 15.75],
        },
        "app_version": "1.0.0",
    }

    with pytest.raises(pydantic.ValidationError):
        AppData.model_validate(invalid_app_data)
