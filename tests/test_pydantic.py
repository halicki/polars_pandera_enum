# Tests for Pydantic integration with Pandera DataFrame schemas

from typing import Any
import pytest
import pydantic
import polars as pl


# Test data shared across test cases
valid_dict: dict[str, Any] = {"x": 1, "df": {"str_col": ["hello", "world"]}}
invalid_dict: dict[str, Any] = {"x": 1, "df": {"str_col": ["hello", "hello"]}}


def test_work_with_pandas():
    """Test that Pandera's default pandas integration works."""
    import pandera as pa
    from pandera.typing import DataFrame, Series

    class SimpleSchema(pa.DataFrameModel):
        str_col: Series[str] = pa.Field(unique=True)

    class PydanticModel(pydantic.BaseModel):
        x: int
        df: DataFrame[SimpleSchema]

    # Valid data should work
    PydanticModel.model_validate(valid_dict)

    # Invalid data should raise ValidationError
    with pytest.raises(pydantic.ValidationError):
        PydanticModel.model_validate(invalid_dict)


@pytest.mark.xfail(reason="Native DataFrame[Schema] without validation will fail")
def test_work_with_polars_original():
    """Test that direct use of DataFrame[Schema] for polars fails without decorator."""
    import pandera.polars as pa
    from pandera.typing.polars import DataFrame, Series

    class SimpleSchema(pa.DataFrameModel):
        str_col: Series[str] = pa.Field(unique=True)

    class PydanticModel(pydantic.BaseModel):
        x: int
        df: DataFrame[SimpleSchema]  # This will fail without validation

        model_config = {"arbitrary_types_allowed": True}

    # Fails because there's no proper validation/conversion
    PydanticModel.model_validate(valid_dict)

    with pytest.raises(pydantic.ValidationError):
        PydanticModel.model_validate(invalid_dict)


def test_work_with_polars_decorated():
    """Test using the pandera_polars_model decorator for validation."""
    import pandera.polars as pa
    from pandera.typing.polars import DataFrame, Series
    from polars_pandera_enum.pydantic_integration import pandera_polars_model

    class SimpleSchema(pa.DataFrameModel):
        str_col: Series[str] = pa.Field(unique=True)

    # Apply the decorator to enable validation
    @pandera_polars_model
    class PydanticModel(pydantic.BaseModel):
        x: int
        df: DataFrame[SimpleSchema]  # Works with the decorator

        model_config = {"arbitrary_types_allowed": True}

    # Valid dictionary input should work
    model = PydanticModel.model_validate(valid_dict)
    assert isinstance(model.df, pl.DataFrame)
    assert model.df.shape == (2, 1)

    # Invalid dictionary input should raise ValidationError
    with pytest.raises(pydantic.ValidationError):
        PydanticModel.model_validate(invalid_dict)

    # Test with direct DataFrame input
    df = pl.DataFrame({"str_col": ["hello", "world"]})
    model = PydanticModel.model_validate({"x": 1, "df": df})
    assert isinstance(model.df, pl.DataFrame)

    # Test with invalid DataFrame
    invalid_df = pl.DataFrame({"str_col": ["hello", "hello"]})
    with pytest.raises(pydantic.ValidationError):
        PydanticModel.model_validate({"x": 1, "df": invalid_df})


def test_multiple_dataframes_with_decorator():
    """Test using the decorator with multiple DataFrame fields of different schemas."""
    import pandera.polars as pa
    from pandera.typing.polars import DataFrame, Series
    from polars_pandera_enum.pydantic_integration import pandera_polars_model

    class UserSchema(pa.DataFrameModel):
        user_id: Series[int] = pa.Field(ge=1)
        username: Series[str] = pa.Field(unique=True)

    class OrderSchema(pa.DataFrameModel):
        order_id: Series[int] = pa.Field(ge=1000)
        user_id: Series[int] = pa.Field(ge=1)
        amount: Series[float] = pa.Field(ge=0)

    # Apply decorator to a model with multiple DataFrame fields
    @pandera_polars_model
    class AppData(pydantic.BaseModel):
        users: DataFrame[UserSchema]
        orders: DataFrame[OrderSchema]
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
            "username": ["alice", "bob", "bob"],  # duplicate username
        },
        "orders": {
            "order_id": [1001, 1002, 1003],
            "user_id": [1, 2, 1],
            "amount": [10.5, 20.0, 15.75],
        },
        "app_version": "1.0.0",
    }

    with pytest.raises(pydantic.ValidationError):
        AppData.model_validate(invalid_app_data)

    # Test with invalid order data (negative user_id)
    invalid_order_data = {
        "users": {"user_id": [1, 2, 3], "username": ["alice", "bob", "carol"]},
        "orders": {
            "order_id": [1001, 1002, 1003],
            "user_id": [1, -2, 1],  # negative user_id
            "amount": [10.5, 20.0, 15.75],
        },
        "app_version": "1.0.0",
    }

    with pytest.raises(pydantic.ValidationError):
        AppData.model_validate(invalid_order_data)


def test_validator_handles_non_dict_df_values():
    """Test that the validator properly handles non-dict, non-DataFrame values."""
    import pandera.polars as pa
    from pandera.typing.polars import Series
    from polars_pandera_enum.pydantic_integration import validate_df_with_schema

    class SimpleSchema(pa.DataFrameModel):
        str_col: Series[str] = pa.Field(unique=True)

    # Invalid type should raise ValueError
    with pytest.raises(ValueError):
        validate_df_with_schema("not a dict or dataframe", SimpleSchema)


def test_config_preservation():
    """Test that the decorator preserves existing model_config values."""
    import pandera.polars as pa
    from pandera.typing.polars import DataFrame, Series
    from polars_pandera_enum.pydantic_integration import pandera_polars_model

    class SimpleSchema(pa.DataFrameModel):
        str_col: Series[str] = pa.Field(unique=True)

    # Define model with custom config
    @pandera_polars_model
    class ModelWithConfig(pydantic.BaseModel):
        df: DataFrame[SimpleSchema]

        model_config = {
            "arbitrary_types_allowed": True,
            "extra": "forbid",
            "validate_assignment": True,
        }

    # Check that config is preserved
    assert ModelWithConfig.model_config["extra"] == "forbid"
    assert ModelWithConfig.model_config["validate_assignment"] is True
    assert ModelWithConfig.model_config["arbitrary_types_allowed"] is True
