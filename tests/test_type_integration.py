# Tests for Pydantic integration using the custom type approach

import pytest
import pydantic
import polars as pl
import pandera.polars as pa
from pandera.typing.polars import Series

from polars_pandera_enum import PolarsDataFrame


# Test data shared across test cases
valid_dict = {"str_col": ["hello", "world"]}
invalid_dict = {"str_col": ["hello", "hello"]}  # Duplicate values, violates unique constraint


class SimpleSchema(pa.DataFrameModel):
    """Test schema with a simple string column having a uniqueness constraint."""
    str_col: Series[str] = pa.Field(unique=True)


class UserSchema(pa.DataFrameModel):
    """Schema for user data."""
    user_id: Series[int] = pa.Field(ge=1)
    username: Series[str] = pa.Field(unique=True)


class OrderSchema(pa.DataFrameModel):
    """Schema for order data."""
    order_id: Series[int] = pa.Field(ge=1000)
    user_id: Series[int] = pa.Field(ge=1)
    amount: Series[float] = pa.Field(ge=0)


def test_polars_dataframe_basic() -> None:
    """Test basic functionality of PolarsDataFrame."""
    # Test creating from dictionary
    df = PolarsDataFrame[SimpleSchema](valid_dict)
    assert isinstance(df.df, pl.DataFrame)
    assert df.shape == (2, 1)
    assert df.columns == ["str_col"]
    
    # Test validation error
    with pytest.raises(ValueError):
        PolarsDataFrame[SimpleSchema](invalid_dict)
    
    # Test creating from DataFrame
    pl_df = pl.DataFrame(valid_dict)
    df = PolarsDataFrame[SimpleSchema](pl_df)
    assert df.shape == (2, 1)


def test_pydantic_integration_without_arbitrary_types() -> None:
    """Test that PolarsDataFrame works in Pydantic models without arbitrary_types_allowed."""
    class PydanticModel(pydantic.BaseModel):
        x: int
        df: PolarsDataFrame[SimpleSchema]
        # Note: No arbitrary_types_allowed=True needed!
    
    # Test with valid data
    model = PydanticModel.model_validate({"x": 1, "df": valid_dict})
    assert isinstance(model.df, PolarsDataFrame)
    assert model.df.shape == (2, 1)
    
    # Test with invalid data
    with pytest.raises(pydantic.ValidationError):
        PydanticModel.model_validate({"x": 1, "df": invalid_dict})


def test_multiple_dataframes() -> None:
    """Test using multiple PolarsDataFrame fields with different schemas."""
    class AppData(pydantic.BaseModel):
        users: PolarsDataFrame[UserSchema]
        orders: PolarsDataFrame[OrderSchema]
        app_version: str
    
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
    
    # Test with valid data
    model = AppData.model_validate(valid_app_data)
    assert isinstance(model.users, PolarsDataFrame)
    assert isinstance(model.orders, PolarsDataFrame)
    assert model.users.shape == (3, 2)
    assert model.orders.shape == (3, 3)
    
    # Test with invalid data (duplicate username)
    invalid_users_data = {
        "users": {"user_id": [1, 2, 3], "username": ["alice", "bob", "bob"]},
        "orders": valid_app_data["orders"],
        "app_version": "1.0.0",
    }
    
    with pytest.raises(pydantic.ValidationError):
        AppData.model_validate(invalid_users_data)


def test_dataframe_like_behavior() -> None:
    """Test that PolarsDataFrame behaves like a regular Polars DataFrame."""
    df = PolarsDataFrame[SimpleSchema](valid_dict)
    
    # Test basic DataFrame-like operations
    assert len(df) == 2
    assert df["str_col"].to_list() == ["hello", "world"]
    # Use schema information rather than df.dtypes
    assert df["str_col"].dtype == pl.Utf8
    
    # Test method delegation
    filtered = df.filter(pl.col("str_col") == "hello")
    assert isinstance(filtered, pl.DataFrame)
    assert len(filtered) == 1
    assert filtered["str_col"][0] == "hello"


def test_serialization() -> None:
    """Test serialization of PolarsDataFrame."""
    df = PolarsDataFrame[SimpleSchema](valid_dict)
    
    class PydanticModel(pydantic.BaseModel):
        df: PolarsDataFrame[SimpleSchema]
    
    model = PydanticModel(df=df)
    serialized = model.model_dump()
    
    # Check that the serialized data is a dictionary
    assert isinstance(serialized["df"], dict)
    assert serialized["df"]["str_col"] == ["hello", "world"]
    
    # Test deserialization
    deserialized = PydanticModel.model_validate(serialized)
    assert isinstance(deserialized.df, PolarsDataFrame)
    assert deserialized.df.shape == (2, 1)
    assert deserialized.df["str_col"].to_list() == ["hello", "world"]