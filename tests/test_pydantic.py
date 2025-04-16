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
    import pandera.polars as pa
    from pandera.typing.polars import DataFrame, Series

    class SimpleSchema(pa.DataFrameModel):
        str_col: Series[str] = pa.Field(unique=True)

    class PydanticModel(pydantic.BaseModel):
        x: int
        df: DataFrame[SimpleSchema]

    PydanticModel.model_validate(valid_dict)

    with pytest.raises(pydantic.ValidationError):
        PydanticModel.model_validate(invalid_dict)
