"""
Integration utilities for using Pandera Polars DataFrames with Pydantic models.
"""

import polars as pl
import inspect
import dataclasses
from typing import (
    Any,
    TypeVar,
    Type,
    get_origin,
    get_args,
    Dict,
    ClassVar,
    Optional,
)
from pydantic import BaseModel, BeforeValidator, Field
from typing_extensions import Annotated
from pandera.typing.polars import DataFrame
from pandera.errors import SchemaError
import pandera.polars as pa


T = TypeVar("T", bound=pa.DataFrameModel)


def validate_df_with_schema(
    value: Any, schema_type: Type[pa.DataFrameModel]
) -> pl.DataFrame:
    """Validate a value using a Pandera schema, converting to DataFrame if needed.

    Args:
        value: The value to validate (dict or DataFrame)
        schema_type: The Pandera schema type to validate against

    Returns:
        A validated Polars DataFrame

    Raises:
        ValueError: If validation fails
    """
    if isinstance(value, dict):
        df = pl.DataFrame(value)
    elif isinstance(value, pl.DataFrame):
        df = value
    else:
        raise ValueError(f"Expected dict or DataFrame, got {type(value)}")

    try:
        # Validate using Pandera
        return DataFrame[schema_type](df)  # type: ignore
    except SchemaError as e:
        raise ValueError(f"Pandera validation error: {str(e)}")


@dataclasses.dataclass
class SchemaInfo:
    """Information about a schema for a model field."""

    schema_type: Type[pa.DataFrameModel]


class SchemaRegistry:
    """Registry for model schemas."""

    schemas: ClassVar[Dict[str, Dict[str, SchemaInfo]]] = {}

    @classmethod
    def register(
        cls, model_name: str, field_name: str, schema_type: Type[pa.DataFrameModel]
    ) -> None:
        """Register a schema for a model field."""
        if model_name not in cls.schemas:
            cls.schemas[model_name] = {}
        cls.schemas[model_name][field_name] = SchemaInfo(schema_type=schema_type)

    @classmethod
    def get_schema_info(cls, model_name: str, field_name: str) -> SchemaInfo:
        """Get schema info for a model field."""
        if model_name in cls.schemas and field_name in cls.schemas[model_name]:
            return cls.schemas[model_name][field_name]
        raise KeyError(f"No schema registered for {model_name}.{field_name}")


class PanderaDFField:
    """Generic field descriptor for DataFrame[Schema] fields."""

    field_name: str

    def __init__(self, field_name: str) -> None:
        self.field_name = field_name

    def __set__(self, instance: Any, value: Any) -> None:
        """Set the attribute value, validating with Pandera."""
        if value is None:
            instance.__dict__[self.field_name] = None
            return

        # Get the schema for this model field
        model_name = instance.__class__.__name__
        try:
            schema_info = SchemaRegistry.get_schema_info(model_name, self.field_name)
        except KeyError:
            # No schema registered, just set the value
            instance.__dict__[self.field_name] = value
            return

        # Validate the value with the schema
        validated_value = validate_df_with_schema(value, schema_info.schema_type)
        instance.__dict__[self.field_name] = validated_value

    def __get__(self, instance: Any, cls: Any) -> Any:
        """Get the attribute value."""
        if instance is None:
            return self
        return instance.__dict__.get(self.field_name, None)


def convert_to_dataframe(
    value: Any, schema_type: Optional[Type[pa.DataFrameModel]] = None
) -> pl.DataFrame:
    """Convert a value to a Polars DataFrame and validate it if a schema is provided.

    Args:
        value: The value to convert (dict or DataFrame)
        schema_type: Optional Pandera schema to validate against

    Returns:
        A Polars DataFrame
    """
    # Convert to DataFrame if it's a dict
    if isinstance(value, dict):
        df = pl.DataFrame(value)
    elif isinstance(value, pl.DataFrame):
        df = value
    else:
        raise ValueError(f"Expected dict or DataFrame, got {type(value)}")

    # Validate if schema_type is provided
    if schema_type is not None:
        return validate_df_with_schema(df, schema_type)
    return df


def validate_with_schema(v: Any, schema_cls: Type[pa.DataFrameModel]) -> pl.DataFrame:
    """Validator function for DataFrames with Pandera schemas.

    Args:
        v: The value to validate (can be a dict, DataFrame, or other type)
        schema_cls: The Pandera schema to validate against

    Returns:
        A validated Polars DataFrame

    Raises:
        ValueError: If the value can't be converted or validated
    """
    # Handle None
    if v is None:
        raise ValueError("Value cannot be None")

    # Convert to DataFrame if needed
    if isinstance(v, dict):
        df = pl.DataFrame(v)
    elif isinstance(v, pl.DataFrame):
        df = v
    else:
        raise ValueError(f"Cannot convert {type(v)} to a Polars DataFrame")

    # Validate with Pandera schema
    try:
        return DataFrame[schema_cls](df)  # type: ignore
    except SchemaError as e:
        raise ValueError(f"Pandera validation error: {str(e)}")


def pandera_polars_field(schema_cls: Type[pa.DataFrameModel]) -> Any:
    """Create an annotated field type for a Pandera schema validated Polars DataFrame.

    Args:
        schema_cls: The Pandera schema class to validate against

    Returns:
        An annotated field type that will convert and validate values
    """

    # Create a validator that captures the schema class
    def validator(v: Any) -> pl.DataFrame:
        return validate_with_schema(v, schema_cls)

    # Return an annotated type with the validator
    return Annotated[pl.DataFrame, BeforeValidator(validator)]


def pandera_polars_model(cls: Type[BaseModel]) -> Type[BaseModel]:
    """Class decorator that adds support for DataFrame[Schema] field validation in Pydantic models.

    This decorator transforms the model to use custom validators for DataFrame[Schema] fields.

    Args:
        cls: The Pydantic model class to decorate

    Returns:
        The decorated class with Pandera validation for DataFrame fields

    Example:
        ```python
        from pydantic import BaseModel
        from pandera.typing.polars import DataFrame
        from polars_pandera_enum import pandera_polars_model

        class UserSchema(pa.DataFrameModel):
            user_id: Series[int] = pa.Field(ge=1)
            username: Series[str] = pa.Field(unique=True)

        @pandera_polars_model
        class MyModel(BaseModel):
            # Use the original DataFrame[Schema] syntax
            users: DataFrame[UserSchema]
            app_name: str

            model_config = {"arbitrary_types_allowed": True}
        ```
    """

    # Make sure arbitrary_types_allowed is True
    model_config = getattr(cls, "model_config", {})
    model_config["arbitrary_types_allowed"] = True

    # Get annotations and fields
    fields = getattr(cls, "__annotations__", {})
    pandera_fields = {}

    # Find DataFrame[Schema] fields
    for field_name, field_type in fields.items():
        origin = get_origin(field_type)
        if origin is DataFrame:
            args = get_args(field_type)
            if (
                args
                and len(args) > 0
                and inspect.isclass(args[0])
                and issubclass(args[0], pa.DataFrameModel)
            ):
                schema_cls = args[0]
                # Register schema for later use
                SchemaRegistry.register(cls.__name__, field_name, schema_cls)
                # Mark this as a Pandera field
                pandera_fields[field_name] = schema_cls

    # If no Pandera fields, return the original class
    if not pandera_fields:
        return cls

    # Create a dictionary to hold field definitions for the new model
    new_annotations = {}
    field_defaults = {}

    # Process each field
    for name, field_type in fields.items():
        if name in pandera_fields:
            # For pandera fields, use our custom validator type
            new_annotations[name] = pandera_polars_field(pandera_fields[name])
            field_defaults[name] = Field()
        else:
            # Keep the original field as is
            new_annotations[name] = field_type

    # Create the new class that will replace the original
    new_cls_dict = {
        "__module__": cls.__module__,
        "__annotations__": new_annotations,
        "model_config": model_config,
        **{name: field_defaults.get(name, Field()) for name in new_annotations},
    }

    # Create the new class
    new_cls = type(cls.__name__, (cls,), new_cls_dict)

    # Copy other class attributes and methods
    for attr_name in dir(cls):
        if attr_name.startswith("__") and attr_name.endswith("__"):
            continue
        if attr_name in ("model_fields", "model_config", "__annotations__"):
            continue
        attr = getattr(cls, attr_name)
        if not hasattr(new_cls, attr_name):
            setattr(new_cls, attr_name, attr)

    return new_cls
