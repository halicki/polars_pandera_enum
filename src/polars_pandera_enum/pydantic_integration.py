"""
Integration utilities for using Pandera Polars DataFrames with Pydantic models.
"""
import polars as pl
from typing import Any, Callable, TypeVar, get_origin, get_args, Annotated, Generic, ClassVar, Type
from pydantic import BeforeValidator, ConfigDict
from pandera.typing.polars import DataFrame
from pandera.errors import SchemaError
import pandera.polars as pa


T = TypeVar("T", bound=pa.DataFrameModel)


def validate_pandera_df(schema_type: Type[pa.DataFrameModel]) -> Callable[[Any], Any]:
    """Create a validator for a specific Pandera DataFrameModel schema type.
    
    Args:
        schema_type: The schema type to validate against.
        
    Returns:
        A validator function that can be used with BeforeValidator.
    """
    schema_cls = schema_type  # Rename to avoid mypy error with variable as type
    
    def validator(value: Any) -> Any:
        """Validate input is a valid Polars DataFrame that matches the schema."""
        if isinstance(value, dict):
            df = pl.DataFrame(value)
            try:
                # Use schema_cls instead of schema_type in type position
                return DataFrame[schema_cls](df)  # type: ignore
            except SchemaError as e:
                raise ValueError(f"Pandera validation error: {str(e)}")
        elif isinstance(value, pl.DataFrame):
            try:
                # Use schema_cls instead of schema_type in type position
                return DataFrame[schema_cls](value)  # type: ignore
            except SchemaError as e:
                raise ValueError(f"Pandera validation error: {str(e)}")
        return value
    
    return validator


class PanderaPolarsField(Generic[T]):
    """A field wrapper that validates Polars DataFrames with Pandera schemas.
    
    This class helps create Pydantic-compatible field types for DataFrame[Schema]
    annotations that automatically handle validation.
    
    Example:
        ```python
        from pydantic import BaseModel
        from polars_pandera_enum.pydantic_integration import PanderaPolarsField
        
        class UserSchema(pa.DataFrameModel):
            user_id: Series[int] = pa.Field(ge=1)
            username: Series[str] = pa.Field(unique=True)
        
        class MyModel(BaseModel):
            # Use the field with your schema
            users: PanderaPolarsField[UserSchema]
            
            model_config = {"arbitrary_types_allowed": True}
        ```
    """
    
    # Class attribute to store the Pandera schema type
    _schema_type: ClassVar[Type[pa.DataFrameModel]]
    
    def __class_getitem__(cls, schema_type: Type[T]) -> Type:
        """Create a validator field type specific to the given schema type."""
        if not issubclass(schema_type, pa.DataFrameModel):
            raise TypeError(f"Schema type must be a subclass of DataFrameModel, got {schema_type}")
            
        # Create a typed validator
        validator = validate_pandera_df(schema_type)
        
        # Create a new annotated type
        field_type = Annotated[
            Any,  # Use Any as the base type
            BeforeValidator(validator)
        ]
        
        # Set schema_type as class attribute so it's available for introspection
        field_type._schema_type = schema_type
        
        return field_type


def pandera_polars_model() -> Callable[[type], type]:
    """Class decorator that adds support for DataFrame[Schema] types in Pydantic models.
    
    This decorator:
    1. Sets arbitrary_types_allowed=True in the model config
    2. Converts DataFrame[Schema] annotations to use our validators
    
    Example:
        ```python
        from pydantic import BaseModel
        from pandera.typing.polars import DataFrame
        from polars_pandera_enum import pandera_polars_model
        
        class UserSchema(pa.DataFrameModel):
            user_id: Series[int] = pa.Field(ge=1)
            username: Series[str] = pa.Field(unique=True)
        
        @pandera_polars_model()
        class MyModel(BaseModel):
            # Use the original DataFrame[Schema] syntax
            users: DataFrame[UserSchema]
            # ...other fields...
        ```
    """
    def decorator(cls: Any) -> Any:
        # First, ensure arbitrary_types_allowed is set
        # Note: Using Any type to avoid mypy errors with dynamic attributes
        if not hasattr(cls, "model_config"):
            cls.model_config = ConfigDict(arbitrary_types_allowed=True)  # type: ignore
        elif isinstance(cls.model_config, dict) and "arbitrary_types_allowed" not in cls.model_config:
            cls.model_config["arbitrary_types_allowed"] = True  # type: ignore
        
        # Process annotations to replace DataFrame[Schema] with our validated type
        new_annotations = {}
        for field_name, annotation in getattr(cls, "__annotations__", {}).items():
            origin = get_origin(annotation)
            if origin is DataFrame:
                args = get_args(annotation)
                if args and issubclass(args[0], pa.DataFrameModel):
                    # Replace with our validated field type
                    schema_type = args[0]
                    validator = validate_pandera_df(schema_type)
                    new_annotations[field_name] = Annotated[Any, BeforeValidator(validator)]
                else:
                    new_annotations[field_name] = annotation
            else:
                new_annotations[field_name] = annotation
        
        # Update the class annotations
        if new_annotations:
            cls.__annotations__ = new_annotations
        
        return cls
    
    return decorator