"""
Pydantic integration for Pandera Polars DataFrames using custom types.

This module provides custom types that work with Pydantic without requiring
arbitrary_types_allowed=True in the model config.
"""
from typing import Any, Type, TypeVar, Generic, ClassVar, get_args
import polars as pl
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from pandera.typing.polars import DataFrame
from pandera.errors import SchemaError
import pandera.polars as pa

# Create a type variable bound to DataFrameModel
T = TypeVar("T", bound=pa.DataFrameModel)


class PolarsDataFrame(Generic[T]):
    """
    A Pydantic-compatible generic type for Polars DataFrames validated with Pandera schemas.
    
    This type allows you to use DataFrame[Schema] annotations in Pydantic models
    without needing to set arbitrary_types_allowed=True.
    
    Example:
        ```python
        from pydantic import BaseModel
        from polars_pandera_enum import PolarsDataFrame
        
        class UserSchema(pa.DataFrameModel):
            user_id: Series[int] = pa.Field(ge=1)
            username: Series[str] = pa.Field(unique=True)
        
        class MyModel(BaseModel):
            # Use PolarsDataFrame with your schema
            users: PolarsDataFrame[UserSchema]
        ```
    """
    # Class variable to store the actual Pandera schema type
    _schema_type: ClassVar[Type[pa.DataFrameModel]]
    
    def __init__(self, value: Any):
        """
        Initialize with value, validating against the schema.
        
        Args:
            value: A dictionary, Polars DataFrame, or PolarsDataFrame
        
        Raises:
            ValueError: If validation fails
        """
        # Get schema type from class
        schema_type = self.__class__._schema_type
        
        # Convert and validate value
        if isinstance(value, dict):
            df = pl.DataFrame(value)
        elif isinstance(value, pl.DataFrame):
            df = value
        elif isinstance(value, PolarsDataFrame):
            # If we get a PolarsDataFrame instance, extract the DataFrame
            df = getattr(value, "df", pl.DataFrame())
        else:
            raise TypeError(f"Expected dict, DataFrame or PolarsDataFrame, got {type(value)}")
        
        # Validate with Pandera
        try:
            self.df = DataFrame[schema_type](df)  # type: ignore
        except SchemaError as e:
            raise ValueError(f"Pandera validation error: {str(e)}")
    
    @classmethod
    def __class_getitem__(cls, schema_type: Type[T]) -> Type["PolarsDataFrame[T]"]:
        """
        Create type with a specific Pandera schema.
        
        Args:
            schema_type: A Pandera DataFrameModel schema class
            
        Returns:
            A specialized PolarsDataFrame type
        """
        if not isinstance(schema_type, type) or not issubclass(schema_type, pa.DataFrameModel):
            raise TypeError(f"Schema type must be a DataFrameModel subclass, got {schema_type}")
        
        # Create new subclass with schema_type stored
        new_cls = type(
            f"PolarsDataFrame[{schema_type.__name__}]",
            (cls,),
            {"_schema_type": schema_type}
        )
        return new_cls
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, 
        _source_type: Any, 
        _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Create a Pydantic core schema for this type.
        
        Args:
            _source_type: Source type annotation
            _handler: Schema handler from Pydantic
            
        Returns:
            A Pydantic core schema that validates using our class
        """
        # If this is the generic base class, we need to access the actual schema type
        # from the parameterized class
        schema_args = get_args(_source_type)
        
        # Determine the validator class to use
        validator_cls: Type[PolarsDataFrame] = cls
        if schema_args:
            # Get first type argument which should be the schema
            schema_type = schema_args[0]
            # Create a schema validator for the parameterized type
            specialized = PolarsDataFrame[schema_type]  # type: ignore
            validator_cls = specialized
        
        # Define the serializer function
        def serialize_dataframe(instance: Any) -> dict:
            if not hasattr(instance, "df"):
                return {}
            
            # Convert to a Python dict with lists
            result = {}
            for col in instance.df.columns:
                result[col] = instance.df[col].to_list()
            return result
        
        # Define a validation function that can properly handle nested PolarsDataFrame instances
        def validate_df_field(obj: Any, info: Any) -> PolarsDataFrame:
            if isinstance(obj, validator_cls):
                # If it's already a PolarsDataFrame with the right schema, return it
                return obj
            elif isinstance(obj, PolarsDataFrame):
                # If it's a PolarsDataFrame but possibly with a different schema,
                # create a new instance with the right schema using the inner DataFrame
                return validator_cls(obj.df)
            else:
                # Otherwise, create a new instance with standard validation
                return validator_cls(obj)
        
        return core_schema.with_info_plain_validator_function(
            function=validate_df_field,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_dataframe,
            ),
            metadata={"title": f"PolarsPanderaDataFrame[{cls._schema_type.__name__}]"},
        )
    
    @classmethod
    def __get_validators__(cls):
        """Legacy Pydantic v1 validation support."""
        yield cls.validate
    
    @classmethod
    def validate(cls, value: Any) -> "PolarsDataFrame":
        """
        Validate a value against this schema type.
        
        Args:
            value: A dictionary or Polars DataFrame
            
        Returns:
            A validated PolarsDataFrame instance
            
        Raises:
            ValueError: If validation fails
        """
        return cls(value)
    
    def __repr__(self) -> str:
        """String representation of the DataFrame."""
        try:
            return f"PolarsDataFrame[{self.__class__._schema_type.__name__}]({self.df!r})"
        except AttributeError:
            return "PolarsDataFrame(uninitialized)"
    
    def __eq__(self, other: Any) -> bool:
        """Compare equality with another DataFrame."""
        if not isinstance(other, PolarsDataFrame):
            return False
        return self.df.equals(other.df)
    
    def to_dict(self) -> dict[str, list[Any]]:
        """Convert DataFrame to dictionary with Python lists."""
        # Custom implementation since to_dict() in Polars returns Series objects
        result = {}
        for col in self.df.columns:
            result[col] = self.df[col].to_list()
        return result
    
    # Special methods that need explicit implementation (can't be forwarded with __getattr__)
    def __getitem__(self, item: Any) -> Any:
        """Get item from the DataFrame."""
        return self.df.__getitem__(item)
    
    def __len__(self) -> int:
        """Return length of the DataFrame."""
        return len(self.df)
    
    # Forward everything else to the underlying DataFrame
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the DataFrame."""
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.df, name)