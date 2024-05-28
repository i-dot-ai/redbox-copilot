from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field


class PersistableModel(BaseModel):
    """Base class for all models that can be persisted to the database."""

    uuid: UUID = Field(default_factory=uuid4)
    created_datetime: datetime = Field(default_factory=datetime.utcnow)
    creator_user_uuid: UUID

    @computed_field  # type: ignore[misc] # Remove if https://github.com/python/mypy/issues/1362 is fixed.
    @property  # Needed for type checking - see https://docs.pydantic.dev/2.0/usage/computed_fields/
    def model_type(self) -> str:
        """Return the name of the model class.

        Returns:
            str: The name of the model class.
        """
        return self.__class__.__name__
