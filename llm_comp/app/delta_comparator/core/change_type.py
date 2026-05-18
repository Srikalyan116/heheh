"""This module handles different types of changes.The changes are categorized and managed within this module."""

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List


class ChangeType(BaseModel):
    """Model representing a type of change and its description.

    Attributes:
        change_type (str): The type of change.
        change_description (str): A description of the change.
    """

    change_type: str = Field(description="type of change")
    change_description: str = Field(
        description="description of changes between two revisions for the change_type"
    )


class Change(BaseModel):
    """Model representing the changes categorized into major and minor changes.

    Attributes:
        major_changes (List[ChangeType]): A list of major changes.
        minor_changes (List[ChangeType]): A list of minor changes.
        no_changes (str) : No change.
    """

    major_changes: List[ChangeType] = Field(
        description="""Major Change as defined in the definition of major change"""
    )
    minor_changes: List[ChangeType] = Field(
        description="""Minor Change as defined in the definition of minor change"""
    )
    no_changes: List[ChangeType] = Field(
        description="""No Change as defined in the definition of no change"""
    )


class AllChanges(BaseModel):
    """Model representing the changes categorized into major, minor changes, no changes.

    Attributes:
        major_changes (List[ChangeType]): A list of major changes.
        minor_changes (List[ChangeType]): A list of minor changes.
        no_changes (str) : No change.
    """

    all_changes: List[Change] = Field(
        description="List of Indentified changes for the given requirements."
    )
