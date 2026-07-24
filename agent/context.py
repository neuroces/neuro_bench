"""Sample-scoped question context for Inspect tools.

A setup solver copies each sample's source/redaction into this store; the
Inspect ``@tool`` wrappers read it (via ``store_as``) to know which asset to
open and what to withhold. This is how per-question redaction reaches the tools.
"""

from __future__ import annotations

from inspect_ai.util import StoreModel
from pydantic import Field


class QuestionContext(StoreModel):
    dandi_id: str = Field(default="")
    version: str = Field(default="")
    asset_path: str = Field(default="")
    redaction: list[str] = Field(default_factory=list)
    channel: int = Field(default=0)
    window: dict | None = Field(default=None)
