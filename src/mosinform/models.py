from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CatalogObject:
    id: str
    kind: str
    name: str
    short: str
    aliases: list[str] = field(default_factory=list)
    head: str | None = None


@dataclass
class Person:
    id: str
    name: str
    short: str
    role: str = ""
    aliases: list[str] = field(default_factory=list)
    object_id: str | None = None


@dataclass
class Message:
    id: str
    source_raw: str
    source: str
    contour: str
    published_at: datetime | None
    title: str
    text: str
    url: str
    file_name: str
    city: str = ""
    object_ids: list[str] = field(default_factory=list)
    speaker_ids: list[str] = field(default_factory=list)
    sentiment: str | None = None
    role_by_object: dict[str, str] = field(default_factory=dict)
    initiated: bool | None = None
    topics: list[str] = field(default_factory=list)
    classified_by: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def body(self) -> str:
        return f"{self.title}\n{self.text}".strip()


@dataclass
class ObjectStats:
    object_id: str
    messages: int = 0
    reach: float | None = None
    media_index: float | None = None
    main_role: int = 0
    episodic_role: int = 0
    background_role: int = 0
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    initiated: int = 0
    initiated_known: int = 0
    unique_media: int = 0
    top2_share: float = 0.0
    speakers: dict[str, int] = field(default_factory=dict)
    media: dict[str, int] = field(default_factory=dict)
    daily: dict[str, int] = field(default_factory=dict)
    from_vendor: bool = False


@dataclass
class ReportBundle:
    period_label: str
    period_start: datetime | None
    period_end: datetime | None
    messages: list[Message]
    object_stats: list[ObjectStats]
    vendor_totals: dict[str, int]
    missing_metrics: list[str]
    notes: list[str]
