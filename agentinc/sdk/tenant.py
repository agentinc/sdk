from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TenantContext(BaseModel):
    tenant_id: str
    org_id: str | None = None
    quotas: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
