"""
Thin service wrapper around :class:`IMetricsRepository`.

Exists so routers depend on a service, not a repository directly - keeps
the layering rules consistent (routers -> services -> repositories).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.repositories.metrics_repository import IMetricsRepository


class MetricsService:
    def __init__(self, repo: IMetricsRepository) -> None:
        self._repo = repo

    async def summary(self) -> Dict[str, Dict[str, Any]]:
        return await self._repo.summary()

    async def recent(self, model: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        return await self._repo.recent(model=model, limit=limit)

    async def reset(self) -> None:
        await self._repo.clear()
