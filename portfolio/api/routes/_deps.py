"""Request-scoped access to the one service instance."""

from __future__ import annotations

from fastapi import Request

from ..services import PortfolioService


def service(request: Request) -> PortfolioService:
    return request.app.state.service
