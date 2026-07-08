"""Routers for the gps_api contract endpoints (docs/API_CONTRACT.md)."""

from typing import NoReturn

from fastapi import HTTPException


def not_implemented(what: str) -> NoReturn:
    """Single 501 stub used by every endpoint until Phase 1 wires the store."""
    raise HTTPException(
        status_code=501,
        detail=f"{what} is not implemented yet — Phase 1 wires this endpoint to the store.",
    )
