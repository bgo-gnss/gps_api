"""Routers for the gps_api contract endpoints (docs/API_CONTRACT.md)."""

from typing import NoReturn

from fastapi import HTTPException


def not_implemented(what: str) -> NoReturn:
    """Single 501 stub for the not-yet-wired endpoints (uniform error shape)."""
    raise HTTPException(
        status_code=501,
        detail=f"{what} is not implemented yet — a later slice wires this endpoint to the store.",
    )
