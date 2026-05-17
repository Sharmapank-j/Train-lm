from fastapi import APIRouter

from app.core.contracts import success_response

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return success_response("ok", {"status": "healthy", "offline_first": True})
