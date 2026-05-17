from fastapi import APIRouter

from app.core.contracts import success_response

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
async def list_models() -> dict:
    return success_response("models fetched", {"items": []})
