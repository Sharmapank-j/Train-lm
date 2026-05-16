from fastapi import APIRouter
from pydantic import BaseModel

from app.core.contracts import success_response

router = APIRouter(prefix="/inference", tags=["inference"])


class ChatRequest(BaseModel):
    model_name: str
    prompt: str


@router.get("/models")
async def available_models() -> dict:
    return success_response("inference models fetched", {"items": []})


@router.post("/chat")
async def chat(payload: ChatRequest) -> dict:
    return success_response(
        "chat completion generated",
        {
            "model_name": payload.model_name,
            "completion": f"[offline-stub] {payload.prompt[:80]}",
            "session_id": "local-session-1",
        },
    )
