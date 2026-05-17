from fastapi import APIRouter
from pydantic import BaseModel

from app.core.contracts import success_response

router = APIRouter(prefix="/exports", tags=["exports"])


class ExportRequest(BaseModel):
    model_id: str
    quantization: str = "Q4_K_M"


@router.post("/gguf")
async def queue_export(payload: ExportRequest) -> dict:
    return success_response("export queued", {"job_id": f"export-{payload.model_id}", "quantization": payload.quantization})
