from fastapi import APIRouter
from pydantic import BaseModel

from app.core.contracts import success_response

router = APIRouter(prefix="/training", tags=["training"])


class TrainingJobCreateRequest(BaseModel):
    base_model: str
    dataset_id: str
    run_name: str
    method: str = "lora"


@router.post("/jobs")
async def queue_training_job(payload: TrainingJobCreateRequest) -> dict:
    return success_response(
        "training job queued",
        {
            "job_id": f"train-{payload.run_name}",
            "state": "queued",
            "base_model": payload.base_model,
            "dataset_id": payload.dataset_id,
            "method": payload.method,
        },
    )
