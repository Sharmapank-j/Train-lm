from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user
from app.core.contracts import success_response
from app.database.session import SessionLocal, get_db
from app.models.orm import ExportJob, User
from app.workers.queue import get_job_queue

router = APIRouter(prefix="/exports", tags=["exports"])


class ExportRequest(BaseModel):
    model_id: str
    quantization: str = "Q4_K_M"


@router.post("/gguf")
async def queue_export(
    payload: ExportRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    job = ExportJob(
        owner_id=current_user.id,
        model_id=payload.model_id,
        quantization=payload.quantization,
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    async def _stub_export(bg_job):
        from asyncio import sleep
        bg_job.progress = 0.1
        await sleep(0)
        with SessionLocal() as session:
            rec = session.get(ExportJob, job.id)
            if rec:
                rec.status = "completed"
                rec.output_path = f"exports/{job.model_id}-{job.quantization}.gguf"
                rec.completed_at = datetime.now(UTC)
                session.commit()
        bg_job.progress = 1.0
        return {"job_id": job.id}

    bg_job = get_job_queue().submit(_stub_export, job_type="export")

    return success_response(
        "export queued",
        {"job_id": job.id, "bg_job_id": bg_job.id, "quantization": payload.quantization},
    )
