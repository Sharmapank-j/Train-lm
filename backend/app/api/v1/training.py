from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user
from app.core.contracts import error_response, success_response
from app.database.session import get_db
from app.models.orm import TrainingJob, User
from app.workers.queue import get_job_queue

router = APIRouter(prefix="/training", tags=["training"])

SUPPORTED_MODELS = {
    "llama", "mistral", "gemma", "phi", "qwen",
    "falcon", "tinyllama", "deepseek",
}
SUPPORTED_OPTIMIZERS = {"adamw", "adam", "sgd", "adafactor"}
SUPPORTED_SCHEDULERS = {"cosine", "linear", "constant", "warmup_cosine"}
QUANTIZATION_TYPES = {"none", "4bit", "8bit"}


class TrainingConfig(BaseModel):
    run_name: str = Field(..., min_length=1, max_length=100)
    base_model: str = Field(..., min_length=1, max_length=255)
    dataset_id: str = Field(..., min_length=1)
    method: Literal["lora", "qlora"] = "lora"
    learning_rate: float = Field(2e-4, gt=0, le=1.0)
    batch_size: int = Field(4, ge=1, le=64)
    epochs: int = Field(3, ge=1, le=100)
    gradient_accumulation_steps: int = Field(4, ge=1, le=128)
    warmup_steps: int = Field(100, ge=0)
    max_seq_length: int = Field(2048, ge=64, le=32768)
    optimizer: str = Field("adamw")
    scheduler: str = Field("cosine")
    lora_rank: int = Field(8, ge=1, le=256)
    lora_alpha: int = Field(8, ge=1, le=512)
    lora_dropout: float = Field(0.1, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    quantization_type: str = Field("4bit")
    seed: int = Field(42, ge=0)
    save_steps: int = Field(100, ge=1)
    eval_steps: int = Field(100, ge=1)
    logging_steps: int = Field(10, ge=1)
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    notes: str = ""

    @field_validator("optimizer")
    @classmethod
    def check_optimizer(cls, v: str) -> str:
        if v not in SUPPORTED_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {SUPPORTED_OPTIMIZERS}")
        return v

    @field_validator("scheduler")
    @classmethod
    def check_scheduler(cls, v: str) -> str:
        if v not in SUPPORTED_SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SUPPORTED_SCHEDULERS}")
        return v

    @field_validator("quantization_type")
    @classmethod
    def check_quant(cls, v: str) -> str:
        if v not in QUANTIZATION_TYPES:
            raise ValueError(f"quantization_type must be one of {QUANTIZATION_TYPES}")
        return v


@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def queue_training_job(
    cfg: TrainingConfig,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    job_record = TrainingJob(
        owner_id=current_user.id,
        run_name=cfg.run_name,
        base_model=cfg.base_model,
        dataset_id=cfg.dataset_id,
        config_snapshot=cfg.model_dump(),
        notes=cfg.notes,
        status="queued",
    )
    db.add(job_record)
    db.commit()
    db.refresh(job_record)

    async def _train_stub(job):
        # Real training would invoke trainer/training/ here
        import asyncio
        job.progress = 0.1
        await asyncio.sleep(0)
        return {"status": "queued_for_training", "db_job_id": job_record.id}

    bg_job = get_job_queue().submit(_train_stub, job_type="training")

    return success_response(
        "training job queued",
        {
            "id": job_record.id,
            "bg_job_id": bg_job.id,
            "run_name": job_record.run_name,
            "state": "queued",
            "base_model": cfg.base_model,
            "method": cfg.method,
        },
    )


@router.get("/jobs")
def list_training_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    q = (
        db.query(TrainingJob)
        .filter(TrainingJob.owner_id == current_user.id)
        .order_by(TrainingJob.created_at.desc())
    )
    total = q.count()
    items = q.offset((page - 1) * page_size).limit(page_size).all()
    total_pages = max(1, (total + page_size - 1) // page_size)
    return success_response(
        "training jobs fetched",
        {
            "items": [
                {
                    "id": j.id,
                    "run_name": j.run_name,
                    "base_model": j.base_model,
                    "status": j.status,
                    "created_at": j.created_at.isoformat(),
                }
                for j in items
            ],
            "pagination": {
                "total_items": total,
                "current_page": page,
                "page_size": page_size,
                "next_page": page + 1 if page < total_pages else None,
                "previous_page": page - 1 if page > 1 else None,
                "total_pages": total_pages,
            },
        },
    )


@router.get("/jobs/{job_id}")
def get_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    job = db.get(TrainingJob, job_id)
    if not job or job.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_response("NOT_FOUND", "Job not found"))
    return success_response(
        "training job fetched",
        {
            "id": job.id,
            "run_name": job.run_name,
            "base_model": job.base_model,
            "status": job.status,
            "config_snapshot": job.config_snapshot,
            "metrics": job.metrics,
            "checkpoint_path": job.checkpoint_path,
            "adapter_path": job.adapter_path,
            "error": job.error,
            "notes": job.notes,
            "created_at": job.created_at.isoformat(),
        },
    )
