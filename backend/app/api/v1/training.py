from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user
from app.config.settings import settings
from app.core.contracts import error_response, success_response
from app.database.session import SessionLocal, get_db
from app.models.orm import Dataset, Model, ModelVersion, TrainingJob, User
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
    dataset_id: str | None = Field(None, description="ID of an uploaded dataset; optional for dry-run jobs")
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
    dataset = None
    if cfg.dataset_id:
        dataset = db.get(Dataset, cfg.dataset_id)
        if not dataset or dataset.is_deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_response("NOT_FOUND", "Dataset not found"))
        if dataset.owner_id != current_user.id and not dataset.is_public:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=error_response("FORBIDDEN", "Dataset access denied"))

    base_model_path = cfg.base_model
    if dataset and not settings.allow_remote_models:
        model_path = Path(cfg.base_model)
        if not model_path.exists():
            alt_path = settings.model_root / cfg.base_model
            if not alt_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=error_response("MODEL_NOT_FOUND", "Base model not found locally"),
                )
            base_model_path = str(alt_path)

    job_record = TrainingJob(
        owner_id=current_user.id,
        run_name=cfg.run_name,
        base_model=base_model_path,
        dataset_id=cfg.dataset_id,
        method=cfg.method,
        config_snapshot=cfg.model_dump(),
        notes=cfg.notes,
        status="queued",
    )
    db.add(job_record)
    db.commit()
    db.refresh(job_record)

    bg_job = None
    if dataset:
        async def _train_job(job):
            from trainer.finetune.engine import FinetuneConfig, run_finetune
            import asyncio

            log_dir = settings.log_root / "training"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{job_record.id}.log"

            def _append_log(message: str) -> None:
                timestamp = datetime.now(UTC).isoformat()
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"[{timestamp}] {message}\n")

            def _update_job(**fields):
                with SessionLocal() as s:
                    rec = s.get(TrainingJob, job_record.id)
                    if not rec:
                        return
                    for key, value in fields.items():
                        setattr(rec, key, value)
                    s.commit()

            _update_job(status="running", started_at=datetime.now(UTC), log_path=str(log_path))
            _append_log("training started")

            cfg_payload = cfg.model_dump()
            finetune_cfg = FinetuneConfig(
                run_name=cfg.run_name,
                base_model=base_model_path,
                dataset_path=dataset.storage_path,
                output_dir=str(settings.checkpoint_root / "finetune"),
                method=cfg.method,
                quantization_type=cfg.quantization_type,
                max_seq_length=cfg.max_seq_length,
                learning_rate=cfg.learning_rate,
                batch_size=cfg.batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                epochs=cfg.epochs,
                warmup_steps=cfg.warmup_steps,
                save_steps=cfg.save_steps,
                logging_steps=cfg.logging_steps,
                optimizer=cfg.optimizer,
                scheduler=cfg.scheduler,
                lora_rank=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.target_modules,
                mixed_precision=cfg.mixed_precision,
                gradient_checkpointing=cfg.gradient_checkpointing,
                seed=cfg.seed,
                allow_remote_model=settings.allow_remote_models,
            )

            def _progress(fraction: float, metrics: dict[str, object]):
                job.progress = fraction
                _update_job(progress=fraction, metrics={**metrics, "latest_step": metrics.get("step")})
                _append_log(f"progress {fraction:.2%} | {metrics}")

            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, run_finetune, finetune_cfg, _progress
                )
            except asyncio.CancelledError:
                _update_job(status="cancelled", completed_at=datetime.now(UTC))
                _append_log("training cancelled")
                raise
            except Exception as exc:  # noqa: BLE001
                _update_job(status="failed", error=str(exc), completed_at=datetime.now(UTC))
                _append_log(f"training failed: {exc}")
                raise

            with SessionLocal() as s:
                rec = s.get(TrainingJob, job_record.id)
                if rec:
                    rec.status = "completed"
                    rec.progress = 1.0
                    rec.metrics = result
                    rec.adapter_path = result.get("adapter_path", "")
                    rec.checkpoint_path = result.get("output_dir", "")
                    rec.completed_at = datetime.now(UTC)
                    model = Model(
                        owner_id=current_user.id,
                        name=cfg.run_name,
                        description=cfg.notes,
                        base_model=base_model_path,
                        model_type="adapter",
                    )
                    s.add(model)
                    s.flush()
                    version = ModelVersion(
                        model_id=model.id,
                        version=1,
                        adapter_path=rec.adapter_path,
                        config_snapshot=cfg_payload,
                        metrics=result,
                    )
                    s.add(version)
                    s.commit()
            _append_log("training completed")
            return {"status": "completed", "db_job_id": job_record.id}

        bg_job = get_job_queue().submit(_train_job, job_type="training")
        job_record.bg_job_id = bg_job.id
        db.commit()

    return success_response(
        "training job queued",
        {
            "id": job_record.id,
            "bg_job_id": bg_job.id if bg_job else "",
            "run_name": job_record.run_name,
            "state": "queued",
            "base_model": base_model_path,
            "method": cfg.method,
            "dataset_id": cfg.dataset_id,
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
                    "progress": j.progress,
                    "dataset_id": j.dataset_id,
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
            "progress": job.progress,
            "config_snapshot": job.config_snapshot,
            "metrics": job.metrics,
            "checkpoint_path": job.checkpoint_path,
            "adapter_path": job.adapter_path,
            "log_path": job.log_path,
            "error": job.error,
            "notes": job.notes,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        },
    )


@router.get("/jobs/{job_id}/logs")
def get_training_logs(
    job_id: str,
    tail: int = Query(200, ge=1, le=2000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    job = db.get(TrainingJob, job_id)
    if not job or job.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_response("NOT_FOUND", "Job not found"))
    if not job.log_path:
        return success_response("training logs fetched", {"lines": []})
    log_path = Path(job.log_path)
    if not log_path.exists():
        return success_response("training logs fetched", {"lines": []})
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return success_response("training logs fetched", {"lines": lines[-tail:]})


@router.post("/jobs/{job_id}/cancel")
def cancel_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    job = db.get(TrainingJob, job_id)
    if not job or job.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_response("NOT_FOUND", "Job not found"))
    if not job.bg_job_id:
        return success_response("training job cancelled", {"id": job_id, "cancelled": False})
    cancelled = get_job_queue().cancel(job.bg_job_id)
    if cancelled:
        job.status = "cancelled"
        job.completed_at = datetime.now(UTC)
        db.commit()
    return success_response("training job cancelled", {"id": job_id, "cancelled": cancelled})


@router.get("/jobs/{job_id}/metrics")
def get_training_metrics(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    job = db.get(TrainingJob, job_id)
    if not job or job.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_response("NOT_FOUND", "Job not found"))
    return success_response("training metrics fetched", {"metrics": job.metrics, "progress": job.progress})
