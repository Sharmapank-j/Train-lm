"""From-scratch LLM creation API.

Endpoints:
  GET  /pretrain/architectures          — list built-in presets
  GET  /pretrain/architectures/{name}   — inspect a preset
  POST /pretrain/architectures/validate — validate a custom architecture
  POST /pretrain/tokenizers             — queue a tokenizer-training job
  GET  /pretrain/tokenizers/{id}        — status of a tokenizer job
  POST /pretrain/jobs                   — queue a from-scratch pre-training job
  GET  /pretrain/jobs                   — list all pretrain jobs (owned)
  GET  /pretrain/jobs/{id}              — status of a pretrain job
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user
from app.core.contracts import error_response, success_response
from app.database.session import get_db
from app.models.orm import PretrainJob, TokenizerJob, User
from app.workers.queue import get_job_queue
from trainer.architecture.config import PRESETS, ArchitectureConfig
from trainer.tokenizer_train.train import TokenizerAlgorithm

router = APIRouter(prefix="/pretrain", tags=["pretrain"])

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ArchValidateRequest(BaseModel):
    """Validate a fully custom architecture config without creating a job."""
    arch_type: str = "llama"
    name: str
    hidden_size: int = 2048
    intermediate_size: int = 5504
    num_hidden_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    vocab_size: int = 32000
    max_position_embeddings: int = 4096


class TokenizerJobRequest(BaseModel):
    run_name: str = Field(..., min_length=1, max_length=100)
    corpus_path: str = Field(
        ...,
        description="Absolute or relative path to a plain-text or JSONL corpus file",
    )
    algorithm: TokenizerAlgorithm = "bpe"
    vocab_size: int = Field(32000, ge=256, le=256_000)
    min_frequency: int = Field(2, ge=1)
    byte_level: bool = True

    @field_validator("corpus_path")
    @classmethod
    def no_traversal(cls, v: str) -> str:
        if ".." in v:
            raise ValueError("Path traversal not allowed in corpus_path")
        return v


class PretrainJobRequest(BaseModel):
    run_name: str = Field(..., min_length=1, max_length=100)
    # Specify a built-in preset name OR a full architecture dict (not both)
    arch_preset: str = Field(
        "",
        description="Built-in preset name, e.g. 'TinyLM-15M', 'SmallLM-125M'",
    )
    architecture: dict = Field(
        default_factory=dict,
        description="Full ArchitectureConfig dict (used when arch_preset is empty)",
    )
    tokenizer_job_id: str = Field(
        "",
        description="ID of a completed TokenizerJob to use; leave blank for fallback",
    )
    corpus_path: str = Field(..., description="Path to pre-training corpus")
    max_seq_length: int = Field(2048, ge=64, le=32768)
    learning_rate: float = Field(3e-4, gt=0)
    batch_size: int = Field(4, ge=1, le=64)
    gradient_accumulation_steps: int = Field(8, ge=1)
    num_train_epochs: int = Field(1, ge=1, le=100)
    max_steps: int = Field(-1, description="-1 = use num_train_epochs")
    warmup_steps: int = Field(200, ge=0)
    save_steps: int = Field(500, ge=1)
    fp16: bool = False
    bf16: bool = False
    seed: int = Field(42, ge=0)
    notes: str = ""

    @field_validator("corpus_path")
    @classmethod
    def no_traversal(cls, v: str) -> str:
        if ".." in v:
            raise ValueError("Path traversal not allowed in corpus_path")
        return v


# ---------------------------------------------------------------------------
# Architecture endpoints
# ---------------------------------------------------------------------------

@router.get("/architectures")
def list_architectures() -> dict:
    """List all built-in architecture presets."""
    items = [
        {
            "name": name,
            "description": cfg.get("description", ""),
            "arch_type": cfg.get("arch_type", "llama"),
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "hidden_size": cfg.get("hidden_size"),
            "vocab_size": cfg.get("vocab_size"),
            "parameter_estimate": ArchitectureConfig(**cfg).parameter_estimate,
        }
        for name, cfg in PRESETS.items()
    ]
    return success_response("architecture presets fetched", {"items": items})


@router.get("/architectures/{preset_name}")
def get_architecture(preset_name: str) -> dict:
    """Return full details for a single architecture preset."""
    if preset_name not in PRESETS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("NOT_FOUND", f"Preset '{preset_name}' not found"),
        )
    arch = ArchitectureConfig(**PRESETS[preset_name])
    return success_response("architecture preset fetched", arch.to_dict())


@router.post("/architectures/validate")
def validate_architecture(payload: ArchValidateRequest) -> dict:
    """Validate a custom architecture config and return computed metadata."""
    try:
        arch = ArchitectureConfig(**payload.model_dump())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_response("INVALID_ARCHITECTURE", str(exc)),
        )
    return success_response("architecture valid", arch.to_dict())


# ---------------------------------------------------------------------------
# Tokenizer training endpoints
# ---------------------------------------------------------------------------

@router.post("/tokenizers", status_code=status.HTTP_201_CREATED)
async def queue_tokenizer_job(
    payload: TokenizerJobRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Queue a custom tokenizer training job."""
    job_record = TokenizerJob(
        owner_id=current_user.id,
        run_name=payload.run_name,
        algorithm=payload.algorithm,
        vocab_size=payload.vocab_size,
        corpus_path=payload.corpus_path,
        config_snapshot=payload.model_dump(),
        status="queued",
    )
    db.add(job_record)
    db.commit()
    db.refresh(job_record)

    async def _run_tokenizer(bg_job):
        import asyncio
        from pathlib import Path
        from trainer.tokenizer_train.train import TokenizerTrainConfig, train_tokenizer

        from app.config.settings import settings

        output_dir = settings.model_root / "tokenizers" / job_record.id
        cfg = TokenizerTrainConfig(
            algorithm=payload.algorithm,
            vocab_size=payload.vocab_size,
            min_frequency=payload.min_frequency,
            byte_level=payload.byte_level,
        )
        bg_job.progress = 0.1
        result = await asyncio.get_event_loop().run_in_executor(
            None, train_tokenizer, Path(payload.corpus_path), output_dir, cfg
        )
        # Update DB record
        from app.database.session import SessionLocal
        from datetime import datetime, UTC
        with SessionLocal() as s:
            rec = s.get(TokenizerJob, job_record.id)
            if rec:
                rec.status = "completed"
                rec.output_dir = result["output_dir"]
                rec.result = result
                rec.completed_at = datetime.now(UTC)
                s.commit()
        bg_job.progress = 1.0
        return result

    bg_job = get_job_queue().submit(_run_tokenizer, job_type="tokenizer")

    return success_response(
        "tokenizer job queued",
        {
            "id": job_record.id,
            "bg_job_id": bg_job.id,
            "run_name": payload.run_name,
            "algorithm": payload.algorithm,
            "vocab_size": payload.vocab_size,
            "status": "queued",
        },
    )


@router.get("/tokenizers/{job_id}")
def get_tokenizer_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    rec = db.get(TokenizerJob, job_id)
    if not rec or rec.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("NOT_FOUND", "Tokenizer job not found"),
        )
    return success_response(
        "tokenizer job fetched",
        {
            "id": rec.id,
            "run_name": rec.run_name,
            "status": rec.status,
            "algorithm": rec.algorithm,
            "vocab_size": rec.vocab_size,
            "output_dir": rec.output_dir,
            "result": rec.result,
            "error": rec.error,
            "created_at": rec.created_at.isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Pre-training job endpoints
# ---------------------------------------------------------------------------

@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def queue_pretrain_job(
    payload: PretrainJobRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Queue a from-scratch LLM pre-training job."""
    # Resolve and validate architecture
    if payload.arch_preset:
        if payload.arch_preset not in PRESETS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_response(
                    "INVALID_PRESET",
                    f"Unknown preset '{payload.arch_preset}'. "
                    f"Available: {list(PRESETS)}",
                ),
            )
        arch = ArchitectureConfig(**PRESETS[payload.arch_preset])
    elif payload.architecture:
        try:
            arch = ArchitectureConfig(**payload.architecture)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_response("INVALID_ARCHITECTURE", str(exc)),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_response(
                "MISSING_ARCHITECTURE",
                "Provide either arch_preset or a full architecture dict",
            ),
        )

    job_record = PretrainJob(
        owner_id=current_user.id,
        run_name=payload.run_name,
        status="queued",
        arch_preset=payload.arch_preset,
        architecture_snapshot=arch.to_dict(),
        tokenizer_job_id=payload.tokenizer_job_id or None,
        pretrain_config=payload.model_dump(),
        notes=payload.notes,
    )
    db.add(job_record)
    db.commit()
    db.refresh(job_record)

    async def _run_pretrain(bg_job):
        import asyncio
        from trainer.pretrain.engine import PretrainConfig, run_pretrain
        from app.config.settings import settings

        tokenizer_path = ""
        if payload.tokenizer_job_id:
            from app.database.session import SessionLocal
            with SessionLocal() as s:
                tok_rec = s.get(TokenizerJob, payload.tokenizer_job_id)
                if tok_rec and tok_rec.status == "completed":
                    tokenizer_path = tok_rec.output_dir

        pretrain_cfg = PretrainConfig(
            run_name=payload.run_name,
            architecture=arch.to_dict(),
            tokenizer_path=tokenizer_path,
            corpus_path=payload.corpus_path,
            max_seq_length=payload.max_seq_length,
            learning_rate=payload.learning_rate,
            batch_size=payload.batch_size,
            gradient_accumulation_steps=payload.gradient_accumulation_steps,
            num_train_epochs=payload.num_train_epochs,
            max_steps=payload.max_steps,
            warmup_steps=payload.warmup_steps,
            save_steps=payload.save_steps,
            fp16=payload.fp16,
            bf16=payload.bf16,
            seed=payload.seed,
            output_dir=str(settings.checkpoint_root / "pretrain"),
        )

        def _progress(fraction, metrics):
            bg_job.progress = fraction

        result = await asyncio.get_event_loop().run_in_executor(
            None, run_pretrain, pretrain_cfg, _progress
        )

        from app.database.session import SessionLocal
        from datetime import datetime, UTC
        with SessionLocal() as s:
            rec = s.get(PretrainJob, job_record.id)
            if rec:
                rec.status = "completed"
                rec.output_dir = result.get("final_model_path", "")
                rec.metrics = {
                    "total_params": result.get("total_params"),
                    "train_loss": result.get("train_loss"),
                    "train_runtime": result.get("train_runtime"),
                }
                rec.completed_at = datetime.now(UTC)
                s.commit()
        return result

    bg_job = get_job_queue().submit(_run_pretrain, job_type="pretrain")

    return success_response(
        "pretrain job queued",
        {
            "id": job_record.id,
            "bg_job_id": bg_job.id,
            "run_name": payload.run_name,
            "arch_preset": payload.arch_preset,
            "architecture": arch.to_dict(),
            "status": "queued",
        },
    )


@router.get("/jobs")
def list_pretrain_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    q = (
        db.query(PretrainJob)
        .filter(PretrainJob.owner_id == current_user.id)
        .order_by(PretrainJob.created_at.desc())
    )
    total = q.count()
    items = q.offset((page - 1) * page_size).limit(page_size).all()
    total_pages = max(1, (total + page_size - 1) // page_size)
    return success_response(
        "pretrain jobs fetched",
        {
            "items": [
                {
                    "id": j.id,
                    "run_name": j.run_name,
                    "status": j.status,
                    "arch_preset": j.arch_preset,
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
def get_pretrain_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    job = db.get(PretrainJob, job_id)
    if not job or job.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("NOT_FOUND", "Pretrain job not found"),
        )
    return success_response(
        "pretrain job fetched",
        {
            "id": job.id,
            "run_name": job.run_name,
            "status": job.status,
            "arch_preset": job.arch_preset,
            "architecture_snapshot": job.architecture_snapshot,
            "pretrain_config": job.pretrain_config,
            "output_dir": job.output_dir,
            "metrics": job.metrics,
            "error": job.error,
            "notes": job.notes,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        },
    )
