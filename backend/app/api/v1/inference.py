from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user
from app.core.contracts import error_response, success_response
from app.database.session import get_db
from app.models.orm import Model, ModelVersion, User
from app.services.inference import get_inference_manager

router = APIRouter(prefix="/inference", tags=["inference"])


class ChatRequest(BaseModel):
    model_id: str | None = None
    model_name: str | None = None
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(128, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)

    @field_validator("model_name")
    @classmethod
    def require_model(cls, v, info):
        data = info.data
        if not v and not data.get("model_id"):
            raise ValueError("model_id or model_name is required")
        return v


@router.get("/models")
async def available_models(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    q = (
        db.query(ModelVersion, Model)
        .join(Model, ModelVersion.model_id == Model.id)
        .filter((Model.owner_id == current_user.id) | (Model.is_public.is_(True)))
        .order_by(ModelVersion.created_at.desc())
    )
    total = q.count()
    rows = q.offset((page - 1) * page_size).limit(page_size).all()
    total_pages = max(1, (total + page_size - 1) // page_size)
    items = [
        {
            "model_id": model.id,
            "model_name": model.name,
            "version_id": version.id,
            "version": version.version,
            "base_model": model.base_model,
            "adapter_path": version.adapter_path,
            "created_at": version.created_at.isoformat(),
        }
        for version, model in rows
    ]
    return success_response(
        "inference models fetched",
        {
            "items": items,
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


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    model: Model | None = None
    if payload.model_id:
        model = db.get(Model, payload.model_id)
    elif payload.model_name:
        model = db.query(Model).filter(Model.name == payload.model_name).first()

    if not model or (model.owner_id != current_user.id and not model.is_public):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_response("NOT_FOUND", "Model not found"))

    version = (
        db.query(ModelVersion)
        .filter(ModelVersion.model_id == model.id)
        .order_by(ModelVersion.version.desc())
        .first()
    )
    if not version:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_response("NOT_FOUND", "Model version not found"))

    manager = get_inference_manager()
    completion = manager.generate(
        cache_key=version.id,
        base_model=model.base_model,
        adapter_path=version.adapter_path or None,
        prompt=payload.prompt,
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
    )

    return success_response(
        "chat completion generated",
        {
            "model_name": model.name,
            "model_id": model.id,
            "completion": completion,
            "session_id": str(uuid4()),
        },
    )
