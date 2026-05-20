from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user
from app.core.contracts import error_response, success_response
from app.database.session import get_db
from app.models.orm import Model, ModelVersion, User

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
async def list_models(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    q = (
        db.query(Model)
        .filter((Model.owner_id == current_user.id) | (Model.is_public.is_(True)))
        .order_by(Model.created_at.desc())
    )
    total = q.count()
    items = q.offset((page - 1) * page_size).limit(page_size).all()
    total_pages = max(1, (total + page_size - 1) // page_size)
    return success_response(
        "models fetched",
        {
            "items": [
                {
                    "id": m.id,
                    "name": m.name,
                    "description": m.description,
                    "base_model": m.base_model,
                    "model_type": m.model_type,
                    "created_at": m.created_at.isoformat(),
                }
                for m in items
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


@router.get("/{model_id}")
def get_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    model = db.get(Model, model_id)
    if not model or (model.owner_id != current_user.id and not model.is_public):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_response("NOT_FOUND", "Model not found"))
    versions = (
        db.query(ModelVersion)
        .filter(ModelVersion.model_id == model.id)
        .order_by(ModelVersion.version.desc())
        .all()
    )
    return success_response(
        "model fetched",
        {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "base_model": model.base_model,
            "model_type": model.model_type,
            "created_at": model.created_at.isoformat(),
            "versions": [
                {
                    "id": v.id,
                    "version": v.version,
                    "adapter_path": v.adapter_path,
                    "merged_path": v.merged_path,
                    "metrics": v.metrics,
                    "created_at": v.created_at.isoformat(),
                }
                for v in versions
            ],
        },
    )


@router.get("/{model_id}/versions")
def list_model_versions(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    model = db.get(Model, model_id)
    if not model or (model.owner_id != current_user.id and not model.is_public):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_response("NOT_FOUND", "Model not found"))
    versions = (
        db.query(ModelVersion)
        .filter(ModelVersion.model_id == model.id)
        .order_by(ModelVersion.version.desc())
        .all()
    )
    return success_response(
        "model versions fetched",
        {
            "items": [
                {
                    "id": v.id,
                    "version": v.version,
                    "adapter_path": v.adapter_path,
                    "merged_path": v.merged_path,
                    "metrics": v.metrics,
                    "created_at": v.created_at.isoformat(),
                }
                for v in versions
            ]
        },
    )
