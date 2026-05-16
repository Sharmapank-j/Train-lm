from __future__ import annotations

import re
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user
from app.config.settings import settings
from app.core.contracts import error_response, success_response
from app.database.session import get_db
from app.models.orm import Dataset, DatasetVersion, User
from app.preprocessing.validator import SUPPORTED_FORMATS, checksum, validate_jsonl
from app.utils.paths import safe_join

router = APIRouter(prefix="/datasets", tags=["datasets"])

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-\.]{1,200}$")


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^\w\-\.]", "_", Path(name).name)
    return safe[:200] or "upload"


def _paginate(query, page: int, page_size: int):
    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()
    total_pages = max(1, (total + page_size - 1) // page_size)
    return items, {
        "total_items": total,
        "current_page": page,
        "page_size": page_size,
        "next_page": page + 1 if page < total_pages else None,
        "previous_page": page - 1 if page > 1 else None,
        "total_pages": total_pages,
    }


@router.get("")
def list_datasets(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    q = (
        db.query(Dataset)
        .filter(Dataset.is_deleted.is_(False))
        .filter((Dataset.owner_id == current_user.id) | (Dataset.is_public.is_(True)))
        .order_by(Dataset.created_at.desc())
    )
    items, pagination = _paginate(q, page, page_size)
    return success_response(
        "datasets fetched",
        {
            "items": [
                {
                    "id": d.id,
                    "name": d.name,
                    "format": d.format,
                    "row_count": d.row_count,
                    "size_bytes": d.size_bytes,
                    "tags": d.tags,
                    "created_at": d.created_at.isoformat(),
                }
                for d in items
            ],
            "pagination": pagination,
        },
    )


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=error_response("FILE_TOO_LARGE", f"Max upload size is {settings.max_upload_bytes // (1024*1024)} MB"),
        )

    safe_name = _sanitize_filename(file.filename or "upload.jsonl")
    fmt = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else "jsonl"
    if fmt not in SUPPORTED_FORMATS:
        fmt = "jsonl"

    # Validate content
    report = validate_jsonl(content)
    if report.errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_response(
                "INVALID_DATASET",
                "Dataset validation failed",
                {"validation_report": report.to_dict()},
            ),
        )

    # Store raw file safely
    raw_dir = settings.data_root / "raw" / current_user.id
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = safe_join(raw_dir, safe_name)
    async with aiofiles.open(dest, "wb") as fh:
        await fh.write(content)

    ds = Dataset(
        owner_id=current_user.id,
        name=safe_name,
        format=fmt,
        storage_path=str(dest),
        row_count=report.valid,
        size_bytes=len(content),
    )
    db.add(ds)
    db.flush()

    ver = DatasetVersion(
        dataset_id=ds.id,
        version_number=1,
        storage_path=str(dest),
        checksum=checksum(content),
        row_count=report.valid,
        size_bytes=len(content),
        validation_report=report.to_dict(),
    )
    db.add(ver)
    db.commit()
    db.refresh(ds)

    return success_response(
        "dataset uploaded",
        {
            "id": ds.id,
            "name": ds.name,
            "format": ds.format,
            "row_count": ds.row_count,
            "size_bytes": ds.size_bytes,
            "validation": report.to_dict(),
        },
    )


@router.get("/{dataset_id}")
def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    ds = db.get(Dataset, dataset_id)
    if not ds or ds.is_deleted or (ds.owner_id != current_user.id and not ds.is_public):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_response("NOT_FOUND", "Dataset not found"))
    return success_response(
        "dataset fetched",
        {
            "id": ds.id,
            "name": ds.name,
            "format": ds.format,
            "description": ds.description,
            "tags": ds.tags,
            "row_count": ds.row_count,
            "size_bytes": ds.size_bytes,
            "created_at": ds.created_at.isoformat(),
        },
    )


@router.delete("/{dataset_id}", status_code=status.HTTP_200_OK)
def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    ds = db.get(Dataset, dataset_id)
    if not ds or ds.is_deleted or ds.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_response("NOT_FOUND", "Dataset not found"))
    ds.is_deleted = True
    db.commit()
    return success_response("dataset deleted", {"id": dataset_id})
