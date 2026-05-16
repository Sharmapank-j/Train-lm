from fastapi import APIRouter
from pydantic import BaseModel

from app.core.contracts import success_response

router = APIRouter(prefix="/datasets", tags=["datasets"])


class DatasetCreateRequest(BaseModel):
    name: str
    format: str


@router.get("")
async def list_datasets() -> dict:
    return success_response("datasets fetched", {"items": [], "pagination": {"total_items": 0, "current_page": 1, "page_size": 20, "next_page": None, "previous_page": None, "total_pages": 0}})


@router.post("")
async def create_dataset(payload: DatasetCreateRequest) -> dict:
    return success_response("dataset registered", {"name": payload.name, "format": payload.format, "version": "v1"})
