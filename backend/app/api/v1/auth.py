from fastapi import APIRouter
from pydantic import BaseModel

from app.core.contracts import success_response

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
async def register(payload: LoginRequest) -> dict:
    return success_response("user registered", {"username": payload.username, "role": "user"})


@router.post("/login")
async def login(payload: LoginRequest) -> dict:
    return success_response(
        "login successful",
        {
            "access_token": "local-dev-token",
            "token_type": "bearer",
            "username": payload.username,
        },
    )
