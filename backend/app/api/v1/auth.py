from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user
from app.auth.security import create_access_token, hash_password, verify_password
from app.core.contracts import error_response, success_response
from app.database.session import get_db
from app.models.orm import User

router = APIRouter(prefix="/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

    @field_validator("username")
    @classmethod
    def username_clean(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3 or len(v) > 64:
            raise ValueError("username must be 3-64 characters")
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("username may only contain letters, digits, - and _")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> dict:
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=error_response("USERNAME_TAKEN", "Username already registered"),
        )
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=error_response("EMAIL_TAKEN", "Email already registered"),
        )
    # First ever user becomes admin
    is_first = db.query(User).count() == 0
    user = User(
        username=payload.username,
        email=str(payload.email),
        hashed_password=hash_password(payload.password),
        role="admin" if is_first else "user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return success_response(
        "user registered",
        {"id": user.id, "username": user.username, "role": user.role},
    )


@router.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> dict:
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_response("INVALID_CREDENTIALS", "Invalid username or password"),
        )
    if not user.is_active or user.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=error_response("ACCOUNT_INACTIVE", "Account is inactive"),
        )
    token = create_access_token(subject=user.id, extra={"role": user.role, "username": user.username})
    return success_response(
        "login successful",
        {
            "access_token": token,
            "token_type": "bearer",
            "username": user.username,
            "role": user.role,
        },
    )


@router.get("/me")
def me(current_user: User = Depends(get_current_user)) -> dict:
    return success_response(
        "current user",
        {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "role": current_user.role,
        },
    )
