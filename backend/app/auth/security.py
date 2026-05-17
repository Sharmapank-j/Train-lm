"""Argon2 password hashing and JWT token utilities."""
from __future__ import annotations

from datetime import datetime, timedelta, UTC

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError
from jose import JWTError, jwt

from app.config.settings import settings

_ph = PasswordHasher()


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    """Return Argon2 hash for *plain* password."""
    return _ph.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches *hashed*. Never raises."""
    try:
        return _ph.verify(hashed, plain)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(hashed: str) -> bool:
    """True if the stored hash should be upgraded (param changes)."""
    return _ph.check_needs_rehash(hashed)


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------

_ALGORITHM = "HS256"


def create_access_token(subject: str, extra: dict | None = None, expires_minutes: int | None = None) -> str:
    expire = datetime.now(UTC) + timedelta(
        minutes=expires_minutes or settings.access_token_expire_minutes
    )
    payload: dict = {"sub": subject, "exp": expire, **(extra or {})}
    return jwt.encode(payload, settings.secret_key, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and return JWT claims. Raises JWTError on failure."""
    return jwt.decode(token, settings.secret_key, algorithms=[_ALGORITHM])


__all__ = [
    "hash_password",
    "verify_password",
    "needs_rehash",
    "create_access_token",
    "decode_access_token",
]
