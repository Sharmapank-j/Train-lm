from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRAIN_LM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Train-LM"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    local_only_mode: bool = True

    # Security
    secret_key: str = Field(default="change-me-in-production", min_length=8)
    access_token_expire_minutes: int = 60
    hash_algorithm: str = "argon2"

    # Database
    database_url: str = "sqlite:///./trainlm.db"

    # Storage paths
    data_root: Path = Path("datasets")
    model_root: Path = Path("models")
    export_root: Path = Path("exports")
    checkpoint_root: Path = Path("checkpoints")
    log_root: Path = Path("logs")

    # Inference
    llama_cli_path: str = "./llama.cpp/build/bin/llama-cli"
    allow_remote_models: bool = False

    # Telegram
    telegram_token: str = ""

    # Limits
    max_upload_bytes: int = 512 * 1024 * 1024  # 512 MB
    max_concurrent_jobs: int = 2
    max_inference_sessions: int = 4

    @field_validator("data_root", "model_root", "export_root", "checkpoint_root", "log_root", mode="before")
    @classmethod
    def coerce_path(cls, v: object) -> Path:
        return Path(str(v))


settings = Settings()
