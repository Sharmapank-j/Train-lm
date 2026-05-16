from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "Train-LM"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    local_only_mode: bool = True
    secret_key: str = Field(default="change-me", min_length=8)
    data_root: Path = Path("datasets")
    model_root: Path = Path("models")
    export_root: Path = Path("exports")


settings = Settings()
