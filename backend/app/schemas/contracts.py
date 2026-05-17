from datetime import datetime
from typing import Any

from pydantic import BaseModel


class APIBaseResponse(BaseModel):
    success: bool
    request_id: str
    timestamp: datetime


class UserResponse(APIBaseResponse):
    message: str
    data: dict[str, Any]
    metadata: dict[str, Any] = {}


class DatasetResponse(UserResponse):
    pass


class DatasetValidationResponse(UserResponse):
    pass


class DatasetVersionResponse(UserResponse):
    pass


class TrainingJobResponse(UserResponse):
    pass


class ExperimentResponse(UserResponse):
    pass


class ModelRegistryResponse(UserResponse):
    pass


class GGUFExportResponse(UserResponse):
    pass


class ChatCompletionResponse(UserResponse):
    pass


class StreamChunkResponse(BaseModel):
    event_type: str
    token: str
    session_id: str
    timestamp: datetime
    model_name: str
    completion_state: str


class WebSocketEventResponse(BaseModel):
    event_id: str
    event_type: str
    user_id: str | None = None
    job_id: str | None = None
    payload: dict[str, Any]
    severity: str
    timestamp: datetime


class QueueStatusResponse(UserResponse):
    pass


class HealthStatusResponse(UserResponse):
    pass


class TelegramBotResponse(UserResponse):
    pass


class SystemSettingsResponse(UserResponse):
    pass
