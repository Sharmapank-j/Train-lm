"""Comprehensive pytest test suite for the Train-LM backend.

Tests cover:
- API contracts (response shape)
- Path traversal guard
- Password hashing & JWT
- Architecture config validation + presets
- Preprocessing validator
- Auth endpoints (register / login / me)
- Dataset endpoints (upload, list, get, delete)
- Training job creation
- Pre-train architecture & job endpoints
- Health endpoint
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make sure the backend package and trainer package are importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
for p in (str(BACKEND), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# In-memory SQLite with StaticPool so all connections share one DB
# (must happen before importing `app` so the env var is visible to settings)
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("TRAIN_LM_DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("TRAIN_LM_SECRET_KEY", "test-secret-key-12345")
os.environ.setdefault("TRAIN_LM_ALLOW_REMOTE_MODELS", "true")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.session import Base, get_db
# Import all ORM models so they are registered in Base.metadata before create_all
import app.models.orm  # noqa: F401
from app.main import app

_TEST_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,  # all connections share a single underlying connection
)

@event.listens_for(_TEST_ENGINE, "connect")
def _set_sqlite_pragma(conn, _record):
    conn.execute("PRAGMA foreign_keys=ON")

_TestSession = sessionmaker(autocommit=False, autoflush=False, bind=_TEST_ENGINE)

# Create all tables once on the shared in-memory connection
Base.metadata.create_all(bind=_TEST_ENGINE)


def _override_get_db():
    db = _TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db

client = TestClient(app, raise_server_exceptions=True)


# ===========================================================================
# Helpers
# ===========================================================================

_registered: set[str] = set()


def _register_and_login(username="alice", password="password123"):
    if username not in _registered:
        client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "email": f"{username}@example.com",
                "password": password,
            },
        )
        _registered.add(username)
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )
    return resp.json()["data"]["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ===========================================================================
# 1. API Contracts
# ===========================================================================

class TestContracts:
    def test_success_shape_has_required_fields(self):
        from app.core.contracts import success_response
        r = success_response("ok", {"x": 1})
        assert r["success"] is True
        assert "request_id" in r
        assert "timestamp" in r
        assert r["data"]["x"] == 1

    def test_error_shape_has_required_fields(self):
        from app.core.contracts import error_response
        r = error_response("CODE", "msg", {"detail": "x"})
        assert r["success"] is False
        assert r["error_code"] == "CODE"
        assert "request_id" in r
        assert "timestamp" in r

    def test_request_ids_are_unique(self):
        from app.core.contracts import success_response
        ids = {success_response("ok")["request_id"] for _ in range(10)}
        assert len(ids) == 10


# ===========================================================================
# 2. Path traversal guard
# ===========================================================================

class TestPathSafety:
    def test_safe_join_allows_valid_path(self):
        from app.utils.paths import safe_join
        root = Path("/tmp")
        result = safe_join(root, "subdir/file.txt")
        assert str(result).startswith("/tmp")

    def test_safe_join_blocks_traversal(self):
        from app.utils.paths import safe_join
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_join(Path("/tmp/safe"), "../etc/passwd")

    def test_safe_join_blocks_absolute_escape(self):
        from app.utils.paths import safe_join
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_join(Path("/tmp/safe"), "/etc/passwd")


# ===========================================================================
# 3. Auth: security primitives
# ===========================================================================

class TestSecurityPrimitives:
    def test_hash_and_verify_password(self):
        from app.auth.security import hash_password, verify_password
        h = hash_password("mysecret")
        assert verify_password("mysecret", h) is True
        assert verify_password("wrong", h) is False

    def test_jwt_roundtrip(self):
        from app.auth.security import create_access_token, decode_access_token
        token = create_access_token("user-id-123", extra={"role": "admin"})
        claims = decode_access_token(token)
        assert claims["sub"] == "user-id-123"
        assert claims["role"] == "admin"

    def test_expired_token_raises(self):
        from app.auth.security import create_access_token, decode_access_token
        from jose import JWTError
        token = create_access_token("user", expires_minutes=-1)
        with pytest.raises(JWTError):
            decode_access_token(token)


# ===========================================================================
# 4. Health endpoint
# ===========================================================================

class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["data"]["status"] == "healthy"


# ===========================================================================
# 5. Auth endpoints
# ===========================================================================

class TestAuthEndpoints:
    def test_register_creates_user(self):
        r = client.post(
            "/api/v1/auth/register",
            json={"username": "bob_reg", "email": "bob_reg@example.com", "password": "password123"},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["success"] is True
        assert body["data"]["username"] == "bob_reg"

    def test_register_duplicate_username_returns_409(self):
        client.post(
            "/api/v1/auth/register",
            json={"username": "dupuser", "email": "dup@example.com", "password": "password123"},
        )
        r = client.post(
            "/api/v1/auth/register",
            json={"username": "dupuser", "email": "dup2@example.com", "password": "password123"},
        )
        assert r.status_code == 409

    def test_register_weak_password_returns_422(self):
        r = client.post(
            "/api/v1/auth/register",
            json={"username": "weakpw", "email": "weak@example.com", "password": "short"},
        )
        assert r.status_code == 422

    def test_login_valid_credentials(self):
        client.post(
            "/api/v1/auth/register",
            json={"username": "loginuser", "email": "login@example.com", "password": "password123"},
        )
        r = client.post(
            "/api/v1/auth/login",
            json={"username": "loginuser", "password": "password123"},
        )
        assert r.status_code == 200
        body = r.json()
        assert "access_token" in body["data"]

    def test_login_wrong_password_returns_401(self):
        client.post(
            "/api/v1/auth/register",
            json={"username": "badpwuser", "email": "badpw@example.com", "password": "password123"},
        )
        r = client.post(
            "/api/v1/auth/login",
            json={"username": "badpwuser", "password": "wrongpassword"},
        )
        assert r.status_code == 401

    def test_me_endpoint_returns_user(self):
        token = _register_and_login("meuser")
        r = client.get("/api/v1/auth/me", headers=_auth(token))
        assert r.status_code == 200
        assert r.json()["data"]["username"] == "meuser"

    def test_me_without_token_returns_403_or_401(self):
        r = client.get("/api/v1/auth/me")
        assert r.status_code in (401, 403)


# ===========================================================================
# 6. Datasets
# ===========================================================================

# Valid JSONL with recognized fields (instruction/output and messages formats)
VALID_JSONL = b'\n'.join([
    json.dumps({"instruction": "What is 2+2?", "output": "4"}).encode(),
    json.dumps({"instruction": "Say hello", "output": "Hello!"}).encode(),
    json.dumps({
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }).encode(),
])

# Malformed JSONL: some valid, some not JSON at all
MIXED_JSONL = b'\n'.join([
    json.dumps({"instruction": "Good line 1", "output": "OK"}).encode(),
    b"this is not json at all",
    json.dumps({"instruction": "Good line 2", "output": "OK"}).encode(),
    b"{broken json",
])

# Completely invalid JSONL
INVALID_JSONL = b"not json\nalso not json\n"


class TestDatasetEndpoints:
    def setup_method(self):
        # Use a unique user per method via the method name
        self._uname = f"dstest_{id(self)}"
        self.token = _register_and_login(self._uname)

    def test_list_datasets_empty(self):
        r = client.get("/api/v1/datasets", headers=_auth(self.token))
        assert r.status_code == 200

    def test_upload_valid_jsonl(self):
        r = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("test.jsonl", io.BytesIO(VALID_JSONL), "application/jsonl")},
            headers=_auth(self.token),
        )
        assert r.status_code == 201
        body = r.json()
        assert body["success"] is True
        assert body["data"]["row_count"] > 0

    def test_upload_invalid_jsonl_returns_422(self):
        r = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("bad.jsonl", io.BytesIO(INVALID_JSONL), "application/jsonl")},
            headers=_auth(self.token),
        )
        assert r.status_code == 422

    def test_get_dataset_after_upload(self):
        upload = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("data.jsonl", io.BytesIO(VALID_JSONL), "application/jsonl")},
            headers=_auth(self.token),
        )
        ds_id = upload.json()["data"]["id"]
        r = client.get(f"/api/v1/datasets/{ds_id}", headers=_auth(self.token))
        assert r.status_code == 200
        assert r.json()["data"]["id"] == ds_id

    def test_delete_dataset(self):
        upload = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("del.jsonl", io.BytesIO(VALID_JSONL), "application/jsonl")},
            headers=_auth(self.token),
        )
        ds_id = upload.json()["data"]["id"]
        r = client.delete(f"/api/v1/datasets/{ds_id}", headers=_auth(self.token))
        assert r.status_code == 200
        r2 = client.get(f"/api/v1/datasets/{ds_id}", headers=_auth(self.token))
        assert r2.status_code == 404

    def test_upload_requires_auth(self):
        r = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("t.jsonl", io.BytesIO(VALID_JSONL), "application/jsonl")},
        )
        assert r.status_code in (401, 403)


# ===========================================================================
# 7. Preprocessing validator (unit)
# ===========================================================================

class TestPreprocessingValidator:
    def test_valid_jsonl_passes(self):
        from app.preprocessing.validator import validate_jsonl
        report = validate_jsonl(VALID_JSONL)
        assert report.valid > 0
        assert report.errors == []

    def test_empty_file_fails(self):
        from app.preprocessing.validator import validate_jsonl
        report = validate_jsonl(b"")
        assert len(report.errors) > 0

    def test_malformed_json_counted_as_invalid(self):
        from app.preprocessing.validator import validate_jsonl
        # MIXED_JSONL has 2 valid records and 2 malformed ones
        report = validate_jsonl(MIXED_JSONL)
        assert report.invalid >= 1
        assert report.valid >= 1

    def test_duplicates_detected(self):
        from app.preprocessing.validator import validate_jsonl
        line = json.dumps({"instruction": "x", "output": "y"}).encode()
        content = line + b"\n" + line + b"\n"
        report = validate_jsonl(content)
        assert report.duplicates >= 1

    def test_token_estimate_positive(self):
        from app.preprocessing.validator import validate_jsonl
        report = validate_jsonl(VALID_JSONL)
        assert report.estimated_tokens > 0


# ===========================================================================
# 8. Preprocessing pipeline (unit)
# ===========================================================================

class TestPreprocessingPipeline:
    def test_pipeline_removes_duplicates(self):
        from app.preprocessing.pipeline import PreprocessConfig, preprocess_jsonl
        line = json.dumps({"instruction": "Hi", "output": "Hello"}).encode()
        content = line + b"\n" + line + b"\n"
        _, result = preprocess_jsonl(content, PreprocessConfig(deduplicate=True))
        assert result.removed_duplicates >= 1

    def test_pipeline_strips_whitespace(self):
        from app.preprocessing.pipeline import PreprocessConfig, preprocess_jsonl
        content = json.dumps({"instruction": "  hello  ", "output": "  world  "}).encode()
        out_bytes, result = preprocess_jsonl(content, PreprocessConfig(strip_whitespace=True))
        record = json.loads(out_bytes.decode())
        assert record["instruction"] == "hello"
        assert record["output"] == "world"

    def test_pipeline_removes_too_long(self):
        from app.preprocessing.pipeline import PreprocessConfig, preprocess_jsonl
        long_text = "x " * 5000
        content = json.dumps({"text": long_text}).encode()
        _, result = preprocess_jsonl(content, PreprocessConfig(max_tokens=10))
        assert result.removed_too_long >= 1


# ===========================================================================
# 9. Training jobs
# ===========================================================================

class TestTrainingEndpoints:
    def setup_method(self):
        self.token = _register_and_login(f"trainer_{id(self)}")

    def test_queue_training_job(self):
        r = client.post(
            "/api/v1/training/jobs",
            json={
                "run_name": "test-run",
                "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "method": "lora",
            },
            headers=_auth(self.token),
        )
        assert r.status_code == 201
        body = r.json()
        assert body["data"]["state"] == "queued"

    def test_training_logs_and_metrics(self):
        r = client.post(
            "/api/v1/training/jobs",
            json={
                "run_name": "log-run",
                "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "method": "lora",
            },
            headers=_auth(self.token),
        )
        job_id = r.json()["data"]["id"]
        logs = client.get(f"/api/v1/training/jobs/{job_id}/logs", headers=_auth(self.token))
        assert logs.status_code == 200
        assert "lines" in logs.json()["data"]
        metrics = client.get(f"/api/v1/training/jobs/{job_id}/metrics", headers=_auth(self.token))
        assert metrics.status_code == 200
        assert "progress" in metrics.json()["data"]

    def test_list_training_jobs(self):
        r = client.get("/api/v1/training/jobs", headers=_auth(self.token))
        assert r.status_code == 200
        assert "items" in r.json()["data"]

    def test_get_training_job_not_found(self):
        r = client.get("/api/v1/training/jobs/nonexistent", headers=_auth(self.token))
        assert r.status_code == 404

    def test_training_requires_auth(self):
        r = client.get("/api/v1/training/jobs")
        assert r.status_code in (401, 403)


# ===========================================================================
# 9.5. Inference endpoints
# ===========================================================================

class TestInferenceEndpoints:
    def setup_method(self):
        self.token = _register_and_login(f"infer_{id(self)}")

    def test_list_inference_models(self):
        r = client.get("/api/v1/inference/models", headers=_auth(self.token))
        assert r.status_code == 200
        assert "items" in r.json()["data"]


# ===========================================================================
# 10. Architecture config (unit)
# ===========================================================================

class TestArchitectureConfig:
    def test_all_presets_are_valid(self):
        from trainer.architecture.config import PRESETS, ArchitectureConfig
        for name, cfg in PRESETS.items():
            arch = ArchitectureConfig(**cfg)
            assert arch.parameter_estimate > 0, f"Bad param estimate for {name}"

    def test_custom_arch_validates(self):
        from trainer.architecture.config import ArchitectureConfig
        arch = ArchitectureConfig(
            name="custom",
            arch_type="llama",
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_hidden_layers=4,
            intermediate_size=1024,
            vocab_size=8192,
            max_position_embeddings=512,
        )
        assert arch.head_dim == 64

    def test_misaligned_heads_raise(self):
        from trainer.architecture.config import ArchitectureConfig
        with pytest.raises(Exception):
            ArchitectureConfig(
                name="bad",
                arch_type="llama",
                hidden_size=512,
                num_attention_heads=7,  # 512 not divisible by 7
                num_key_value_heads=7,
                num_hidden_layers=2,
                intermediate_size=1024,
                vocab_size=8192,
                max_position_embeddings=512,
            )

    def test_kv_heads_exceed_attn_heads_raise(self):
        from trainer.architecture.config import ArchitectureConfig
        with pytest.raises(Exception):
            ArchitectureConfig(
                name="bad",
                arch_type="llama",
                hidden_size=512,
                num_attention_heads=4,
                num_key_value_heads=8,  # > num_attention_heads
                num_hidden_layers=2,
                intermediate_size=1024,
                vocab_size=8192,
                max_position_embeddings=512,
            )

    def test_parameter_estimate_increases_with_size(self):
        from trainer.architecture.config import PRESETS, ArchitectureConfig
        tiny = ArchitectureConfig(**PRESETS["TinyLM-15M"]).parameter_estimate
        large = ArchitectureConfig(**PRESETS["LargeLM-1B"]).parameter_estimate
        assert large > tiny


# ===========================================================================
# 11. Pre-train API endpoints
# ===========================================================================

class TestPretrainEndpoints:
    def setup_method(self):
        self.token = _register_and_login(f"ptuser_{id(self)}")

    def test_list_architectures(self):
        r = client.get("/api/v1/pretrain/architectures")
        assert r.status_code == 200
        items = r.json()["data"]["items"]
        assert len(items) >= 4

    def test_get_architecture_preset(self):
        r = client.get("/api/v1/pretrain/architectures/TinyLM-15M")
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["name"] == "TinyLM-15M"
        assert data["parameter_estimate"] > 0

    def test_get_architecture_not_found(self):
        r = client.get("/api/v1/pretrain/architectures/DoesNotExist")
        assert r.status_code == 404

    def test_validate_architecture_valid(self):
        r = client.post(
            "/api/v1/pretrain/architectures/validate",
            json={
                "name": "my-model",
                "arch_type": "llama",
                "hidden_size": 512,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "num_hidden_layers": 4,
                "intermediate_size": 1024,
                "vocab_size": 8192,
                "max_position_embeddings": 512,
            },
        )
        assert r.status_code == 200
        assert r.json()["data"]["head_dim"] == 64

    def test_validate_architecture_invalid_heads(self):
        r = client.post(
            "/api/v1/pretrain/architectures/validate",
            json={
                "name": "bad",
                "arch_type": "llama",
                "hidden_size": 512,
                "num_attention_heads": 7,
                "num_key_value_heads": 7,
                "num_hidden_layers": 2,
                "intermediate_size": 1024,
                "vocab_size": 8192,
                "max_position_embeddings": 512,
            },
        )
        assert r.status_code == 422

    def test_queue_pretrain_job_with_preset(self):
        r = client.post(
            "/api/v1/pretrain/jobs",
            json={
                "run_name": "tiny-test",
                "arch_preset": "TinyLM-15M",
                "corpus_path": "/tmp/corpus.txt",
            },
            headers=_auth(self.token),
        )
        assert r.status_code == 201
        body = r.json()
        assert body["data"]["status"] == "queued"
        assert body["data"]["arch_preset"] == "TinyLM-15M"

    def test_queue_pretrain_job_with_custom_arch(self):
        r = client.post(
            "/api/v1/pretrain/jobs",
            json={
                "run_name": "custom-test",
                "architecture": {
                    "name": "custom",
                    "arch_type": "llama",
                    "hidden_size": 256,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "num_hidden_layers": 2,
                    "intermediate_size": 512,
                    "vocab_size": 4096,
                    "max_position_embeddings": 256,
                },
                "corpus_path": "/tmp/corpus.txt",
            },
            headers=_auth(self.token),
        )
        assert r.status_code == 201
        body = r.json()
        assert body["data"]["status"] == "queued"

    def test_queue_pretrain_missing_arch_returns_422(self):
        r = client.post(
            "/api/v1/pretrain/jobs",
            json={"run_name": "no-arch", "corpus_path": "/tmp/corpus.txt"},
            headers=_auth(self.token),
        )
        assert r.status_code == 422

    def test_queue_pretrain_invalid_preset_returns_422(self):
        r = client.post(
            "/api/v1/pretrain/jobs",
            json={
                "run_name": "bad-preset",
                "arch_preset": "GigaLM-999T",
                "corpus_path": "/tmp/corpus.txt",
            },
            headers=_auth(self.token),
        )
        assert r.status_code == 422

    def test_queue_pretrain_path_traversal_blocked(self):
        r = client.post(
            "/api/v1/pretrain/jobs",
            json={
                "run_name": "traversal",
                "arch_preset": "TinyLM-15M",
                "corpus_path": "../../etc/passwd",
            },
            headers=_auth(self.token),
        )
        assert r.status_code == 422

    def test_list_pretrain_jobs(self):
        r = client.get("/api/v1/pretrain/jobs", headers=_auth(self.token))
        assert r.status_code == 200
        assert "items" in r.json()["data"]

    def test_get_pretrain_job(self):
        create_r = client.post(
            "/api/v1/pretrain/jobs",
            json={
                "run_name": "fetch-test",
                "arch_preset": "TinyLM-15M",
                "corpus_path": "/tmp/corpus.txt",
            },
            headers=_auth(self.token),
        )
        job_id = create_r.json()["data"]["id"]
        r = client.get(f"/api/v1/pretrain/jobs/{job_id}", headers=_auth(self.token))
        assert r.status_code == 200
        assert r.json()["data"]["id"] == job_id

    def test_get_pretrain_job_not_found(self):
        r = client.get("/api/v1/pretrain/jobs/nonexistent", headers=_auth(self.token))
        assert r.status_code == 404

    def test_pretrain_requires_auth(self):
        r = client.get("/api/v1/pretrain/jobs")
        assert r.status_code in (401, 403)
