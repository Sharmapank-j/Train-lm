# Train-LM

Train-LM is an offline-first, self-hosted platform scaffold for local dataset workflows, LoRA/QLoRA fine-tuning, experiment tracking, model registry, GGUF export, and local inference.

## Architecture

- `backend/`: FastAPI API layer, auth stubs, contracts, service boundaries
- `frontend/`: React + Vite UI shell with route structure
- `trainer/`: training engine package boundaries
- `inference/`: local inference backend boundaries (llama.cpp / Ollama)
- `telegram/`: bot integration boundaries
- `scripts/`: local/dev/Termux bootstrap scripts
- `docker/` and `docker-compose.yml`: container deployment scaffold
- `docs/`: architecture and operational docs

## Quick start (local)

```bash
# backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# frontend
cd ../frontend
npm install
npm run dev
```

## Docker

```bash
docker compose up --build
```

## Current foundation scope

This repository currently provides a production-oriented MVP with:

- standardized API success/error response contracts
- API routes for auth, datasets, training, exports, models, inference, health
- safe path handling helper for artifact storage protections
- frontend dashboard for auth, datasets, training jobs, model registry, chat
- Docker and script entrypoints for local and Termux workflows
- starter tests for contract and safety primitives
 
## MVP capabilities

- Dataset upload + validation (JSONL)
- LoRA / QLoRA training jobs with logs and metrics
- Model registry with adapter versions
- Local inference / chat endpoint for registered adapters

## Testing

```bash
python -m unittest discover -s backend/tests
```

## Roadmap (high level)

1. persistent auth and RBAC
2. dataset upload/validation/preprocessing workers
3. LoRA/QLoRA training orchestration and experiment tracking
4. GGUF export integration with llama.cpp
5. streaming chat playground + Telegram runtime

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
