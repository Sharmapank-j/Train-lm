# Train-LM Architecture (Foundation)

Train-LM follows a modular offline-first architecture:

- API: FastAPI backend with typed contracts and stable response shape
- Training: isolated trainer package for LoRA/QLoRA workers with job queue integration
- Inference: pluggable local backends (llama.cpp primary, Ollama optional)
- UI: React dashboard for datasets, training, model registry, and chat
- Integration: Telegram bot module for local assistant access
- Deployment: native scripts + Docker Compose + Termux guidance

This foundation intentionally favors clear boundaries and contributor onboarding.
