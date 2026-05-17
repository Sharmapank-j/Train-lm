# Contributing to Train-LM

## Development flow

1. Fork and create a feature branch.
2. Keep changes focused and modular.
3. Add or update tests relevant to your changes.
4. Run the local test command:
   ```bash
   python -m unittest discover -s backend/tests
   ```
5. Open a pull request with clear notes and screenshots for UI updates.

## Standards

- Follow explicit module boundaries.
- Keep API contracts stable.
- Prefer local/offline-compatible dependencies.
- Never commit secrets.
