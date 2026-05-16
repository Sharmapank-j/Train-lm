#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Start backend: cd backend && uvicorn app.main:app --reload"
echo "Start frontend: cd frontend && npm run dev"
