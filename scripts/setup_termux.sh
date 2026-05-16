#!/usr/bin/env bash
set -euo pipefail
pkg update -y
pkg install -y python nodejs clang cmake git

echo "Install backend deps: cd backend && pip install -r requirements.txt"
echo "Install frontend deps: cd frontend && npm install"
echo "Compile llama.cpp with scripts/build_llama.sh"
