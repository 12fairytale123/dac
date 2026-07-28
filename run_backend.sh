#!/usr/bin/env bash
# 一键启动后端（降级模式无需 torch）
pip install -q fastapi "uvicorn[standard]" pydantic numpy
uvicorn backend.main:app --reload --port 8000
