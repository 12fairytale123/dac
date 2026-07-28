# -*- coding: utf-8 -*-
"""FastAPI 后端入口。

启动：
    uvicorn backend.main:app --reload --port 8000
接口文档：http://localhost:8000/docs
"""

from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .service import service
from .schemas import (ClassOverview, StudentDetail, StrategyRequest,
                      StrategyResponse, KTPredictRequest, KTPredictResponse)

app = FastAPI(title="智析学情·情知合一 · 教师决策辅助系统",
              description="认知—情感多模态学情诊断与教学调适 API", version="0.1.0")

# 允许前端(Vite 默认 5173)跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/status")
def status():
    """返回系统运行模式（真实模型 / 降级）。"""
    return service.status()


@app.get("/api/class/{class_id}", response_model=ClassOverview)
def class_overview(class_id: str):
    """班级总览：掌握热力图、情绪时间轴、困境象限、学生列表。"""
    return service.class_overview(class_id)


@app.get("/api/student/{student_id}", response_model=StudentDetail)
def student_detail(student_id: str):
    """个体详情：能力画像、掌握曲线、遗忘风险、多模态证据。"""
    s = service.student_detail(student_id)
    if s is None:
        raise HTTPException(404, f"学生 {student_id} 不存在")
    return s


@app.post("/api/strategy", response_model=StrategyResponse)
def strategy(req: StrategyRequest):
    """知识—情感联合决策：生成课前/课中/课后教学调适建议。"""
    r = service.generate_strategy(req.student_id, req.context.model_dump())
    if r is None:
        raise HTTPException(404, f"学生 {req.student_id} 不存在")
    return r


@app.post("/api/kt/predict", response_model=KTPredictResponse)
def kt_predict(req: KTPredictRequest):
    """认知诊断 demo：输入作答序列 → 掌握度画像与薄弱点。"""
    return service.kt_predict(req.concepts, req.responses, req.gaps)


@app.get("/")
def root():
    return {"service": "智析学情·情知合一", "docs": "/docs",
            "mode": service.status()["mode"]}
