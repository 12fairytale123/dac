# -*- coding: utf-8 -*-
"""后端 API 的请求 / 响应数据模型（Pydantic）。"""

from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class StudentSummary(BaseModel):
    id: str
    name: str
    cognitive_score: float           # 0-100，象限图 X 轴
    emotion_score: float             # 0-100，象限图 Y 轴
    engagement: float                # 0-1 参与度
    difficulty_type: str             # 正常/认知缺陷型/情绪阻塞型/双重风险型
    risk_level: str                  # 低/中/高
    weak_concepts: List[str] = []


class ForgettingItem(BaseModel):
    concept: str
    risk: float


class StudentDetail(StudentSummary):
    mastery: Dict[str, float] = {}           # 每知识点掌握度
    mastery_curve: List[List[float]] = []    # (T, K) 掌握曲线
    forgetting: List[ForgettingItem] = []
    evidence: Dict[str, float] = {}          # 各模态证据权重
    teacher_comment: str = ""


class ClassOverview(BaseModel):
    class_id: str
    class_name: str
    n_students: int
    concept_names: List[str]
    mastery_heatmap: List[List[float]]       # (n_students, n_concepts)
    emotion_timeline: List[Dict[str, float]] # 每时段 {t, positivity, engagement}
    type_distribution: Dict[str, int]        # 困境类型分布
    students: List[StudentSummary]
    class_positivity: float
    class_engagement: float


class TeacherContextIn(BaseModel):
    subject: str = "数学"
    grade: str = "高一"
    goal: str = "掌握本单元核心概念"
    progress: str = "第 3 章 / 共 8 章"
    class_note: str = ""


class StrategyRequest(BaseModel):
    student_id: str
    context: TeacherContextIn = Field(default_factory=TeacherContextIn)


class StrategyResponse(BaseModel):
    student_id: str
    student_name: str
    difficulty_type: str
    strategy: Dict[str, List[str]]           # {课前:[...], 课中:[...], 课后:[...]}
    engine: str                              # LLM / 模板


# ---- 原始模型 demo 接口 ---- #
class KTPredictRequest(BaseModel):
    concepts: List[int]
    responses: List[int]
    gaps: Optional[List[float]] = None


class KTPredictResponse(BaseModel):
    cognitive_score: float
    mastery: Dict[str, float]
    weak_concepts: List[str]
    forgetting: Dict[str, float] = {}
    source: str                              # IKT-model / heuristic
