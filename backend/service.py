# -*- coding: utf-8 -*-
"""服务编排层：把「知识追踪 → 情感 → 融合 → 决策」串成一条流水线。

自适应运行模式：
  - 若已装 torch 且存在 checkpoints/ikt.pt → 用真实 IKT 做认知诊断；
  - 否则进入「降级模式」，用 mock_data 的示例班级 + 模板策略引擎，
    保证前后端在任何机器上都能立即跑通、界面有数据。
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional

from models import TORCH_AVAILABLE, get_engine
from models.constants import DIFFICULTY_TYPES, rule_based_difficulty
from models.llm_decision import Diagnosis, TeacherContext
from . import mock_data

_CKPT = os.path.join("checkpoints", "ikt.pt")


class DiagnosisService:
    def __init__(self):
        self.strategy_engine = get_engine()
        self.engine_name = type(self.strategy_engine).__name__
        self.ikt = None
        self.concept_names = mock_data.CONCEPTS
        if TORCH_AVAILABLE and os.path.exists(_CKPT):
            self._load_ikt()
        self.mode = "IKT-model" if self.ikt is not None else "heuristic"

    def _load_ikt(self):
        import torch
        from models.knowledge_tracing import IKT
        ckpt = torch.load(_CKPT, map_location="cpu")
        self.ikt = IKT(ckpt["n_concepts"])
        self.ikt.load_state_dict(ckpt["state_dict"])
        self.ikt.eval()
        self.concept_names = ckpt.get("concept_names", self.concept_names)

    # ---- 班级总览 ---- #
    def class_overview(self, class_id: Optional[str] = None) -> dict:
        return mock_data.CLASS_OVERVIEW

    # ---- 个体详情 ---- #
    def student_detail(self, student_id: str) -> Optional[dict]:
        return mock_data.STUDENTS.get(student_id)

    # ---- 原始 KT 推理接口（认知诊断 demo）---- #
    def kt_predict(self, concepts: List[int], responses: List[int],
                   gaps: Optional[List[float]] = None) -> dict:
        if self.ikt is not None:
            import torch
            c = torch.tensor(concepts)
            r = torch.tensor(responses)
            g = torch.tensor(gaps) if gaps else None
            tsl = torch.zeros(self.ikt.n_concepts)
            exp = self.ikt.explain(c, r, g, time_since_last=tsl,
                                   concept_names=self.concept_names)
            return {
                "cognitive_score": exp["cognitive_score"],
                "mastery": exp["mastery"],
                "weak_concepts": exp["weak_concepts"],
                "forgetting": exp.get("forgetting", {}),
                "source": "IKT-model",
            }
        return self._kt_heuristic(concepts, responses)

    def _kt_heuristic(self, concepts: List[int], responses: List[int]) -> dict:
        """无模型时：按每个知识点的历史正确率估掌握度。"""
        K = len(self.concept_names)
        correct = [0.0] * K
        total = [0.0] * K
        for c, r in zip(concepts, responses):
            if 0 <= c < K:
                total[c] += 1
                correct[c] += r
        mastery = {}
        for i in range(K):
            m = (correct[i] + 1) / (total[i] + 2) if total[i] else 0.4   # 拉普拉斯平滑
            mastery[self.concept_names[i]] = round(m, 3)
        weak = [k for k, v in sorted(mastery.items(), key=lambda x: x[1]) if v < 0.6]
        score = round(sum(mastery.values()) / K * 100, 1)
        return {"cognitive_score": score, "mastery": mastery,
                "weak_concepts": weak, "forgetting": {}, "source": "heuristic"}

    # ---- 联合决策：生成教学策略 ---- #
    def generate_strategy(self, student_id: str, ctx: dict) -> Optional[dict]:
        s = mock_data.STUDENTS.get(student_id)
        if s is None:
            return None
        diag = Diagnosis(
            student=s["name"],
            difficulty_type=s["difficulty_type"],
            weak_concepts=s["weak_concepts"],
            forgetting={f["concept"]: f["risk"] for f in s["forgetting"]},
            cognitive_score=s["cognitive_score"],
            emotion_score=s["emotion_score"],
            evidence=s["evidence"],
        )
        context = TeacherContext(**ctx)
        strategy = self.strategy_engine.generate(diag, context)
        engine = "LLM" if "LLM" in self.engine_name else "模板"
        return {
            "student_id": student_id, "student_name": s["name"],
            "difficulty_type": s["difficulty_type"],
            "strategy": strategy, "engine": engine,
        }

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "torch_available": TORCH_AVAILABLE,
            "ikt_loaded": self.ikt is not None,
            "strategy_engine": self.engine_name,
            "difficulty_types": DIFFICULTY_TYPES,
        }


service = DiagnosisService()
