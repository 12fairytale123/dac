# -*- coding: utf-8 -*-
"""多模态融合与学习困境识别层。

对应申报书 图 3-3「基于注意力机制的多模态融合」与「研究方法 3」。

把四路状态向量对齐、融合，并分类学习困境类型：
    0 正常
    1 认知缺陷型     （知识薄弱、情绪正常）
    2 情绪阻塞型     （知识尚可、情绪/动机受阻）
    3 双重风险型     （认知 + 情绪双低）

可解释性设计：
  - 融合用「模态注意力」——每个模态一个可读权重，说明本次判断主要看了哪路证据。
  - to_axes() 把融合结果映射到 (认知轴, 情绪轴) 二维分数 → 前端象限图的坐标，
    直接对应「认知×情感协同决定学习表现」的核心科学假设：四类困境 = 四个象限。
"""

from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import DIFFICULTY_TYPES, MODALITIES, rule_based_difficulty  # noqa: F401


class AttentionFusion(nn.Module):
    """模态注意力融合：4 路状态向量 → 统一表示 + 可读注意力权重。"""

    def __init__(self, dims: Dict[str, int], hidden: int = 128):
        super().__init__()
        # 把每路投到同一维度
        self.proj = nn.ModuleDict({k: nn.Linear(v, hidden) for k, v in dims.items()})
        self.query = nn.Parameter(torch.randn(hidden))
        self.hidden = hidden

    def forward(self, states: Dict[str, torch.Tensor]):
        keys = list(states.keys())
        tokens = torch.stack([F.relu(self.proj[k](states[k])) for k in keys], dim=1)  # (B,M,H)
        score = (tokens * self.query).sum(-1) / (self.hidden ** 0.5)                  # (B,M)
        attn = torch.softmax(score, dim=-1)                                           # (B,M)
        fused = (tokens * attn.unsqueeze(-1)).sum(1)                                  # (B,H)
        return fused, attn, keys


class DifficultyClassifier(nn.Module):
    """学习困境分类器 + 认知/情绪双轴回归。"""

    def __init__(self, dims: Dict[str, int], hidden: int = 128, n_types: int = 4):
        super().__init__()
        self.fusion = AttentionFusion(dims, hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3))
        self.type_head = nn.Linear(hidden, n_types)
        self.axis_head = nn.Linear(hidden, 2)               # (认知轴, 情绪轴) logits

    def forward(self, states: Dict[str, torch.Tensor]):
        fused, attn, keys = self.fusion(states)
        z = self.mlp(fused)
        return {
            "type_logits": self.type_head(z),
            "axes": torch.sigmoid(self.axis_head(z)),       # (B,2) ∈[0,1]
            "modality_attention": attn,                     # (B,M) 可读证据权重
            "modality_order": keys,
        }

    @torch.no_grad()
    def diagnose(self, states: Dict[str, torch.Tensor]) -> List[dict]:
        """批量诊断 → 每个学生一个可读结果（供后端）。"""
        self.eval()
        out = self.forward(states)
        prob = torch.softmax(out["type_logits"], dim=-1)
        axes = out["axes"]
        attn = out["modality_attention"]
        keys = out["modality_order"]
        results = []
        for i in range(prob.shape[0]):
            t = int(prob[i].argmax())
            results.append({
                "difficulty_type": DIFFICULTY_TYPES[t],
                "type_confidence": round(float(prob[i, t]), 3),
                "cognitive_score": round(float(axes[i, 0]) * 100, 1),   # 象限 X 轴
                "emotion_score": round(float(axes[i, 1]) * 100, 1),     # 象限 Y 轴
                "evidence_weights": {keys[m]: round(float(attn[i, m]), 3)
                                     for m in range(len(keys))},
            })
        return results


if __name__ == "__main__":
    torch.manual_seed(0)
    dims = {"cog": 32, "audio": 128, "text": 128, "image": 128}
    clf = DifficultyClassifier(dims)
    states = {k: torch.randn(3, v) for k, v in dims.items()}
    for r in clf.diagnose(states):
        print(r["difficulty_type"], r["cognitive_score"], r["emotion_score"], r["evidence_weights"])
