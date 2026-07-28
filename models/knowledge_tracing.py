# -*- coding: utf-8 -*-
"""认知诊断层：知识追踪模型。

对应申报书「研究内容二 / 科学问题二」：在保留深度序列建模能力的同时，
把「黑箱」的隐状态映射回可解释的「知识点—能力—遗忘」空间。

本文件提供两个模型：
  1) DKT   —— 深度知识追踪基线（LSTM，标准做法，用于性能对照）。
  2) IKT   —— 可解释知识追踪（本项目主模型）：
        LSTM 序列建模  +  每概念可读的「掌握度画像」M_t ∈ [0,1]^K
        +  CDM 风格透明预测（区分度 a_c、难度 b_c）
        +  每概念可学习的遗忘率 θ_c → 输出「遗忘风险」。

设计要点（可解释性从何而来）：
  - mastery_head 直接输出每个知识点的掌握概率，教师可读 → 「能力画像 / 能力短板」。
  - 预测下一题是否答对时，使用透明的 IRT/CDM 链接函数
        p = sigmoid( a_c * (M[c] - b_c) )
    即「掌握度越高于该点难度 → 越可能答对」，而不是不可解释的全连接层。
  - 遗忘风险  risk_c = 1 - exp(-θ_c * Δt)  随距上次练习时间增长，θ_c 可学习。

数据约定（单概念交互，DKT 家族标准做法，如 ASSISTments 的 skill_id）：
  - 每次交互 = (concept_id, response, time_gap)
  - response ∈ {0,1}；concept_id ∈ [0, K)
  - 交互被编码为整数  token = response * K + concept + 1，其中 0 保留给 padding。
  Q 矩阵（题目→多知识点）的扩展见 README。
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_interactions(concepts: torch.Tensor, responses: torch.Tensor, n_concepts: int) -> torch.Tensor:
    """把 (concept, response) 编码为 embedding 的整数 token；0 = padding。"""
    return responses.long() * n_concepts + concepts.long() + 1


# --------------------------------------------------------------------------- #
#  DKT 基线
# --------------------------------------------------------------------------- #
class DKT(nn.Module):
    """深度知识追踪（Piech et al., 2015）——性能对照基线。"""

    def __init__(self, n_concepts: int, hidden: int = 128, layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.n_concepts = n_concepts
        self.emb = nn.Embedding(2 * n_concepts + 1, hidden, padding_idx=0)
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_concepts)

    def forward(self, concepts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """返回每个时刻对「所有知识点」的答对概率 logits，形状 (B, T, K)。"""
        tokens = encode_interactions(concepts, responses, self.n_concepts)
        h, _ = self.lstm(self.emb(tokens))
        return self.out(self.drop(h))

    def next_prob(self, concepts, responses, next_concepts):
        """预测下一步在 next_concepts 上答对的概率，形状 (B, T)。"""
        logits = self.forward(concepts, responses)               # (B,T,K)
        gathered = logits.gather(-1, next_concepts.long().unsqueeze(-1)).squeeze(-1)
        return torch.sigmoid(gathered)


# --------------------------------------------------------------------------- #
#  IKT 可解释知识追踪（本项目主模型）
# --------------------------------------------------------------------------- #
class IKT(nn.Module):
    """Interpretable Knowledge Tracing.

    forward 同时返回：
      - next_prob  : 下一题答对概率（用于训练的 BCE 损失）
      - mastery    : 每一步、每个知识点的掌握度画像 (B, T, K) ∈ [0,1]
    """

    def __init__(self, n_concepts: int, emb_dim: int = 128, hidden: int = 128,
                 layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.n_concepts = n_concepts
        self.emb = nn.Embedding(2 * n_concepts + 1, emb_dim, padding_idx=0)
        #  +1 维输入通道给「距上次练习的时间间隔」，让模型显式感知遗忘
        self.lstm = nn.LSTM(emb_dim + 1, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.mastery_head = nn.Linear(hidden, n_concepts)        # → sigmoid = 掌握度画像

        #  CDM / IRT 透明预测参数（每个知识点一套，可解释）
        self.discrimination = nn.Parameter(torch.ones(n_concepts))    # a_c，经 softplus 保证>0
        self.difficulty = nn.Parameter(torch.zeros(n_concepts))       # b_c
        #  每概念遗忘率 θ_c（经 softplus 保证>0），用于遗忘风险计算
        self.forget_logit = nn.Parameter(torch.full((n_concepts,), -2.0))

    # ---- 组件 ---- #
    def mastery_of(self, concepts, responses, time_gap=None):
        """输出每一步的掌握度画像 (B, T, K)。"""
        tokens = encode_interactions(concepts, responses, self.n_concepts)
        emb = self.emb(tokens)                                    # (B,T,E)
        if time_gap is None:
            time_gap = torch.zeros(emb.shape[:2], device=emb.device)
        gap_feat = torch.log1p(time_gap.float()).unsqueeze(-1)    # 压缩长尾
        h, _ = self.lstm(torch.cat([emb, gap_feat], dim=-1))
        return torch.sigmoid(self.mastery_head(self.drop(h)))     # (B,T,K)

    def cdm_predict(self, mastery, target_concepts):
        """CDM 透明链接：p = sigmoid(a_c * (M[c] - b_c))。mastery:(B,T,K)。"""
        m = mastery.gather(-1, target_concepts.long().unsqueeze(-1)).squeeze(-1)  # (B,T)
        a = F.softplus(self.discrimination)[target_concepts.long()]
        b = self.difficulty[target_concepts.long()]
        return torch.sigmoid(a * (m - b))

    def forward(self, concepts, responses, next_concepts, time_gap=None):
        mastery = self.mastery_of(concepts, responses, time_gap)  # (B,T,K)
        prob = self.cdm_predict(mastery, next_concepts)           # (B,T)
        return prob, mastery

    def forgetting_risk(self, time_since_last: torch.Tensor) -> torch.Tensor:
        """遗忘风险 = 1 - exp(-θ_c * Δt)。time_since_last:(K,) → 返回 (K,)。"""
        theta = F.softplus(self.forget_logit)
        return 1.0 - torch.exp(-theta * time_since_last.clamp(min=0))

    # ---- 供后端调用的可解释推理 ---- #
    @torch.no_grad()
    def explain(self, concepts, responses, time_gap=None, time_since_last=None,
                concept_names=None, weak_threshold: float = 0.6):
        """对「单个学生」的一条作答序列做可解释诊断。

        参数均为 1D（长度 T）张量。返回 dict：
          mastery      : {概念: 当前掌握度}
          mastery_curve: (T, K) 掌握度随时间的曲线（画「掌握曲线」用）
          weak_concepts: 掌握度低于阈值的知识点，从弱到强排序
          forgetting   : {概念: 遗忘风险}（若提供 time_since_last）
          cognitive_score: 0-100 的综合认知得分（供象限图使用）
        """
        self.eval()
        c = concepts.long().unsqueeze(0)
        r = responses.long().unsqueeze(0)
        g = None if time_gap is None else time_gap.unsqueeze(0)
        curve = self.mastery_of(c, r, g).squeeze(0)               # (T,K)
        current = curve[-1]                                       # (K,)
        names = concept_names or [f"KC{i}" for i in range(self.n_concepts)]

        mastery = {names[i]: float(current[i]) for i in range(self.n_concepts)}
        order = torch.argsort(current)                            # 从弱到强
        weak = [names[i] for i in order.tolist() if current[i] < weak_threshold]

        result = {
            "mastery": mastery,
            "mastery_curve": curve.tolist(),
            "weak_concepts": weak,
            "cognitive_score": round(float(current.mean()) * 100, 1),
        }
        if time_since_last is not None:
            risk = self.forgetting_risk(time_since_last)
            result["forgetting"] = {names[i]: float(risk[i]) for i in range(self.n_concepts)}
        return result


# --------------------------------------------------------------------------- #
#  掩码 BCE 损失（忽略 padding）
# --------------------------------------------------------------------------- #
def masked_bce(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(1e-6, 1 - 1e-6)
    loss = F.binary_cross_entropy(pred, target.float(), reduction="none")
    return (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)


if __name__ == "__main__":
    # 冒烟测试：随机数据前向一遍
    torch.manual_seed(0)
    K, B, T = 10, 4, 20
    c = torch.randint(0, K, (B, T))
    r = torch.randint(0, 2, (B, T))
    nc = torch.randint(0, K, (B, T))
    gap = torch.rand(B, T) * 5

    ikt = IKT(K)
    prob, mastery = ikt(c, r, nc, gap)
    print("IKT next_prob", prob.shape, "mastery", mastery.shape)

    exp = ikt.explain(c[0], r[0], gap[0], time_since_last=torch.rand(K) * 10,
                      concept_names=[f"知识点{i}" for i in range(K)])
    print("认知得分:", exp["cognitive_score"], "| 薄弱点前三:", exp["weak_concepts"][:3])
