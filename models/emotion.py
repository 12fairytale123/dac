# -*- coding: utf-8 -*-
"""情感/状态计算层：音频、文本、图像三路非侵入式编码器。

对应申报书「研究内容一」的非认知维度与「研究方法 2」：
  - 音频：课堂/师生互动语音 → Wav2vec 声学表征 → 班级情绪极性、活跃度、师生契合度。
  - 文本：教师卷面评语、师生对话 → 中文 BERT → 学习态度、努力程度、风险信号。
  - 图像：卷面 / 笔迹 → CNN → 认真程度、紧张(书写压力)、疲惫。

工程约定（为了在没有大模型权重时也能跑）：
  - 每个编码器的「主干」（Wav2vec / BERT / ResNet）是可插拔的：
      * 有 torch + transformers/torchvision 时可加载真实预训练主干；
      * 没有时，编码器直接接收「已抽好的特征向量」，只训练轻量分类/回归头。
  - 这样后端 demo 无需下载数 GB 模型即可联调；正式实验再换真实主干。

每个编码器输出一个「状态向量」（供后续融合层使用）以及可读的标签分数。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  音频情感编码器
# --------------------------------------------------------------------------- #
class AudioEmotionEncoder(nn.Module):
    """课堂音频 → 班级情绪。

    输入两种之一：
      - raw 波形（需要 transformers 的 Wav2Vec2，见 from_waveform）
      - 预抽取的声学特征序列 feats: (B, T, F_in)（默认路径，随处可跑）

    输出：state (B, H) 状态向量；logits 三类情绪极性(消极/中性/积极)；
          engagement (B,) 参与度/活跃度 0-1。
    """

    def __init__(self, feat_dim: int = 768, hidden: int = 128, n_emotion: int = 3):
        super().__init__()
        self.bilstm = nn.LSTM(feat_dim, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden, 1)                # 时间维注意力聚合
        self.state = nn.Linear(2 * hidden, hidden)
        self.emotion_head = nn.Linear(hidden, n_emotion)    # 情绪极性
        self.engage_head = nn.Linear(hidden, 1)             # 参与度/活跃度

    def forward(self, feats: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h, _ = self.bilstm(feats)                           # (B,T,2H)
        score = self.attn(h).squeeze(-1)                    # (B,T)
        if mask is not None:
            score = score.masked_fill(~mask.bool(), float("-inf"))
        w = torch.softmax(score, dim=-1).unsqueeze(-1)      # (B,T,1)
        pooled = (h * w).sum(dim=1)                         # (B,2H)
        state = F.relu(self.state(pooled))                  # (B,H)
        return {
            "state": state,
            "emotion_logits": self.emotion_head(state),
            "engagement": torch.sigmoid(self.engage_head(state)).squeeze(-1),
        }

    @staticmethod
    def from_waveform(waveforms, sample_rate: int = 16000, model_name: str = "facebook/wav2vec2-base"):
        """用 Wav2Vec2 从原始波形抽特征（正式实验用；需 transformers）。"""
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor  # 延迟导入
        fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        backbone = Wav2Vec2Model.from_pretrained(model_name).eval()
        inputs = fe(waveforms, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = backbone(**inputs).last_hidden_state       # (B,T,768)
        return out


# --------------------------------------------------------------------------- #
#  文本状态编码器（教师评语 / 师生对话）
# --------------------------------------------------------------------------- #
class TextStateEncoder(nn.Module):
    """教师评语文本 → 学习态度 / 努力程度 / 风险信号。

    主路径接收句向量 emb: (B, D)（可由 BERT[CLS] 得到，见 embed_texts）。
    输出：state 向量 + 三个可读回归分数(attitude/effort/risk) ∈ [0,1]。
    """

    def __init__(self, emb_dim: int = 768, hidden: int = 128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.heads = nn.Linear(hidden, 3)                   # attitude / effort / risk

    def forward(self, emb: torch.Tensor):
        state = self.proj(emb)
        s = torch.sigmoid(self.heads(state))
        return {"state": state, "attitude": s[:, 0], "effort": s[:, 1], "risk": s[:, 2]}

    @staticmethod
    def embed_texts(texts: List[str], model_name: str = "bert-base-chinese"):
        """用中文 BERT 取 [CLS] 句向量（正式实验用；需 transformers）。"""
        from transformers import AutoTokenizer, AutoModel   # 延迟导入
        tok = AutoTokenizer.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name).eval()
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            return backbone(**enc).last_hidden_state[:, 0]   # (B,768)


# --------------------------------------------------------------------------- #
#  图像状态编码器（卷面 / 笔迹）
# --------------------------------------------------------------------------- #
class ImageStateEncoder(nn.Module):
    """卷面 / 笔迹图像 → 认真程度 / 书写压力 / 疲惫。

    默认用一个轻量 CNN（无需下载权重）；可通过 use_resnet=True 换 torchvision ResNet。
    输入 images: (B, 1, H, W) 灰度卷面图。
    """

    def __init__(self, hidden: int = 128, use_resnet: bool = False):
        super().__init__()
        self.use_resnet = use_resnet
        if use_resnet:
            import torchvision                              # 延迟导入
            net = torchvision.models.resnet18(weights=None)
            net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
            net.fc = nn.Linear(net.fc.in_features, hidden)
            self.backbone = net
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                nn.Flatten(), nn.Linear(64, hidden), nn.ReLU(),
            )
        self.heads = nn.Linear(hidden, 3)                   # neatness / pressure / fatigue

    def forward(self, images: torch.Tensor):
        state = self.backbone(images)
        s = torch.sigmoid(self.heads(state))
        return {"state": state, "neatness": s[:, 0], "pressure": s[:, 1], "fatigue": s[:, 2]}


@dataclass
class ModalityOutput:
    """把三路编码器的可读结果打包，方便后端序列化。"""
    audio: Dict[str, float] = field(default_factory=dict)
    text: Dict[str, float] = field(default_factory=dict)
    image: Dict[str, float] = field(default_factory=dict)


if __name__ == "__main__":
    torch.manual_seed(0)
    a = AudioEmotionEncoder()(torch.randn(2, 50, 768))
    t = TextStateEncoder()(torch.randn(2, 768))
    im = ImageStateEncoder()(torch.randn(2, 1, 64, 64))
    print("audio engagement:", a["engagement"].tolist())
    print("text risk:", t["risk"].tolist())
    print("image fatigue:", im["fatigue"].tolist())
