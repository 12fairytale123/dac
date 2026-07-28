# -*- coding: utf-8 -*-
"""智析学情·情知合一 —— 模型层。

torch-free 的部分总是可导入；依赖 torch 的深度模型在未安装 torch 时
会被跳过（后端据此自动降级到启发式 / 模板模式）。
"""

# 无 torch 依赖（后端降级模式也需要）
from .constants import DIFFICULTY_TYPES, MODALITIES, rule_based_difficulty
from .llm_decision import (Diagnosis, TeacherContext, get_engine,
                           LLMStrategyEngine, TemplateStrategyEngine)

__all__ = [
    "DIFFICULTY_TYPES", "MODALITIES", "rule_based_difficulty",
    "Diagnosis", "TeacherContext", "get_engine",
    "LLMStrategyEngine", "TemplateStrategyEngine",
]

TORCH_AVAILABLE = False
try:  # 依赖 torch 的深度模型
    from .knowledge_tracing import DKT, IKT, masked_bce, encode_interactions
    from .emotion import (AudioEmotionEncoder, TextStateEncoder,
                          ImageStateEncoder, ModalityOutput)
    from .fusion import DifficultyClassifier, AttentionFusion
    TORCH_AVAILABLE = True
    __all__ += ["DKT", "IKT", "masked_bce", "encode_interactions",
                "AudioEmotionEncoder", "TextStateEncoder", "ImageStateEncoder",
                "ModalityOutput", "DifficultyClassifier", "AttentionFusion"]
except ImportError:  # 未装 torch → 后端进入降级模式
    pass
