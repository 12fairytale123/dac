# -*- coding: utf-8 -*-
"""无第三方依赖的常量与规则（保证在未装 torch 时也能被后端导入）。"""

from __future__ import annotations

DIFFICULTY_TYPES = ["正常", "认知缺陷型", "情绪阻塞型", "双重风险型"]
MODALITIES = ["认知(IKT)", "音频", "文本", "图像"]


def rule_based_difficulty(cognitive: float, emotion: float, thr: float = 55.0) -> str:
    """启发式：按认知/情绪双轴（0-100）落入四象限。"""
    low_cog, low_emo = cognitive < thr, emotion < thr
    if low_cog and low_emo:
        return "双重风险型"
    if low_cog:
        return "认知缺陷型"
    if low_emo:
        return "情绪阻塞型"
    return "正常"
