# -*- coding: utf-8 -*-
"""示例班级数据。

作用：在没有真实课堂数据 / 未训练模型时，让整套前后端立即跑通、界面有真实感。
接入真实数据后，由 service 层用真实模型输出替换本模块。
数据用固定随机种子生成，保证每次启动一致、可复现。
"""

from __future__ import annotations
import numpy as np

from data.synthetic import CONCEPTS, simulate_student
from models.constants import rule_based_difficulty

CLASS_ID = "G1-3"
CLASS_NAME = "高一(3)班"
N_CONCEPTS = len(CONCEPTS)

_NAMES = ["王梓涵", "李思远", "张晓彤", "刘子墨", "陈欣怡", "杨浩然", "赵梦琪", "黄俊杰",
          "周雨桐", "吴天佑", "徐若曦", "孙嘉成", "马艺涵", "朱博文", "胡歆然", "郭子轩",
          "何欣悦", "高梓睿", "林诗涵", "罗宇航", "郑雅雯", "谢明轩", "唐бела", "邓昕妍"]
_NAMES = [n for n in _NAMES if n.isprintable()][:24]

_COMMENTS = {
    "正常": "思路清晰，作业工整，本单元掌握扎实。",
    "认知缺陷型": "基础概念还有漏洞，计算步骤容易跳步出错，需要补一补。",
    "情绪阻塞型": "其实听得懂，但课上不太敢举手，遇到难题容易先放弃。",
    "双重风险型": "近期状态低落，作业敷衍，卷面涂改多，知识点也跟不上。",
}


def _risk_level(dtype: str) -> str:
    return {"正常": "低", "认知缺陷型": "中", "情绪阻塞型": "中", "双重风险型": "高"}[dtype]


def _build():
    rng = np.random.default_rng(2025)
    students = []
    heatmap = []
    for i, name in enumerate(_NAMES):
        # 用模拟器得到掌握向量（保留知识点间的相对强弱），
        # 再平移到目标认知水平，制造覆盖四象限的真实分布。
        _, _, _, mastery = simulate_student(N_CONCEPTS, seq_len=45, rng=rng)
        target = float(np.clip(rng.normal(0.6, 0.16), 0.15, 0.95))
        mastery = np.clip(mastery + (target - mastery.mean()), 0.02, 0.99)
        cognitive = float(mastery.mean()) * 100
        # 情绪得分：独立采样，与认知轴共同决定困境象限
        emotion = float(np.clip(rng.normal(62, 20), 5, 98))
        engagement = float(np.clip(rng.normal(0.65, 0.18), 0.05, 0.99))
        dtype = rule_based_difficulty(cognitive, emotion)

        order = np.argsort(mastery)
        weak = [CONCEPTS[j] for j in order[:3] if mastery[j] < 0.6]
        forgetting = {CONCEPTS[j]: float(np.clip(rng.uniform(0.2, 0.9) *
                      (1 - mastery[j]), 0, 1)) for j in order[:4]}
        # 证据权重（模态注意力）：按困境类型给出合理的主导模态
        ev = _evidence_for(dtype, rng)

        # 掌握曲线：造一条从低到当前的平滑上升曲线（供个体页画图）
        curve = _fake_curve(mastery, rng)

        students.append({
            "id": f"S{i+1:02d}", "name": name,
            "cognitive_score": round(cognitive, 1),
            "emotion_score": round(emotion, 1),
            "engagement": round(engagement, 2),
            "difficulty_type": dtype, "risk_level": _risk_level(dtype),
            "weak_concepts": weak,
            "mastery": {CONCEPTS[j]: round(float(mastery[j]), 3) for j in range(N_CONCEPTS)},
            "mastery_curve": curve,
            "forgetting": [{"concept": k, "risk": round(v, 3)}
                           for k, v in sorted(forgetting.items(), key=lambda x: -x[1])],
            "evidence": ev,
            "teacher_comment": _COMMENTS[dtype],
        })
        heatmap.append([round(float(mastery[j]), 3) for j in range(N_CONCEPTS)])

    timeline = _class_timeline(students, rng)
    dist = {}
    for s in students:
        dist[s["difficulty_type"]] = dist.get(s["difficulty_type"], 0) + 1

    overview = {
        "class_id": CLASS_ID, "class_name": CLASS_NAME, "n_students": len(students),
        "concept_names": CONCEPTS, "mastery_heatmap": heatmap,
        "emotion_timeline": timeline, "type_distribution": dist,
        "class_positivity": round(float(np.mean([s["emotion_score"] for s in students])), 1),
        "class_engagement": round(float(np.mean([s["engagement"] for s in students])), 2),
        "students": [{k: s[k] for k in ("id", "name", "cognitive_score", "emotion_score",
                      "engagement", "difficulty_type", "risk_level", "weak_concepts")}
                     for s in students],
    }
    return overview, {s["id"]: s for s in students}


def _evidence_for(dtype: str, rng) -> dict:
    base = {"认知(IKT)": 0.25, "音频": 0.25, "文本": 0.25, "图像": 0.25}
    if dtype in ("认知缺陷型", "双重风险型"):
        base["认知(IKT)"] += 0.25
    if dtype in ("情绪阻塞型", "双重风险型"):
        base["音频"] += 0.15; base["文本"] += 0.1
    noise = {k: max(0.02, v + rng.uniform(-0.05, 0.05)) for k, v in base.items()}
    tot = sum(noise.values())
    return {k: round(v / tot, 3) for k, v in noise.items()}


def _fake_curve(final_mastery, rng, steps: int = 12):
    curve = []
    start = np.clip(final_mastery - rng.uniform(0.15, 0.4, len(final_mastery)), 0.05, 1)
    for t in range(steps):
        frac = t / (steps - 1)
        row = np.clip(start + (final_mastery - start) * frac +
                      rng.normal(0, 0.02, len(final_mastery)), 0.02, 0.99)
        curve.append([round(float(x), 3) for x in row])
    return curve


def _class_timeline(students, rng, segments: int = 8):
    base_pos = np.mean([s["emotion_score"] for s in students]) / 100
    base_eng = np.mean([s["engagement"] for s in students])
    tl = []
    for t in range(segments):
        wave = 0.12 * np.sin(t / 1.5)                     # 课堂节奏起伏
        tl.append({
            "t": t * 5,                                   # 第 t*5 分钟
            "positivity": round(float(np.clip(base_pos + wave + rng.normal(0, 0.03), 0, 1)), 3),
            "engagement": round(float(np.clip(base_eng + wave + rng.normal(0, 0.03), 0, 1)), 3),
        })
    return tl


CLASS_OVERVIEW, STUDENTS = _build()


if __name__ == "__main__":
    print("班级：", CLASS_NAME, "| 人数：", CLASS_OVERVIEW["n_students"])
    print("困境分布：", CLASS_OVERVIEW["type_distribution"])
    print("示例学生：", STUDENTS["S01"]["name"], STUDENTS["S01"]["difficulty_type"],
          STUDENTS["S01"]["weak_concepts"])
