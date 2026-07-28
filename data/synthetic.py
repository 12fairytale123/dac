# -*- coding: utf-8 -*-
"""合成作答数据生成器。

在拿到真实课堂数据前，用带「学习 + 遗忘」机制的模拟器生成作答序列，
让 IKT 有可学习的信号，并驱动 demo。真实数据接入后替换此模块即可
（保持返回 (concepts, responses, time_gaps, next_concepts, mask) 结构一致）。
"""

from __future__ import annotations
import numpy as np

# 示例知识点（可换成真实知识图谱节点）
CONCEPTS = [
    "有理数运算", "整式加减", "一元一次方程", "因式分解", "一元二次方程",
    "二次函数", "不等式", "相似三角形", "勾股定理", "概率初步",
]


def simulate_student(n_concepts: int, seq_len: int, rng: np.random.Generator):
    """模拟一个学生的一条作答序列。返回 numpy 数组。"""
    true_diff = rng.uniform(-0.5, 0.5, n_concepts)          # 各知识点难度 b_c
    mastery = rng.uniform(0.1, 0.4, n_concepts)             # 初始掌握度
    learn_rate = rng.uniform(0.05, 0.2, n_concepts)         # 练习提升速度
    forget = rng.uniform(0.005, 0.03, n_concepts)          # 遗忘速度
    last_seen = np.zeros(n_concepts)

    concepts, responses, gaps = [], [], []
    t = 0
    for _ in range(seq_len):
        c = rng.integers(0, n_concepts)
        gap = float(rng.integers(0, 5))                    # 距上次交互的相对时间
        t += gap + 1
        # 遗忘：所有概念随时间衰减
        mastery = np.clip(mastery - forget * (t - last_seen) * 0.02, 0.02, 0.98)
        # 作答：掌握度越高越可能答对
        p = 1 / (1 + np.exp(-4 * (mastery[c] - (true_diff[c] + 0.5))))
        r = int(rng.random() < p)
        # 学习：练习该点后掌握度提升（答对提升更多）
        mastery[c] = np.clip(mastery[c] + learn_rate[c] * (0.6 + 0.4 * r), 0.02, 0.99)
        last_seen[c] = t
        concepts.append(c); responses.append(r); gaps.append(gap)

    return (np.array(concepts), np.array(responses),
            np.array(gaps, dtype=np.float32), mastery)


def make_dataset(n_students: int = 800, n_concepts: int = len(CONCEPTS),
                 seq_len: int = 40, seed: int = 42):
    """生成一批学生序列，返回 padding 后的张量友好数组。"""
    rng = np.random.default_rng(seed)
    C, R, G = [], [], []
    for _ in range(n_students):
        c, r, g, _ = simulate_student(n_concepts, seq_len, rng)
        C.append(c); R.append(r); G.append(g)
    C, R, G = np.stack(C), np.stack(R), np.stack(G)
    # 训练目标：预测「下一题」→ 用输入序列 [:-1] 预测 [1:]
    concepts_in, responses_in, gaps_in = C[:, :-1], R[:, :-1], G[:, :-1]
    next_concepts, next_responses = C[:, 1:], R[:, 1:]
    mask = np.ones_like(next_responses)
    return {
        "concepts": concepts_in, "responses": responses_in, "gaps": gaps_in,
        "next_concepts": next_concepts, "next_responses": next_responses,
        "mask": mask, "n_concepts": n_concepts, "concept_names": CONCEPTS[:n_concepts],
    }


if __name__ == "__main__":
    d = make_dataset(n_students=5, seq_len=12)
    print("样本形状：", d["concepts"].shape, "知识点数：", d["n_concepts"])
    print("正确率：", round(float(d["next_responses"].mean()), 3))
