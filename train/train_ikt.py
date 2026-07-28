# -*- coding: utf-8 -*-
"""训练可解释知识追踪(IKT)模型。

用法：
    python -m train.train_ikt            # 用合成数据训练并保存 checkpoints/ikt.pt
真实数据接入：把 data.synthetic.make_dataset 换成真实数据加载器即可，
保持字段名一致（concepts/responses/gaps/next_concepts/next_responses/mask）。
"""

from __future__ import annotations
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.knowledge_tracing import IKT, masked_bce            # noqa: E402
from data.synthetic import make_dataset                          # noqa: E402


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """无 sklearn 依赖的 AUC（按秩计算）。"""
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    n_pos, n_neg = y_true.sum(), (1 - y_true).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def train(epochs: int = 15, batch_size: int = 64, lr: float = 3e-3, val_ratio: float = 0.2):
    data = make_dataset()
    K = data["n_concepts"]
    tensors = [torch.tensor(data[k]) for k in
               ("concepts", "responses", "gaps", "next_concepts", "next_responses", "mask")]
    n = tensors[0].shape[0]
    n_val = int(n * val_ratio)
    ds = TensorDataset(*tensors)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n - n_val, n_val], generator=torch.Generator().manual_seed(0))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = IKT(K)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for c, r, g, nc, nr, m in train_dl:
            prob, mastery = model(c, r, nc, g)
            loss = masked_bce(prob, nr, m)
            # 掌握度平滑正则：相邻时刻掌握度不应剧烈跳变（提升可解释稳定性）
            smooth = (mastery[:, 1:] - mastery[:, :-1]).pow(2).mean()
            loss = loss + 1e-3 * smooth
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * c.shape[0]

        # 验证 AUC
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for c, r, g, nc, nr, m in val_dl:
                prob, _ = model(c, r, nc, g)
                ys.append(nr.flatten().numpy()); ps.append(prob.flatten().numpy())
        auc = auc_score(np.concatenate(ys), np.concatenate(ps))
        print(f"epoch {ep:02d}  loss={tot / (n - n_val):.4f}  val_AUC={auc:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_concepts": K,
                "concept_names": data["concept_names"]}, "checkpoints/ikt.pt")
    print("已保存 checkpoints/ikt.pt")


if __name__ == "__main__":
    train()
