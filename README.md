# 智析学情·情知合一 — 面向教师决策的 AI 课堂精准教学辅助系统

本仓库是申报书系统的**可运行原型（reference prototype）**：把「多源数据采集 →
认知与情感诊断 → 知识–情感联合决策 → 教学强化与情感支持闭环」这条主线，
用一套可读、可扩展、能立即跑通的代码实现出来。

> 定位说明：这是研究/申报用的**参考实现与工程骨架**，不是生产系统。深度模型
> 要达到申报书中的精度需接入真实课堂数据训练；开箱即用的是「降级/示例模式」，
> 用来演示完整的前后端闭环与交互。

---

## 一、代码结构与申报书的对应关系

| 目录 / 文件 | 申报书对应 | 说明 |
|---|---|---|
| `models/knowledge_tracing.py` | 研究内容二 · 可解释知识追踪 | `DKT` 基线 + `IKT`（LSTM 序列建模 + CDM 透明预测 + 可学习遗忘率），输出能力画像/薄弱点/遗忘风险 |
| `models/emotion.py` | 研究内容一 · 非侵入式情感计算 | 音频(Wav2vec)/文本(BERT)/图像(CNN) 三路编码器，主干可插拔 |
| `models/fusion.py` | 图 3-3 · 多模态注意力融合 | 模态注意力融合 + 学习困境四分类 + 认知/情绪双轴回归 |
| `models/llm_decision.py` | 研究内容三 · 联合决策「最后一公里」 | 课前/课中/课后策略生成 + 安全约束过滤 + 离线模板兜底 |
| `data/synthetic.py` | — | 带「学习+遗忘」机制的合成作答数据，供 IKT 训练与 demo |
| `train/train_ikt.py` | — | IKT 训练脚本（含无 sklearn 的 AUC 评估） |
| `backend/` | 系统整体框架（图 2-2） | FastAPI 服务，串起整条诊断流水线 |
| `frontend/` | 教师决策仪表板 | 象限图 / 掌握热力图 / 情绪时间轴 / 个体画像 / 策略生成 |

核心科学假设「认知—情感协同决定学习表现」在前端被直接可视化为**认知×情感象限图**：
横轴认知、纵轴情绪，四个象限 = 四类学习困境（正常 / 认知缺陷型 / 情绪阻塞型 / 双重风险型）。

---

## 二、快速开始

### 1. 后端（降级模式，无需 torch，30 秒跑通）
```bash
pip install fastapi "uvicorn[standard]" pydantic numpy
uvicorn backend.main:app --reload --port 8000
# 打开 http://localhost:8000/docs 查看接口
```

### 2. 前端
```bash
cd frontend
npm install
npm run dev
# 打开 http://localhost:5173
```
> 前端内置示例数据：即使后端没启动也能独立预览界面；后端启动后会自动改用真实接口。

### 3. 训练真实 IKT（可选）
```bash
pip install torch
python -m train.train_ikt          # 生成 checkpoints/ikt.pt
# 之后重启后端，/api/status 的 mode 会变为 "IKT-model"
```

### 4. 接入真实大模型做策略生成（可选）
```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...    # 或 OPENAI_API_KEY
# 后端会自动从模板引擎切换到 LLM 引擎
```

---

## 三、主要 API

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/status` | 查看运行模式（真实模型 / 降级） |
| GET | `/api/class/{id}` | 班级总览：热力图、情绪轴、困境分布、学生列表 |
| GET | `/api/student/{id}` | 个体画像：掌握度、掌握曲线、遗忘风险、多模态证据 |
| POST | `/api/kt/predict` | 认知诊断 demo：作答序列 → 掌握度画像 |
| POST | `/api/strategy` | 联合决策：生成课前/课中/课后教学建议 |

---

## 四、把原型升级为真实系统

1. **数据接入**：用真实作答日志替换 `data/synthetic.py`；补齐音频/图像/文本的采集与预处理
   （VAD、降噪、分段 / 去噪、裁剪、归一化 / 分句、脱敏），保持数据字段结构不变。
2. **认知侧**：`IKT` 已支持单知识点交互；扩展 Q 矩阵（题目→多知识点）时，把 `cdm_predict`
   改为对题目所需知识点做补偿式/合取式聚合即可。
3. **情感侧**：调用 `AudioEmotionEncoder.from_waveform` / `TextStateEncoder.embed_texts`
   接入真实 Wav2vec / 中文 BERT 主干；`ImageStateEncoder(use_resnet=True)` 接 ResNet。
4. **融合与困境分类**：用教师标注的困难学生名单做监督，训练 `DifficultyClassifier`；
   其 `modality_attention` 即为可解释的「证据权重」。
5. **决策安全**：`llm_decision.CONSTRAINTS` 与 `_BLOCKLIST` 是安全约束入口，
   按校本要求扩充；教师的「有用/无用、易执行/难执行」反馈回流到提示工程与策略库。

---

## 五、技术要点小结

- **可解释性**：IKT 的 `mastery_head` 直接输出每个知识点的掌握概率，预测走透明的
  IRT/CDM 链接 `p = σ(a·(mastery − b))`，避免全连接「黑箱」；融合层用模态注意力
  给出每次判断的证据权重。
- **非侵入 + 低成本**：情感只用音频/文本/卷面图像，规避视觉隐私与高成本采集。
- **可降级**：未装 torch / 无模型权重 / 无 API key 时，系统自动退到启发式与模板引擎，
  保证任何环境都能演示完整闭环。

四川大学 · 计算机学院 · 大学生创新训练计划
