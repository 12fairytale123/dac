# -*- coding: utf-8 -*-
"""知识—情感联合决策层：基于大模型的教学调适策略生成。

对应申报书「研究内容三 / 科学问题三」的「最后一公里」与「研究方法 4」。

把结构化诊断结果（认知能力画像 + 情绪状态 + 困境类型）与教师输入的上下文
（教学目标、进度、班级特点）组合成 prompt，按「课前 / 课中 / 课后」三阶段
生成可执行的教学策略；并做安全约束过滤（不超出教师操作空间、贴合课程标准）。

两种运行模式：
  - LLMStrategyEngine：调用真实大模型（Anthropic / OpenAI，读环境变量 API key）。
  - 没 key / 调用失败：自动降级为 TemplateStrategyEngine，用规则模板产出可用建议，
    保证系统离线也能演示完整闭环。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os

STAGES = ["课前", "课中", "课后"]

#  安全 / 教学规范约束：这些约束会写进 system prompt，并对生成结果做后置过滤
CONSTRAINTS = [
    "建议必须落在教师日常可执行范围内（备课、讲授、练习、辅导、家校沟通），不得建议增设考试或惩罚。",
    "涉及学生心理时只做温和提醒与关怀，不下诊断、不贴负面标签，必要时建议转介学校心理老师。",
    "尊重教师专业判断，所有内容以「建议」呈现，不使用强制口吻。",
    "不得泄露或编造学生隐私信息，只基于给定诊断证据。",
]
_BLOCKLIST = ["体罚", "羞辱", "公开批评", "确诊", "精神病", "抑郁症患者"]


@dataclass
class Diagnosis:
    """单个学生 / 班级的结构化诊断输入。"""
    student: str
    difficulty_type: str
    weak_concepts: List[str] = field(default_factory=list)
    forgetting: Dict[str, float] = field(default_factory=dict)
    cognitive_score: float = 0.0
    emotion_score: float = 0.0
    evidence: Dict[str, float] = field(default_factory=dict)


@dataclass
class TeacherContext:
    subject: str = "数学"
    grade: str = "高一"
    goal: str = "掌握本单元核心概念"
    progress: str = "第 3 章 / 共 8 章"
    class_note: str = ""


def build_prompt(diag: Diagnosis, ctx: TeacherContext) -> str:
    """把诊断 + 上下文拼成结构化 prompt（供 LLM）。"""
    forget = ", ".join(f"{k}({v:.0%})" for k, v in sorted(
        diag.forgetting.items(), key=lambda x: -x[1])[:3]) or "无明显"
    return (
        f"【学科】{ctx.subject}  【年级】{ctx.grade}  【进度】{ctx.progress}\n"
        f"【教学目标】{ctx.goal}\n"
        f"【班级备注】{ctx.class_note or '无'}\n"
        f"【学生】{diag.student}\n"
        f"【困境类型】{diag.difficulty_type}\n"
        f"【认知得分】{diag.cognitive_score}/100  【情绪得分】{diag.emotion_score}/100\n"
        f"【薄弱知识点】{', '.join(diag.weak_concepts[:5]) or '无'}\n"
        f"【高遗忘风险点】{forget}\n"
        f"【证据权重】{json.dumps(diag.evidence, ensure_ascii=False)}\n\n"
        "请分「课前 / 课中 / 课后」三阶段给出针对该生的教学调适建议，"
        "每阶段 1-3 条，条目简短可执行；对情绪风险给出温和的关怀/沟通建议。"
        "只输出 JSON：{\"课前\":[...],\"课中\":[...],\"课后\":[...]}。"
    )


def _sanitize(strategy: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """后置安全过滤：剔除命中黑名单的条目。"""
    clean = {}
    for stage, items in strategy.items():
        kept = [s for s in items if not any(b in s for b in _BLOCKLIST)]
        clean[stage] = kept or ["（本阶段暂无附加建议，按原计划推进）"]
    return clean


# --------------------------------------------------------------------------- #
#  模板兜底引擎（离线可用）
# --------------------------------------------------------------------------- #
class TemplateStrategyEngine:
    """基于困境类型 + 薄弱点的规则模板，无需联网即可生成合理建议。"""

    def generate(self, diag: Diagnosis, ctx: TeacherContext) -> Dict[str, List[str]]:
        weak = "、".join(diag.weak_concepts[:2]) or "薄弱知识点"
        hot = max(diag.forgetting, key=diag.forgetting.get) if diag.forgetting else weak
        t = diag.difficulty_type
        s: Dict[str, List[str]] = {"课前": [], "课中": [], "课后": []}

        if t in ("认知缺陷型", "双重风险型"):
            s["课前"].append(f"备课时把「{weak}」拆成小步台阶，准备 1-2 道低起点例题作铺垫。")
            s["课中"].append(f"讲到「{weak}」时放慢节奏，用变式题即时检验，优先请该生尝试基础一档。")
            s["课后"].append(f"布置针对「{weak}」的分层练习，先巩固再拔高；安排 5 分钟个别答疑。")
        if t in ("情绪阻塞型", "双重风险型"):
            s["课前"].append("课前用一句轻松的私下问候降低其紧张感，营造安全的提问氛围。")
            s["课中"].append("多给正向反馈与小成功体验，避免公开点名施压；关注其参与度变化。")
            s["课后"].append("课后温和沟通了解畏难/情绪来源，必要时联系家长或转介心理老师协同关怀。")
        if t == "正常":
            s["课前"].append("状态平稳，可适当增加一道拓展题保持挑战性。")
            s["课中"].append("可请其分享思路，带动同伴讨论。")
            s["课后"].append(f"提醒复习易遗忘的「{hot}」，保持节奏即可。")
        else:
            s["课后"].append(f"一周后回访「{hot}」的掌握情况，验证干预是否见效。")
        return _sanitize(s)


# --------------------------------------------------------------------------- #
#  大模型引擎（在线，自动降级）
# --------------------------------------------------------------------------- #
class LLMStrategyEngine:
    """优先调用真实 LLM；失败则回退到模板引擎。"""

    def __init__(self, provider: str = "anthropic", model: Optional[str] = None):
        self.provider = provider
        self.model = model or ("claude-sonnet-4-6" if provider == "anthropic" else "gpt-4o-mini")
        self.fallback = TemplateStrategyEngine()

    def _system(self) -> str:
        return ("你是嵌入课堂教学系统的「教学设计助手」，服务一线中学教师，"
                "目标是减轻教师认知负担并提升教学精准度。必须遵守：\n- "
                + "\n- ".join(CONSTRAINTS))

    def generate(self, diag: Diagnosis, ctx: TeacherContext) -> Dict[str, List[str]]:
        prompt = build_prompt(diag, ctx)
        try:
            if self.provider == "anthropic":
                raw = self._call_anthropic(prompt)
            else:
                raw = self._call_openai(prompt)
            parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            return _sanitize({k: list(parsed.get(k, [])) for k in STAGES})
        except Exception as e:                              # 网络/解析/无 key → 兜底
            print(f"[LLM] 降级到模板引擎：{e}")
            return self.fallback.generate(diag, ctx)

    def _call_anthropic(self, prompt: str) -> str:
        import anthropic                                    # 延迟导入
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model=self.model, max_tokens=1024, system=self._system(),
            messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text

    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI                           # 延迟导入
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=self.model, max_tokens=1024,
            messages=[{"role": "system", "content": self._system()},
                      {"role": "user", "content": prompt}])
        return resp.choices[0].message.content


def get_engine() -> "LLMStrategyEngine | TemplateStrategyEngine":
    """有 API key 用 LLM，否则直接用模板引擎。"""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLMStrategyEngine("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        return LLMStrategyEngine("openai")
    return TemplateStrategyEngine()


if __name__ == "__main__":
    d = Diagnosis("张三", "双重风险型", ["一元二次方程", "因式分解"],
                  {"因式分解": 0.7, "配方法": 0.5}, 42.0, 38.0, {"认知(IKT)": 0.5, "音频": 0.3})
    strat = get_engine().generate(d, TeacherContext())
    print(json.dumps(strat, ensure_ascii=False, indent=2))
