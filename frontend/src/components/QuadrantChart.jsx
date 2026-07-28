import React, { useState } from 'react'
import { TYPE_COLOR } from '../api.js'

// 认知×情感象限图：每个学生是一个点，坐标 = (认知得分, 情绪得分)。
// 四个象限恰好对应四类学习困境 —— 直接可视化「认知—情感协同决定学习表现」。
const W = 560, H = 440
const PAD = { l: 46, r: 18, t: 18, b: 40 }
const THR = 55                                    // 与后端 rule_based_difficulty 一致

const px = x => PAD.l + (x / 100) * (W - PAD.l - PAD.r)
const py = y => H - PAD.b - (y / 100) * (H - PAD.t - PAD.b)

const QUADRANTS = [
  { x: 78, y: 82, label: '正常',       color: 'var(--ok)' },
  { x: 22, y: 82, label: '认知缺陷型', color: 'var(--def)' },
  { x: 78, y: 22, label: '情绪阻塞型', color: 'var(--blk)' },
  { x: 22, y: 22, label: '双重风险型', color: 'var(--risk)' },
]

export default function QuadrantChart({ students, selectedId, onSelect }) {
  const [hover, setHover] = useState(null)

  return (
    <div className="quadrant-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
        {/* 象限底色（认知靛蓝 × 情绪玫红的极轻着色） */}
        <rect x={px(0)} y={py(100)} width={px(THR)-px(0)} height={py(0)-py(THR)} fill="var(--def)" opacity="0.04" />
        <rect x={px(THR)} y={py(100)} width={px(100)-px(THR)} height={py(0)-py(THR)} fill="var(--ok)" opacity="0.045" />
        <rect x={px(0)} y={py(THR)} width={px(THR)-px(0)} height={py(0)-py(THR)} fill="var(--risk)" opacity="0.05" />
        <rect x={px(THR)} y={py(THR)} width={px(100)-px(THR)} height={py(0)-py(THR)} fill="var(--blk)" opacity="0.04" />

        {/* 阈值分割线 */}
        <line x1={px(THR)} y1={py(0)} x2={px(THR)} y2={py(100)} stroke="var(--line)" strokeDasharray="4 4" />
        <line x1={px(0)} y1={py(THR)} x2={px(100)} y2={py(THR)} stroke="var(--line)" strokeDasharray="4 4" />

        {/* 坐标轴 */}
        <line x1={px(0)} y1={py(0)} x2={px(100)} y2={py(0)} stroke="var(--ink-2)" strokeWidth="1" />
        <line x1={px(0)} y1={py(0)} x2={px(0)} y2={py(100)} stroke="var(--ink-2)" strokeWidth="1" />
        <text x={px(50)} y={H-8} textAnchor="middle" className="quad-label" fill="var(--cog)">认知得分 →</text>
        <text x={13} y={py(50)} textAnchor="middle" className="quad-label" fill="var(--aff)"
              transform={`rotate(-90 13 ${py(50)})`}>情绪得分 →</text>

        {/* 象限名 */}
        {QUADRANTS.map(q => (
          <text key={q.label} x={px(q.x)} y={py(q.y)} textAnchor="middle"
                className="quad-label" fill={q.color} opacity="0.55">{q.label}</text>
        ))}

        {/* 学生点：颜色=困境类型，半径=风险等级 */}
        {students.map(s => {
          const r = s.risk_level === '高' ? 9 : s.risk_level === '中' ? 7 : 5.5
          const sel = s.id === selectedId
          return (
            <circle key={s.id} className="quad-dot"
              cx={px(s.cognitive_score)} cy={py(s.emotion_score)} r={sel ? r + 2 : r}
              fill={TYPE_COLOR[s.difficulty_type]}
              stroke={sel ? 'var(--ink)' : '#fff'} strokeWidth={sel ? 2 : 1.2}
              opacity={hover && hover.id !== s.id && !sel ? 0.5 : 0.9}
              onMouseEnter={() => setHover(s)} onMouseLeave={() => setHover(null)}
              onClick={() => onSelect(s.id)} />
          )
        })}
      </svg>

      {hover && (
        <div className="quad-tooltip" style={{
          left: `${(px(hover.cognitive_score) / W) * 100}%`,
          top: `${(py(hover.emotion_score) / H) * 100}%`,
        }}>
          {hover.name} · {hover.difficulty_type}<br />
          认知 <b>{hover.cognitive_score}</b> · 情绪 <b>{hover.emotion_score}</b>
        </div>
      )}
    </div>
  )
}
