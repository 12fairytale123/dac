import React from 'react'
import { TYPE_COLOR } from '../api.js'

// —— 顶部指标条 ——
export function ClassStrip({ ov }) {
  const highRisk = ov.students.filter(s => s.risk_level === '高').length
  const needAttention = ov.students.filter(s => s.difficulty_type !== '正常').length
  const metrics = [
    { k: '学生总数', v: ov.n_students, unit: '人', pct: 1, color: 'var(--ink-2)' },
    { k: '班级情绪指数', v: ov.class_positivity, unit: '/100', pct: ov.class_positivity/100, color: 'var(--aff)' },
    { k: '平均参与度', v: Math.round(ov.class_engagement*100), unit: '%', pct: ov.class_engagement, color: 'var(--cog)' },
    { k: '需关注学生', v: needAttention, unit: '人', pct: needAttention/ov.n_students, color: 'var(--blk)' },
    { k: '高风险学生', v: highRisk, unit: '人', pct: highRisk/ov.n_students, color: 'var(--risk)', accent: true },
  ]
  return (
    <div className="strip">
      {metrics.map(m => (
        <div key={m.k} className={`metric${m.accent ? ' accent-risk' : ''}`}>
          <div className="k">{m.k}</div>
          <div className="v num">{m.v}<small> {m.unit}</small></div>
          <div className="bar"><i style={{ width: `${Math.min(100, m.pct*100)}%`, background: m.color }} /></div>
        </div>
      ))}
    </div>
  )
}

// —— 困境类型分布 ——
export function TypeDistribution({ dist, total }) {
  const order = ['正常', '认知缺陷型', '情绪阻塞型', '双重风险型']
  return (
    <div className="card">
      <div className="card-head"><span className="eyebrow">Cohort</span></div>
      <h2>学习困境分布</h2>
      <p className="desc">按「认知×情感」双轴归因得到的四类学生占比</p>
      <div className="dist">
        {order.map(t => {
          const c = dist[t] || 0
          return (
            <div className="item" key={t}>
              <div className="lbl"><i style={{ background: TYPE_COLOR[t] }} />{t}</div>
              <div className="track"><b style={{ width: `${total ? (c/total)*100 : 0}%`, background: TYPE_COLOR[t] }} /></div>
              <div className="cnt">{c}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// —— 情绪时间轴（课堂节奏） ——
export function EmotionTimeline({ timeline }) {
  const W = 320, H = 130, PAD = 22
  const xs = timeline.map((_, i) => PAD + (i/(timeline.length-1))*(W-2*PAD))
  const line = (key) => timeline.map((d, i) =>
    `${xs[i]},${H-PAD - d[key]*(H-2*PAD)}`).join(' ')
  return (
    <div className="card timeline">
      <div className="card-head"><span className="eyebrow">Live · 45min</span></div>
      <h2>课堂情绪时间轴</h2>
      <p className="desc">基于课堂音频的班级积极性与参与度起伏，用于捕捉注意力低谷</p>
      <div className="legend-inline">
        <span><i style={{ background: 'var(--aff)' }} />积极性</span>
        <span><i style={{ background: 'var(--cog)' }} />参与度</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%">
        {[0.25,0.5,0.75].map(g => (
          <line key={g} x1={PAD} y1={H-PAD-g*(H-2*PAD)} x2={W-PAD} y2={H-PAD-g*(H-2*PAD)}
                stroke="var(--line-2)" />
        ))}
        <polyline points={line('positivity')} fill="none" stroke="var(--aff)" strokeWidth="2.2" />
        <polyline points={line('engagement')} fill="none" stroke="var(--cog)" strokeWidth="2.2" strokeDasharray="5 4" />
        {timeline.map((d, i) => (
          <text key={i} x={xs[i]} y={H-6} textAnchor="middle" fontSize="9" fill="var(--muted)"
                fontFamily="var(--mono)">{d.t}′</text>
        ))}
      </svg>
    </div>
  )
}

// —— 掌握热力图（知识点 × 学生） ——
export function MasteryHeatmap({ ov, onSelect }) {
  const heatColor = v => {
    // 低→高：玫红(薄弱) → 中性 → 靛蓝(掌握)
    if (v < 0.5) { const t = v/0.5; return `rgba(214,69,107,${0.72-0.4*t})` }
    const t = (v-0.5)/0.5; return `rgba(79,93,214,${0.18+0.55*t})`
  }
  return (
    <div className="card">
      <div className="card-head"><span className="eyebrow">Knowledge Tracing · IKT</span></div>
      <h2>知识点掌握热力图</h2>
      <p className="desc">每格为该生在对应知识点的掌握度；越玫红越薄弱，越靛蓝越扎实。点名字看详情。</p>
      <div className="heatmap-scroll">
        <table className="heatmap">
          <thead>
            <tr>
              <th></th>
              {ov.concept_names.map(c => <th key={c} className="rot"><div>{c}</div></th>)}
            </tr>
          </thead>
          <tbody>
            {ov.students.map((s, i) => (
              <tr key={s.id}>
                <td className="name" style={{ cursor: 'pointer' }} onClick={() => onSelect(s.id)}>{s.name}</td>
                {ov.mastery_heatmap[i].map((v, j) => (
                  <td key={j} className="heat-cell" style={{ background: heatColor(v) }}
                      title={`${s.name} · ${ov.concept_names[j]}：${Math.round(v*100)}`} />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
