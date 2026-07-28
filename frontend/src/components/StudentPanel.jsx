import React, { useEffect, useState } from 'react'
import { getStudent, getStrategy, TYPE_COLOR } from '../api.js'

const STAGES = ['课前', '课中', '课后']

export default function StudentPanel({ studentId, context, onClose }) {
  const [s, setS] = useState(null)
  const [strategy, setStrategy] = useState(null)
  const [loading, setLoading] = useState(false)
  const [stage, setStage] = useState('课前')

  useEffect(() => {
    setS(null); setStrategy(null); setStage('课前')
    if (studentId) getStudent(studentId).then(setS)
  }, [studentId])

  if (!studentId) return null

  const genStrategy = async () => {
    setLoading(true)
    const r = await getStrategy(studentId, context)
    setStrategy(r); setLoading(false)
  }

  const color = s ? TYPE_COLOR[s.difficulty_type] : 'var(--muted)'
  const masteryRows = s ? Object.entries(s.mastery).sort((a, b) => a[1] - b[1]) : []
  const masteryColor = v => v < 0.5 ? 'var(--risk)' : v < 0.7 ? 'var(--blk)' : 'var(--ok)'

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="panel" role="dialog" aria-label="学生诊断详情">
        {!s ? <div className="empty">加载中…</div> : (
          <>
            <div className="panel-head">
              <button className="close" onClick={onClose} aria-label="关闭">×</button>
              <div className="who">
                <div className="avatar" style={{ background: color }}>{s.name[0]}</div>
                <div>
                  <h3>{s.name}</h3>
                  <span className="type-tag" style={{ background: color }}>
                    {s.difficulty_type} · 风险{s.risk_level}
                  </span>
                </div>
              </div>
              <div className="axis-mini">
                <div className="a">
                  <div className="t"><span>认知得分</span><span className="val num" style={{ color: 'var(--cog)' }}>{s.cognitive_score}</span></div>
                  <div className="track"><b style={{ width: `${s.cognitive_score}%`, background: 'var(--cog)' }} /></div>
                </div>
                <div className="a">
                  <div className="t"><span>情绪得分</span><span className="val num" style={{ color: 'var(--aff)' }}>{s.emotion_score}</span></div>
                  <div className="track"><b style={{ width: `${s.emotion_score}%`, background: 'var(--aff)' }} /></div>
                </div>
              </div>
            </div>

            <div className="panel-body">
              {/* 教师经验判断（文本模态） */}
              <div className="section">
                <span className="eyebrow">教师评语 · 文本模态</span>
                <div className="comment">{s.teacher_comment}</div>
              </div>

              {/* 能力画像 */}
              <div className="section">
                <span className="eyebrow">能力画像 · IKT 掌握度</span>
                <div className="mastery-list">
                  {masteryRows.map(([k, v]) => (
                    <div className="mastery-row" key={k}>
                      <span className="nm" title={k}>{k}</span>
                      <span className="tr"><b style={{ width: `${v*100}%`, background: masteryColor(v) }} /></span>
                      <span className="pc">{Math.round(v*100)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* 遗忘风险 */}
              {s.forgetting?.length > 0 && (
                <div className="section">
                  <span className="eyebrow">遗忘风险 · 需及时复习</span>
                  <div className="chips">
                    {s.forgetting.map(f => (
                      <span key={f.concept} className="chip warn">{f.concept} {Math.round(f.risk*100)}%</span>
                    ))}
                  </div>
                </div>
              )}

              {/* 多模态证据权重 */}
              <div className="section">
                <span className="eyebrow">诊断证据 · 模态注意力</span>
                <div className="evidence">
                  {Object.entries(s.evidence).map(([k, v]) => (
                    <div className="ev" key={k}>
                      <span>{k}</span>
                      <span className="tr"><b style={{ width: `${v*100}%` }} /></span>
                      <span className="pc">{Math.round(v*100)}%</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* 联合决策：教学策略 */}
              <div className="section">
                <span className="eyebrow">知识—情感联合决策</span>
                {!strategy ? (
                  <button className="btn-primary" onClick={genStrategy} disabled={loading}>
                    {loading ? '正在生成教学建议…' : '生成课前 / 课中 / 课后教学建议'}
                  </button>
                ) : (
                  <>
                    <div className="stages">
                      {STAGES.map(st => (
                        <button key={st} className={stage === st ? 'on' : ''} onClick={() => setStage(st)}>{st}</button>
                      ))}
                    </div>
                    <div className="strat-list">
                      {(strategy.strategy[stage] || []).map((item, i) => (
                        <div className="strat-item" key={i}>
                          <span className="idx">{i + 1}</span><span>{item}</span>
                        </div>
                      ))}
                    </div>
                    <div className="engine-note">生成引擎：{strategy.engine} · 建议仅供参考，最终由教师决定</div>
                  </>
                )}
              </div>
            </div>
          </>
        )}
      </aside>
    </>
  )
}
