import React, { useEffect, useState } from 'react'
import { getOverview, getStatus, TYPE_COLOR } from './api.js'
import QuadrantChart from './components/QuadrantChart.jsx'
import { ClassStrip, TypeDistribution, EmotionTimeline, MasteryHeatmap } from './components/Panels.jsx'
import StudentPanel from './components/StudentPanel.jsx'

// 教学上下文（可由教师在真实产品中编辑；此处给默认值）
const CONTEXT = { subject: '数学', grade: '高一', goal: '掌握一元二次方程与因式分解',
  progress: '第 3 章 / 共 8 章', class_note: '' }

const TYPES = ['正常', '认知缺陷型', '情绪阻塞型', '双重风险型']

export default function App() {
  const [ov, setOv] = useState(null)
  const [status, setStatus] = useState(null)
  const [selected, setSelected] = useState(null)

  useEffect(() => {
    getOverview().then(setOv)
    getStatus().then(setStatus)
  }, [])

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="logo">智析学情<span className="dot">·</span></div>
          <div className="sub">情知合一 · 教师决策辅助系统</div>
        </div>

        {ov && (
          <div className="side-block">
            <span className="eyebrow">当前班级</span>
            <div className="class-pill">
              <div className="name">{ov.class_name}</div>
              <div className="meta">{ov.n_students} 名学生 · {CONTEXT.subject}</div>
            </div>
          </div>
        )}

        <div className="side-block">
          <span className="eyebrow">困境类型</span>
          <div className="legend">
            {TYPES.map(t => (
              <div className="row" key={t}>
                <span className="swatch" style={{ background: TYPE_COLOR[t] }} />{t}
              </div>
            ))}
          </div>
        </div>

        <div className="mode-badge">
          <span className="live" />
          运行模式：{status ? status.mode : '连接中'}
        </div>
      </aside>

      <main className="main">
        <div className="page-head">
          <div>
            <span className="eyebrow">Teacher Decision Dashboard</span>
            <h1>课堂学情总览</h1>
          </div>
          <div className="hint">认知—情感多模态诊断 · 点击任意学生查看画像与教学建议</div>
        </div>

        {!ov ? <div className="empty">加载班级数据…</div> : (
          <>
            <ClassStrip ov={ov} />

            <div className="grid-hero">
              <div className="card">
                <div className="card-head"><span className="eyebrow">Cognitive × Affective</span></div>
                <h2>学情象限图</h2>
                <p className="desc">横轴认知、纵轴情绪，二者共同决定学习表现；四象限即四类学习困境。</p>
                <QuadrantChart students={ov.students} selectedId={selected} onSelect={setSelected} />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                <TypeDistribution dist={ov.type_distribution} total={ov.n_students} />
                <EmotionTimeline timeline={ov.emotion_timeline} />
              </div>
            </div>

            <MasteryHeatmap ov={ov} onSelect={setSelected} />
          </>
        )}
      </main>

      <StudentPanel studentId={selected} context={CONTEXT} onClose={() => setSelected(null)} />
    </div>
  )
}
