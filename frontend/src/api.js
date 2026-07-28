// 与后端交互；后端未启动时回退到内置示例数据，保证界面可独立预览。

const CONCEPTS = ['有理数运算','整式加减','一元一次方程','因式分解','一元二次方程',
  '二次函数','不等式','相似三角形','勾股定理','概率初步']

export const TYPE_COLOR = {
  '正常':'var(--ok)', '认知缺陷型':'var(--def)', '情绪阻塞型':'var(--blk)', '双重风险型':'var(--risk)',
}

// —— 内置示例班级（约 12 人，覆盖四象限）——
function demoStudents() {
  const rows = [
    ['S01','王梓涵',78,84,0.82,'正常','低'],
    ['S02','李思远',41,79,0.6,'认知缺陷型','中'],
    ['S03','张晓彤',83,44,0.55,'情绪阻塞型','中'],
    ['S04','刘子墨',38,36,0.32,'双重风险型','高'],
    ['S05','陈欣怡',69,71,0.75,'正常','低'],
    ['S06','杨浩然',47,52,0.5,'双重风险型','高'],
    ['S07','赵梦琪',88,66,0.8,'正常','低'],
    ['S08','黄俊杰',44,81,0.68,'认知缺陷型','中'],
    ['S09','周雨桐',72,40,0.48,'情绪阻塞型','中'],
    ['S10','吴天佑',61,63,0.7,'正常','低'],
    ['S11','徐若曦',35,74,0.58,'认知缺陷型','中'],
    ['S12','孙嘉成',52,38,0.42,'双重风险型','高'],
  ]
  return rows.map(([id,name,c,e,eng,t,r]) => ({
    id, name, cognitive_score:c, emotion_score:e, engagement:eng,
    difficulty_type:t, risk_level:r,
    weak_concepts: c<60 ? [CONCEPTS[3],CONCEPTS[4],CONCEPTS[6]].slice(0, c<45?3:2) : [],
  }))
}

function demoOverview() {
  const students = demoStudents()
  const heatmap = students.map(s => CONCEPTS.map((_, j) => {
    const base = s.cognitive_score/100
    return Math.max(0.05, Math.min(0.98, base + (Math.sin((j+1)*(s.id.charCodeAt(2)))*0.22)))
  }))
  const timeline = Array.from({length:8}, (_,t) => ({
    t: t*5,
    positivity: Math.max(0, Math.min(1, 0.62 + 0.12*Math.sin(t/1.5))),
    engagement: Math.max(0, Math.min(1, 0.66 + 0.12*Math.sin(t/1.5 + .4))),
  }))
  const dist = {}
  students.forEach(s => dist[s.difficulty_type] = (dist[s.difficulty_type]||0)+1)
  return {
    class_id:'G1-3', class_name:'高一(3)班（示例）', n_students:students.length,
    concept_names:CONCEPTS, mastery_heatmap:heatmap, emotion_timeline:timeline,
    type_distribution:dist, class_positivity:63.4, class_engagement:0.63, students,
  }
}

function demoDetail(id) {
  const s = demoStudents().find(x => x.id===id)
  if (!s) return null
  const mastery = {}; CONCEPTS.forEach((c,j)=>{
    mastery[c] = Math.max(0.05, Math.min(0.98, s.cognitive_score/100 + Math.sin((j+1)*7)*0.2))
  })
  const curve = Array.from({length:12},(_,t)=>CONCEPTS.map(c=>{
    const f=t/11; return Math.max(0.05, Math.min(0.98, mastery[c]-0.25+0.25*f))
  }))
  const weak = Object.entries(mastery).sort((a,b)=>a[1]-b[1]).slice(0,3)
    .filter(([,v])=>v<0.6).map(([k])=>k)
  const forgetting = Object.entries(mastery).sort((a,b)=>a[1]-b[1]).slice(0,4)
    .map(([k,v])=>({concept:k, risk: +(0.6*(1-v)).toFixed(3)}))
  const ev = s.difficulty_type==='双重风险型' ? {'认知(IKT)':0.34,'音频':0.28,'文本':0.24,'图像':0.14}
    : s.difficulty_type==='情绪阻塞型' ? {'认知(IKT)':0.2,'音频':0.36,'文本':0.28,'图像':0.16}
    : s.difficulty_type==='认知缺陷型' ? {'认知(IKT)':0.44,'音频':0.18,'文本':0.22,'图像':0.16}
    : {'认知(IKT)':0.28,'音频':0.26,'文本':0.24,'图像':0.22}
  const comment = {
    '正常':'思路清晰，作业工整，本单元掌握扎实。',
    '认知缺陷型':'基础概念还有漏洞，计算步骤容易跳步出错，需要补一补。',
    '情绪阻塞型':'其实听得懂，但课上不太敢举手，遇到难题容易先放弃。',
    '双重风险型':'近期状态低落，作业敷衍，卷面涂改多，知识点也跟不上。',
  }[s.difficulty_type]
  return {...s, mastery, mastery_curve:curve, weak_concepts:weak, forgetting,
    evidence:ev, teacher_comment:comment}
}

function demoStrategy(id) {
  const s = demoDetail(id)
  const weak = s.weak_concepts.slice(0,2).join('、') || '薄弱知识点'
  const strat = {'课前':[], '课中':[], '课后':[]}
  if (['认知缺陷型','双重风险型'].includes(s.difficulty_type)) {
    strat['课前'].push(`备课时把「${weak}」拆成小步台阶，准备 1-2 道低起点例题作铺垫。`)
    strat['课中'].push(`讲到「${weak}」时放慢节奏，用变式题即时检验，优先请该生尝试基础一档。`)
    strat['课后'].push(`布置针对「${weak}」的分层练习，先巩固再拔高；安排 5 分钟个别答疑。`)
  }
  if (['情绪阻塞型','双重风险型'].includes(s.difficulty_type)) {
    strat['课前'].push('课前用一句轻松的私下问候降低其紧张感，营造安全的提问氛围。')
    strat['课中'].push('多给正向反馈与小成功体验，避免公开点名施压；关注其参与度变化。')
    strat['课后'].push('课后温和沟通了解畏难/情绪来源，必要时联系家长或转介心理老师协同关怀。')
  }
  if (s.difficulty_type==='正常') {
    strat['课前'].push('状态平稳，可适当增加一道拓展题保持挑战性。')
    strat['课中'].push('可请其分享思路，带动同伴讨论。')
    strat['课后'].push('保持节奏即可，提醒复习易遗忘的知识点。')
  }
  return {student_id:id, student_name:s.name, difficulty_type:s.difficulty_type,
    strategy:strat, engine:'模板(离线示例)'}
}

async function tryFetch(url, opts) {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), 1500)
  try {
    const res = await fetch(url, {...opts, signal: ctrl.signal})
    clearTimeout(timer)
    if (!res.ok) throw new Error(res.status)
    return await res.json()
  } catch (e) { clearTimeout(timer); return null }
}

export async function getOverview() {
  return (await tryFetch('/api/class/G1-3')) || demoOverview()
}
export async function getStudent(id) {
  return (await tryFetch(`/api/student/${id}`)) || demoDetail(id)
}
export async function getStrategy(id, context) {
  const real = await tryFetch('/api/strategy', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({student_id:id, context}),
  })
  return real || demoStrategy(id)
}
export async function getStatus() {
  return (await tryFetch('/api/status')) || {mode:'demo(前端离线示例)', strategy_engine:'—'}
}
