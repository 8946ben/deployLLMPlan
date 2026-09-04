'use strict';
/* UI 交互:状态管理、面板渲染、拖放画板、config.json 导入、实时报告 */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

let uidSeq = 1;
const state = {
  model: Object.assign({}, MODELS[0]),
  fwId: 'vllm',
  quantId: 'fp16',
  kvQuantId: 'fp16',
  ctx: 4096,
  conc: 1,
  promptLen: 1024,
  icId: 'auto', // GPU 卡间互联方式
  board: [], // {uid, type: 'gpu'|'cpu'|'ram', refId}
};

const byId = (list, id) => list.find(x => x.id === id);
const curFw = () => byId(FRAMEWORKS, state.fwId);
const curQuant = () => byId(curFw().quants, state.quantId) || curFw().quants[0];
const curKvQuant = () => byId(curFw().kvQuants, state.kvQuantId) || curFw().kvQuants[0];
const boardItems = (type) => state.board.filter(b => b.type === type).map(b => byId(type === 'gpu' ? GPUS : type === 'cpu' ? CPUS : RAMS, b.refId)).filter(Boolean);

function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

/* ---------------- 初始化 ---------------- */
function init() {
  buildModelSelect();
  buildFrameworkCards();
  buildPalette();
  bindEvents();
  const restored = loadState();
  if (restored) { applyStateToInputs(); }
  else { applyScenario('demo-v100', true); }
  renderAll();
}

function renderAll() {
  applyStateToInputs();
  renderModelSummary();
  renderQuantSelectors();
  renderBoard();
  renderReport();
}

/* ---------------- 模型面板 ---------------- */
function buildModelSelect() {
  $('#modelSel').innerHTML = '<option value="">— 选择预设模型 —</option>' +
    MODELS.map(m => '<option value="' + m.id + '">' + esc(m.name) + '</option>').join('');
}

function fillModelInputs(m) {
  $('#m-layers').value = m.layers; $('#m-hidden').value = m.hidden;
  $('#m-heads').value = m.heads; $('#m-kvheads').value = m.kvHeads;
  $('#m-headdim').value = m.headDim || '';
  $('#m-kvlayers').value = m.kvAttnLayers || m.layers;
  $('#m-inter').value = m.inter; $('#m-vocab').value = m.vocab;
  $('#m-ctx').value = m.maxCtx; $('#m-tied').checked = !!m.tied;
  $('#m-experts').value = m.experts || 1; $('#m-expertsActive').value = m.expertsActive || 1;
}

function readModelInputs() {
  const num = (sel, def) => { const v = parseFloat($(sel).value); return isFinite(v) && v > 0 ? v : def; };
  const m = state.model;
  m.layers = num('#m-layers', m.layers); m.hidden = num('#m-hidden', m.hidden);
  m.heads = num('#m-heads', m.heads); m.kvHeads = num('#m-kvheads', m.heads);
  m.headDim = parseFloat($('#m-headdim').value) || 0;
  m.kvAttnLayers = num('#m-kvlayers', m.layers);
  m.inter = num('#m-inter', m.inter); m.vocab = num('#m-vocab', m.vocab);
  m.maxCtx = num('#m-ctx', m.maxCtx); m.tied = $('#m-tied').checked;
  m.experts = Math.max(1, num('#m-experts', 1));
  m.expertsActive = Math.max(1, num('#m-expertsActive', 1));
}

function renderModelSummary() {
  const p = computeParams(state.model);
  const isMoe = (state.model.experts || 1) > 1;
  const kvLayers = state.model.kvAttnLayers || state.model.layers;
  const hybrid = kvLayers < state.model.layers;
  const kv = kvPerToken(state.model, 16);
  $('#modelSummary').innerHTML =
    '<div class="ms-row"><span>总参数量</span><b>' + fmtB(p.total) + ' <i>(' + (p.total / 1e8).toFixed(0) + '亿)</i></b></div>' +
    (isMoe ? '<div class="ms-row"><span>每token激活参数 (MoE)</span><b>' + fmtB(p.active) + '</b></div>' : '') +
    '<div class="ms-row"><span>FP16 权重体积</span><b>' + gb(p.total * 2) + '</b></div>' +
    '<div class="ms-row"><span>KV缓存 @FP16</span><b>' + (kv / 1024).toFixed(0) + ' KB/token · ' + gb(kv * state.ctx) + '/' + fmtK(state.ctx) + '上下文' + (hybrid ? ' <i>(混合注意力 ' + kvLayers + '/' + state.model.layers + ' 层有KV)</i>' : '') + '</div>';
}

/* ---------------- 框架面板 ---------------- */
function buildFrameworkCards() {
  $('#fwList').innerHTML = FRAMEWORKS.map(f =>
    '<div class="fw-card" data-fw="' + f.id + '">' +
    '<div class="fw-head"><b>' + f.name + '</b><span class="fw-check">✓</span></div>' +
    '<div class="fw-tag">' + esc(f.tag) + '</div></div>').join('');
  $$('#fwList .fw-card').forEach(el => el.addEventListener('click', () => {
    state.fwId = el.dataset.fw;
    const fw = curFw();
    if (!byId(fw.quants, state.quantId)) state.quantId = fw.quants[0].id;
    if (!byId(fw.kvQuants, state.kvQuantId)) state.kvQuantId = fw.kvQuants[0].id;
    renderAll(); saveState();
  }));
}

function renderQuantSelectors() {
  const fw = curFw();
  $$('#fwList .fw-card').forEach(el => el.classList.toggle('sel', el.dataset.fw === state.fwId));
  $('#quantSel').innerHTML = fw.quants.map(q => '<option value="' + q.id + '"' + (q.id === state.quantId ? ' selected' : '') + '>' + esc(q.name) + '</option>').join('');
  const kvRow = $('#kvRow');
  if (fw.kvQuants.length > 1) {
    kvRow.style.display = '';
    $('#kvQuantSel').innerHTML = fw.kvQuants.map(q => '<option value="' + q.id + '"' + (q.id === state.kvQuantId ? ' selected' : '') + '>' + esc(q.name) + '</option>').join('');
  } else { kvRow.style.display = 'none'; }
  $('#fwNote').textContent = fw.desc;
}

/* ---------------- 硬件库 + 画板 ---------------- */
function buildPalette() {
  const gpuCard = (g) =>
    '<div class="pal-card vendor-' + g.vendor + '" draggable="true" data-type="gpu" data-id="' + g.id + '">' +
    '<div class="pal-name">🖧 ' + esc(g.name) + '</div>' +
    '<div class="pal-spec">' + g.vram + 'GB · ' + g.bw + 'GB/s' + (g.vendor === 'nvidia' ? ' · CC' + g.cc : '') + '</div>' +
    '<div class="pal-note">' + esc(g.note) + '</div><button class="add-btn" title="添加到画板">+</button></div>';
  const cpuCard = (c) =>
    '<div class="pal-card vendor-cpu" draggable="true" data-type="cpu" data-id="' + c.id + '">' +
    '<div class="pal-name">🧠 ' + esc(c.name) + '</div>' +
    '<div class="pal-spec">' + c.cores + '核 · ' + c.channels + '通道DDR5</div>' +
    '<div class="pal-note">' + esc(c.note) + '</div><button class="add-btn">+</button></div>';
  const ramCard = (r) =>
    '<div class="pal-card vendor-ram" draggable="true" data-type="ram" data-id="' + r.id + '">' +
    '<div class="pal-name">🧩 ' + esc(r.name) + '</div>' +
    '<div class="pal-spec">+38.4GB/s/条</div>' +
    '<div class="pal-note">' + esc(r.note) + '</div><button class="add-btn">+</button></div>';
  $('#palGpu').innerHTML = GPUS.map(gpuCard).join('');
  $('#palCpu').innerHTML = CPUS.map(cpuCard).join('');
  $('#palRam').innerHTML = RAMS.map(ramCard).join('');
  $$('.pal-card').forEach(el => {
    el.addEventListener('dragstart', e => { e.dataTransfer.setData('text/plain', el.dataset.type + ':' + el.dataset.id); e.dataTransfer.effectAllowed = 'copy'; });
    el.addEventListener('click', e => { if (e.target.classList.contains('add-btn')) return; addItem(el.dataset.type, el.dataset.id); });
    const btn = el.querySelector('.add-btn');
    if (btn) btn.addEventListener('click', () => addItem(el.dataset.type, el.dataset.id));
  });
}

function addItem(type, refId) {
  state.board.push({ uid: uidSeq++, type, refId });
  renderBoard(); renderReport(); saveState();
}

function renderBoard() {
  const board = $('#board');
  const groups = [];
  const idx = {};
  for (const it of state.board) {
    const key = it.type + ':' + it.refId;
    if (idx[key] === undefined) { idx[key] = groups.length; groups.push({ type: it.type, refId: it.refId, count: 0 }); }
    groups[idx[key]].count++;
  }
  const cat = { gpu: GPUS, cpu: CPUS, ram: RAMS };
  board.innerHTML = groups.length ? groups.map(gr => {
    const item = byId(cat[gr.type], gr.refId); if (!item) return '';
    const spec = gr.type === 'gpu' ? item.vram + 'GB · ' + item.bw + 'GB/s' + (item.vendor === 'nvidia' ? ' · CC' + item.cc : '')
      : gr.type === 'cpu' ? item.cores + '核 · ' + item.channels + '通道' : item.gb + 'GB · ' + item.bw + 'GB/s';
    return '<div class="board-item type-' + gr.type + '" data-key="' + gr.type + ':' + gr.refId + '">' +
      '<div class="bi-head"><span class="bi-name">' + (gr.type === 'gpu' ? '🖧' : gr.type === 'cpu' ? '🧠' : '🧩') + ' ' + esc(item.name) + '</span>' +
      '<span class="bi-count">×' + gr.count + '</span></div>' +
      '<div class="bi-spec">' + spec + '</div>' +
      '<div class="bi-ops"><button data-op="inc">+1</button><button data-op="dec">−1</button><button data-op="del" class="danger">移除</button></div></div>';
  }).join('') : '<div class="board-empty">画板为空 — 点击左侧硬件卡的 <b>+</b>,或直接把硬件拖到这里<br><small>试试右上角「快速场景」</small></div>';

  $$('#board .board-item').forEach(el => {
    const [type, refId] = el.dataset.key.split(':');
    el.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => {
      const op = btn.dataset.op;
      if (op === 'inc') addItem(type, refId);
      else if (op === 'dec') { for (let i = state.board.length - 1; i >= 0; i--) if (state.board[i].type === type && state.board[i].refId === refId) { state.board.splice(i, 1); break; } }
      else state.board = state.board.filter(b => !(b.type === type && b.refId === refId));
      renderBoard(); renderReport(); saveState();
    }));
  });

  // 汇总条
  const gpus = boardItems('gpu'), cpus = boardItems('cpu'), rams = boardItems('ram');
  const vram = gpus.reduce((s, g) => s + g.vram, 0);
  const bwSum = gpus.reduce((s, g) => s + g.bw, 0);
  const ramGB = rams.reduce((s, r) => s + r.gb, 0);
  const ic = resolveInterconnect(gpus, state.icId);
  const icText = ic ? (ic.bw ? ic.name.split(' (')[0] + ' · <b>' + ic.bw + '</b> GB/s' : ic.name) : '—';
  $('#boardTotals').innerHTML =
    '<span>GPU <b>' + gpus.length + '</b> 块</span><span>显存 <b>' + vram + '</b> GB</span>' +
    '<span>带宽合计 <b>' + bwSum + '</b> GB/s</span>' +
    '<span>互联 <b>' + (gpus.length > 1 ? icText : '单卡') + '</b></span>' +
    '<span>CPU <b>' + cpus.length + '</b> 个平台</span><span>内存 <b>' + ramGB + '</b> GB</span>';
  // 自动模式的互联选项显示当前推断结果
  const autoOpt = $('#icSel option[value="auto"]');
  if (autoOpt) autoOpt.textContent = ic ? '自动 — ' + ic.name.split(' (')[0] + (ic.bw ? ' (' + ic.bw + ' GB/s)' : '') : '自动(按GPU推断)';
  renderTopology();
}

/* ---------------- 集群拓扑图 ---------------- */
function renderTopology() {
  const el = $('#topo');
  const gpus = boardItems('gpu'), cpus = boardItems('cpu'), rams = boardItems('ram');
  const ic = resolveInterconnect(gpus, state.icId);
  if (!gpus.length && !cpus.length && !rams.length) {
    el.innerHTML = '<div class="topo-empty">放置硬件后,这里显示集群拓扑与每条链路的速率</div>';
    return;
  }
  const W = 720, NW = 116, NH = 50, GAP = 14, cx = W / 2;
  const shortName = (x) => x.name.replace(/^NVIDIA |^AMD |^Apple /, '');
  // 同型号 GPU 合并为一个节点
  const groups = [];
  for (const g of gpus) {
    const last = groups[groups.length - 1];
    if (last && last.item.id === g.id) last.n++; else groups.push({ item: g, n: 1 });
  }
  const n = groups.length;
  const nCards = gpus.length;
  const startX = Math.max(8, (W - (n * NW + Math.max(n - 1, 0) * GAP)) / 2);
  const fabMulti = ic && ic.type === 'fabric' && nCards >= 3;
  const gpuY = fabMulti ? 96 : (ic && ic.type === 'fabric' && nCards === 2 ? 64 : 44);
  const node = (x, y, w, title, sub, cls) =>
    '<g class="topo-node ' + cls + '" transform="translate(' + x.toFixed(1) + ',' + y + ')">' +
    '<rect width="' + w + '" height="' + NH + '" rx="10"></rect>' +
    '<text class="t" x="' + w / 2 + '" y="21">' + esc(title) + '</text>' +
    '<text class="s" x="' + w / 2 + '" y="38">' + esc(sub) + '</text></g>';
  const line = (x1, y1, x2, y2, cls) =>
    '<line class="topo-link ' + cls + '" x1="' + x1.toFixed(1) + '" y1="' + y1 + '" x2="' + x2.toFixed(1) + '" y2="' + y2 + '"></line>';
  const label = (x, y, txt, cls) =>
    '<text class="topo-label ' + cls + '" x="' + x.toFixed(1) + '" y="' + y + '" text-anchor="middle">' + esc(txt) + '</text>';
  const icLabel = ic ? (ic.bw ? ic.name.split(' (')[0] + ' · ' + ic.bw + ' GB/s' : ic.name) : '';
  let svg = '';
  // GPU 互联层:3卡以上 fabric 走交换芯片星型;2卡异型号直连;2卡同型号画卡间回环;PCIe/网络走 CPU 星型
  if (ic && ic.type === 'fabric') {
    if (fabMulti) {
      const swW = 210;
      for (let i = 0; i < n; i++) svg += line(cx, 68, startX + i * (NW + GAP) + NW / 2, gpuY, 'fab');
      svg += node(cx - swW / 2, 18, swW, ic.id === 'xgmi' ? 'IF 总线 (xGMI)' : 'NVSwitch / NVLink 交换',
        icLabel + (nCards > n ? ' · ' + nCards + ' 卡全互联' : ''), 'sw');
    } else if (nCards === 2 && n === 2) {
      svg += line(startX + NW, gpuY + NH / 2, startX + NW + GAP, gpuY + NH / 2, 'fab');
      svg += label(startX + NW + GAP / 2, gpuY - 8, icLabel, 'fab');
    } else if (nCards === 2 && n === 1) {
      const x0 = startX, x1 = startX + NW;
      svg += line(x0 + 26, gpuY, x0 + 26, gpuY - 22, 'fab') + line(x0 + 26, gpuY - 22, x1 - 26, gpuY - 22, 'fab') + line(x1 - 26, gpuY - 22, x1 - 26, gpuY, 'fab');
      svg += label(cx, gpuY - 30, icLabel + '(卡间直连)', 'fab');
    }
  }
  for (let i = 0; i < n; i++) {
    const gr = groups[i];
    svg += node(startX + i * (NW + GAP), gpuY, NW, shortName(gr.item), (gr.n > 1 ? '×' + gr.n + ' · ' : '') + gr.item.vram + 'GB', 'g');
  }
  if (ic && ic.type === 'unified' && n) {
    svg += label(cx, gpuY + NH + 22, '统一内存架构:权重与 KV 共享 ' + gpus[0].bw + ' GB/s 带宽(无独立显存/内存之分)', 'ram');
  }
  // CPU / PCIe 层
  let H = gpuY + NH + 12;
  if (n >= 1 || cpus.length) {
    const cpuY = gpuY + NH + 54;
    if (n >= 1 && ic && (ic.type === 'pcie' || ic.type === 'net')) {
      for (let i = 0; i < n; i++) svg += line(startX + i * (NW + GAP) + NW / 2, gpuY + NH, cx, cpuY, 'pcie');
      svg += label(cx + 130, (gpuY + NH + cpuY) / 2 + 12, icLabel, 'pcie');
    } else if (n >= 1) {
      const pgen = Math.min(...gpus.map(g => g.pcieGen || 3));
      for (let i = 0; i < n; i++) svg += line(startX + i * (NW + GAP) + NW / 2, gpuY + NH, cx, cpuY, 'pcie');
      svg += label(cx + 130, (gpuY + NH + cpuY) / 2 + 12, 'PCIe ' + pgen + '.0 x16 · ' + (pgen >= 5 ? 64 : pgen >= 4 ? 32 : 16) + ' GB/s(接主机)', 'pcie');
    }
    const chan = cpus.reduce((s, c) => s + c.channels, 0);
    svg += node(cx - 75, cpuY, 150, cpus.length ? shortName(cpus[0]) + (cpus.length > 1 ? ' ×' + cpus.length : '') : '主机(未放CPU)',
      cpus.length ? cpus[0].cores + '核 · ' + chan + '通道' : '建议加 CPU 平台 + 内存', 'c');
    H = cpuY + NH + 12;
    // 内存层
    if (rams.length && cpus.length) {
      const ramY = cpuY + NH + 44;
      const ramGB = rams.reduce((s, r) => s + r.gb, 0);
      const rbw = Math.min(rams.length * 38.4, chan * 38.4).toFixed(0);
      svg += line(cx, cpuY + NH, cx, ramY, 'ram');
      svg += label(cx, ramY - 8, 'DDR5 ×' + rams.length + ' 条 · ' + Math.min(rams.length, chan) + ' 通道 · ~' + rbw + ' GB/s', 'ram');
      svg += node(cx - 75, ramY, 150, '内存 ' + ramGB + ' GB', rams.length + ' × DDR5-4800', 'r');
      H = ramY + NH + 10;
    }
  }
  el.innerHTML = '<svg viewBox="0 0 ' + W + ' ' + Math.max(H, 120).toFixed(0) + '" preserveAspectRatio="xMidYMin meet">' + svg + '</svg>';
}

/* ---------------- 分析报告 ---------------- */
/* 展开状态跨重渲染保留:键如 verdict / weights / sp-decode */
const expandedDetails = new Set();

function renderReport() {
  const result = analyze({
    model: state.model, fw: curFw(), quant: curQuant(), kvQuant: curKvQuant(),
    ctx: state.ctx, conc: state.conc, promptLen: state.promptLen,
    gpus: boardItems('gpu'), cpus: boardItems('cpu'), rams: boardItems('ram'),
    ic: resolveInterconnect(boardItems('gpu'), state.icId),
  });
  const m = result.mem, s = result.speed, D = result.details;
  const scale = Math.max(m.requiredB, m.totalVramB, 1);
  const seg = (bytes, cls, label) =>
    '<div class="seg ' + cls + '" style="width:' + (bytes / scale * 100) + '%"><span>' + label + '</span></div>';
  const boundaryPct = m.totalVramB > 0 ? m.totalVramB / scale * 100 : 0;

  let html = '<div class="banner ' + result.verdict + ' exp' + (expandedDetails.has('verdict') ? ' open' : '') + '" data-exp="verdict">' + result.bannerText + '<span class="chev">▸</span></div>';
  if (expandedDetails.has('verdict')) html += '<div class="exp-panel">' + fmtLines(D.verdict) + '</div>';

  // 配置摘要
  html += '<div class="cfg-line">' + esc(state.model.name || '自定义模型') + ' · ' + fmtB(m.paramsTotal) +
    (m.paramsActive < m.paramsTotal ? ' (激活' + fmtB(m.paramsActive) + ')' : '') +
    ' · ' + curFw().name + ' ' + curQuant().name + ' · 上下文' + fmtK(state.ctx) + ' × ' + state.conc + '路</div>';

  // 显存构成
  html += '<h3>显存需求 <small class="hint-exp">点击各行展开计算</small></h3><table class="mem-table">' +
    rowExp('weights', '模型权重 (' + curQuant().name + ')', gb(m.weightsB), D.weights) +
    rowExp('kv', 'KV 缓存 (' + fmtK(state.ctx) + '×' + state.conc + '路)', gb(m.kvB), D.kv) +
    rowExp('overhead', '框架/激活开销', gb(m.overheadB), D.overhead) +
    rowExp('required', '<b>合计需求</b>', '<b>' + gb(m.requiredB) + '</b>', D.required) +
    rowExp('vram', '硬件显存总量', gb(m.totalVramB), D.vram) +
    (m.requiredB <= m.totalVramB
      ? rowExp('surplus', '剩余', gb(m.totalVramB - m.requiredB), D.surplus)
      : rowExp('surplus', '<span class="bad">缺口</span>', '<span class="bad">' + gb(m.requiredB - m.totalVramB) + '</span>', D.surplus)) +
    '</table>';
  html += '<div class="membar-wrap"><div class="membar">' + seg(m.weightsB, 's-w', '') + seg(m.kvB, 's-kv', '') + seg(m.overheadB, 's-oh', '') +
    (m.totalVramB > m.requiredB ? '<div class="seg s-free" style="width:' + ((m.totalVramB - m.requiredB) / scale * 100) + '%"></div>' : '') +
    (m.totalVramB > 0 ? '<div class="mem-boundary" style="left:' + boundaryPct + '%"></div>' : '') + '</div>' +
    '<div class="mem-legend"><i class="s-w"></i>权重 ' + gb(m.weightsB) + ' <i class="s-kv"></i>KV ' + gb(m.kvB) + ' <i class="s-oh"></i>开销 ' + gb(m.overheadB) +
    (m.totalVramB > m.requiredB ? ' <i class="s-free"></i>空闲 ' + gb(m.totalVramB - m.requiredB) : ' <i class="s-over"></i>超出 ' + gb(m.requiredB - m.totalVramB)) +
    ' <span class="boundary-note">┃显存上限</span></div></div>';

  // 检查清单
  html += '<h3>检查清单</h3><ul class="checks">' + result.checks.map(c =>
    '<li class="' + c.level + '"><span class="dot"></span>' + c.msg + '</li>').join('') + '</ul>';

  // 速度
  if (s) {
    const tps = (v) => v == null ? '—' : (v >= 100 ? v.toFixed(0) : v.toFixed(1)) + ' tok/s';
    const SPD = D.speed || {};
    html += '<h3>性能预估 <small class="hint-exp">点击各项展开计算</small></h3><div class="stats">' +
      statExp('sp-decode', tps(s.decodeTps), '解码速度(单流)', SPD.decode) +
      statExp('sp-best', s.decodeTpsBest > s.decodeTps * 1.05 ? tps(s.decodeTpsBest) : '—', '解码(空上下文)', SPD.best) +
      statExp('sp-prefill', tps(s.prefillTps), '预填充速度', SPD.prefill) +
      statExp('sp-ttft', s.ttftS == null ? '—' : (s.ttftS >= 1 ? s.ttftS.toFixed(1) + ' s' : Math.round(s.ttftS * 1000) + ' ms'), '首字延迟TTFT(' + fmtK(state.promptLen) + ' tokens)', SPD.ttft) +
      statExp('sp-conc', s.maxConc === null ? '—' : (s.maxConc > 0 ? s.maxConc + ' 路' : '0 (超限)'), '最大并发@' + fmtK(state.ctx), SPD.maxConc) +
      statExp('sp-agg', s.aggTps == null ? '—' : tps(s.aggTps), '当前' + state.conc + '路总吞吐', SPD.agg) + '</div>';
    const SPD_KEYS = { 'sp-decode': 'decode', 'sp-best': 'best', 'sp-prefill': 'prefill', 'sp-ttft': 'ttft', 'sp-conc': 'maxConc', 'sp-agg': 'agg' };
    const openSp = Object.keys(SPD_KEYS).find(k => expandedDetails.has(k));
    if (openSp && SPD[SPD_KEYS[openSp]]) html += '<div class="exp-panel spd-panel">' + fmtLines(SPD[SPD_KEYS[openSp]]) + '</div>';
    if (result.mode !== 'gpu') {
      html += lineExp('sp-layers', 'layer-line', '层分布:约 <b>' + s.gpuLayers + '</b> 层在 GPU,其余在 CPU 内存', SPD.layers);
    }
    const gpus = boardItems('gpu');
    const bw = result.mode === 'cpu' ? s.cpuBwBps / GB : gpus.reduce((a, g) => a + g.bw, 0);
    const commTxt = s.tpCommMs > 0 ? ' · TP通信 ' + Math.round(s.tpCommMs * 1000) + 'µs/token(' + (s.icName || '').split(' (')[0] + (s.icBw ? ' ' + s.icBw + 'GB/s' : '') + ')' : '';
    html += lineExp('sp-bneck', 'bottleneck-line', '瓶颈分析:每 token 需读取 <b>' + (s.bytesPerTok / GB).toFixed(1) + ' GB</b>' +
      (result.mode === 'gpu' ? ',解码受显存带宽限制(有效 ~' + bw.toFixed(0) + ' GB/s,带宽利用率 ~' + Math.round(s.bwUtil * 100) + '%' + commTxt + ')' :
        result.mode === 'offload' ? ',其中 CPU 部分走内存带宽 ' + (s.cpuBwBps / GB).toFixed(0) + ' GB/s,是主要瓶颈' :
          ',解码完全受内存带宽 ' + (s.cpuBwBps / GB).toFixed(0) + ' GB/s 限制'), SPD.bneck);
  } else {
    html += '<h3>性能预估</h3><div class="no-speed">当前配置无法运行,不产生速度估算。</div>';
  }

  // 建议
  if (result.suggestions.length) {
    html += '<h3>建议</h3><ul class="suggestions">' + result.suggestions.map(t => '<li>💡 ' + t + '</li>').join('') + '</ul>';
  }
  html += '<div class="disclaimer">* 估算基于公开规格与经验公式(带宽/算力利用率模型),用于部署规划参考,实测会因实现细节而异。</div>';
  $('#report').innerHTML = html;
}
function fmtLines(lines) { return (lines || []).map(l => esc(l)).join('<br>'); }
function rowExp(key, label, val, lines) {
  const open = expandedDetails.has(key);
  return '<tr class="exp-row' + (open ? ' open' : '') + '" data-exp="' + key + '"><td>' + label + '<span class="chev">▸</span></td>' +
    '<td class="num">' + val + '</td></tr>' +
    (open ? '<tr class="row-detail"><td colspan="2">' + fmtLines(lines) + '</td></tr>' : '');
}
function statExp(key, v, label, lines) {
  const open = expandedDetails.has(key);
  return '<div class="stat exp' + (open ? ' open' : '') + '" data-exp="' + key + '" title="' + esc(label) + ' — 点击展开计算">' +
    '<div class="stat-v">' + v + '</div><div class="stat-l">' + label + '<span class="chev">▸</span></div></div>';
}
function lineExp(key, cls, inner, lines) {
  const open = expandedDetails.has(key);
  return '<div class="' + cls + ' exp' + (open ? ' open' : '') + '" data-exp="' + key + '">' + inner + '<span class="chev">▸</span></div>' +
    (open ? '<div class="exp-panel">' + fmtLines(lines) + '</div>' : '');
}

/* ---------------- 场景 / 导入 / 持久化 ---------------- */
function applyScenario(id, silent) {
  const sc = byId(SCENARIOS, id); if (!sc) return;
  const preset = byId(MODELS, sc.model);
  if (preset) state.model = Object.assign({}, preset);
  state.fwId = sc.fw; state.quantId = sc.quant; state.kvQuantId = sc.kvQuant || curFw().kvQuants[0].id;
  state.ctx = sc.ctx; state.conc = sc.conc || 1; state.icId = 'auto';
  state.board = sc.board.map(([type, refId]) => ({ uid: uidSeq++, type, refId }));
  $('#scenarioSel').value = id;
  if (!silent) toast('已载入场景:' + sc.name);
  renderAll(); saveState();
}

function importConfigFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const j = JSON.parse(reader.result);
      const pick = (...keys) => { for (const k of keys) if (j[k] !== undefined) return j[k]; };
      const m = {
        name: (j.architectures && j.architectures[0]) || file.name.replace(/\.json$/i, ''),
        layers: pick('num_hidden_layers', 'n_layer'), hidden: pick('hidden_size', 'n_embd'),
        heads: pick('num_attention_heads', 'n_head'), kvHeads: pick('num_key_value_heads', 'n_head_kv') || pick('num_attention_heads', 'n_head'),
        headDim: pick('head_dim') || 0, inter: pick('intermediate_size', 'n_inner'),
        vocab: pick('vocab_size'), maxCtx: pick('max_position_embeddings', 'n_ctx', 'max_seq_len') || 4096,
        tied: !!pick('tie_word_embeddings'), experts: pick('num_experts', 'n_routed_experts') || 1,
        expertsActive: pick('num_experts_per_tok', 'n_experts_per_tok') || 1,
      };
      if (!m.layers || !m.hidden || !m.heads) throw new Error('缺少关键字段(num_hidden_layers/hidden_size/num_attention_heads)');
      state.model = m;
      state.ctx = m.maxCtx; // 默认使用模型原生上下文
      $('#modelSel').value = '';
      toast('已导入:' + m.name + ' (可在下方微调字段)');
      renderAll(); saveState();
    } catch (err) { toast('导入失败:' + err.message, true); }
  };
  reader.readAsText(file);
}

const LS_KEY = 'llmdeploy-state-v1';
function saveState() {
  try { localStorage.setItem(LS_KEY, JSON.stringify({ model: state.model, fwId: state.fwId, quantId: state.quantId, kvQuantId: state.kvQuantId, ctx: state.ctx, conc: state.conc, promptLen: state.promptLen, icId: state.icId, board: state.board })); } catch (e) { /* ignore */ }
}
function loadState() {
  try {
    const raw = localStorage.getItem(LS_KEY); if (!raw) return false;
    const j = JSON.parse(raw);
    if (!j.model || !byId(FRAMEWORKS, j.fwId) || !Array.isArray(j.board)) return false;
    Object.assign(state, j); return true;
  } catch (e) { return false; }
}

let toastTimer = null;
function toast(msg, isErr) {
  const t = $('#toast');
  t.textContent = msg; t.className = 'toast show' + (isErr ? ' err' : '');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.className = 'toast'; }, 2600);
}

function applyStateToInputs() {
  fillModelInputs(state.model);
  $('#s-ctx').value = state.ctx; $('#s-conc').value = state.conc; $('#s-prompt').value = state.promptLen;
}

/* ---------------- 事件绑定 ---------------- */
function bindEvents() {
  $('#modelSel').addEventListener('change', e => {
    const m = byId(MODELS, e.target.value); if (!m) return;
    state.model = Object.assign({}, m); state.ctx = m.maxCtx; // 默认使用模型原生上下文
    renderAll(); saveState();
  });
  $$('#panel-model input, #panel-model select').forEach(el => el.addEventListener('input', () => {
    readModelInputs(); renderModelSummary(); renderReport(); saveState();
  }));
  $('#quantSel').addEventListener('change', e => { state.quantId = e.target.value; renderReport(); saveState(); });
  $('#kvQuantSel').addEventListener('change', e => { state.kvQuantId = e.target.value; renderReport(); saveState(); });
  $('#s-ctx').addEventListener('input', e => { state.ctx = Math.max(1, parseInt(e.target.value) || 1); renderModelSummary(); renderReport(); saveState(); });
  $('#s-conc').addEventListener('input', e => { state.conc = Math.max(1, parseInt(e.target.value) || 1); renderReport(); saveState(); });
  $('#s-prompt').addEventListener('input', e => { state.promptLen = Math.max(1, parseInt(e.target.value) || 1); renderReport(); saveState(); });

  const board = $('#board');
  board.addEventListener('dragover', e => { e.preventDefault(); board.classList.add('dragover'); });
  board.addEventListener('dragleave', () => board.classList.remove('dragover'));
  board.addEventListener('drop', e => {
    e.preventDefault(); board.classList.remove('dragover');
    const data = e.dataTransfer.getData('text/plain');
    const [type, id] = data.split(':');
    if (type === 'gpu' || type === 'cpu' || type === 'ram') addItem(type, id);
  });
  $('#clearBoard').addEventListener('click', () => { state.board = []; renderBoard(); renderReport(); saveState(); });
  $('#icSel').innerHTML = INTERCONNECTS.map(i => '<option value="' + i.id + '">' + esc(i.name + (i.bw ? ' · ' + i.bw + ' GB/s' : '')) + '</option>').join('');
  $('#icSel').addEventListener('change', e => { state.icId = e.target.value || 'auto'; renderBoard(); renderReport(); saveState(); });
  $('#scenarioSel').innerHTML = '<option value="">— 快速场景 —</option>' + SCENARIOS.map(s => '<option value="' + s.id + '">' + esc(s.name) + '</option>').join('');
  $('#scenarioSel').addEventListener('change', e => { if (e.target.value) applyScenario(e.target.value); });

  $('#importBtn').addEventListener('click', () => $('#importFile').click());
  $('#importFile').addEventListener('change', e => { if (e.target.files[0]) importConfigFile(e.target.files[0]); e.target.value = ''; });

  // 报告各项点击展开/收起计算细节(性能类 stat 卡片同时只展开一个)
  $('#report').addEventListener('click', e => {
    const t = e.target.closest('[data-exp]');
    if (!t) return;
    const key = t.dataset.exp;
    if (key.indexOf('sp-') === 0) {
      for (const k of [...expandedDetails]) if (k.indexOf('sp-') === 0 && k !== key) expandedDetails.delete(k);
    }
    if (expandedDetails.has(key)) expandedDetails.delete(key); else expandedDetails.add(key);
    renderReport();
  });
}

document.addEventListener('DOMContentLoaded', init);
