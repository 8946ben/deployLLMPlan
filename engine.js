'use strict';
/* 分析引擎:根据模型配置/框架/量化/硬件,计算显存需求、可行性检查与速度估算。纯函数,不触碰DOM。 */

const GiB = 1024 ** 3;
const GB = 1e9; // 硬件容量/带宽按十进制 GB 计,与厂商规格书一致
const DDR5_PER_CHANNEL = 38.4; // GB/s,DDR5-4800

function gb(x) {
  const v = x / GB;
  return (v >= 100 ? v.toFixed(0) : v.toFixed(1)) + ' GB';
}
function fmtK(n) { return n >= 1024 ? Math.round(n / 1024) + 'K' : String(n); }
function fmtB(n) { return (n / 1e9).toFixed(1) + 'B'; }

/* 由 transformer 配置估算参数量(总参数 / 每 token 激活参数,MoE 时两者不同) */
function computeParams(m) {
  const H = +m.hidden || 0, L = +m.layers || 0, A = +m.heads || 0;
  const K = +m.kvHeads || A, I = +m.inter || 0, V = +m.vocab || 0;
  const d = +m.headDim || (A ? H / A : 0);
  // 注意力:Q/K/V/O 投影 (GQA: K/V 头数更少);MLA 等压缩注意力可用 attnPerLayer 覆盖每层参数量
  const attn = +m.attnPerLayer > 0 ? +m.attnPerLayer : (H * (A * d) + 2 * H * (K * d) + (A * d) * H);
  const E = +m.experts || 1;
  const topk = Math.min(Math.max(+m.expertsActive || 1, 1), E);
  const mlpDense = 3 * H * I;
  const mlpTotal = E > 1 ? E * mlpDense : mlpDense;
  const mlpActive = E > 1 ? topk * mlpDense : mlpDense;
  const router = E > 1 ? H * E : 0;
  const perLayer = attn + mlpTotal + router + 4 * H; // 4H≈两层RMSNorm
  const embed = V * H;
  const lmHead = m.tied ? 0 : V * H;
  // extraParams: 计入权重但不参与每 token 读取的参数(n-gram 嵌入表、视觉塔等稀疏访问部分)
  const total = Math.round(perLayer * L + embed + lmHead + H + (+m.extraParams || 0));
  const activePerLayer = attn + mlpActive + router + 4 * H;
  const active = Math.round(activePerLayer * L + embed + lmHead + H);
  return { total, active, headDim: d };
}

/* 每 token 的 KV 缓存字节数 = 2(K和V) × 带KV缓存的全注意力层数 × KV头数 × 头维度 × 每元素字节数
   混合注意力架构(如 Qwen3.8)中线性注意力层状态恒定,不产生逐 token KV,用 kvAttnLayers 区分。 */
function kvPerToken(m, kvBpw) {
  if (+m.kvBytesPerTok > 0) return m.kvBytesPerTok * (kvBpw / 16); // MLA 压缩 KV:按 FP16 基准字节数折算
  const d = +m.headDim || m.hidden / m.heads;
  const kvLayers = +m.kvAttnLayers || +m.layers;
  return 2 * kvLayers * (+m.kvHeads || +m.heads) * d * (kvBpw / 8);
}

/* 多卡读取 bytes 的理想耗时(秒)。
   tensor(张量并行):各卡并行读各自分片,取最慢;无NVLink时通信惩罚0.8。
   pipeline(层切分,llama.cpp式):各卡按显存比例分担,逐卡串行。 */
function splitTime(bytes, gpus, tpMode, allNvlink) {
  if (!gpus.length || bytes <= 0) return 0;
  if (tpMode === 'tensor' && gpus.length > 1) {
    const share = bytes / gpus.length;
    let t = 0;
    for (const g of gpus) t = Math.max(t, share / (g.bw * GB));
    return allNvlink ? t : t / 0.8;
  }
  const totalVram = gpus.reduce((s, g) => s + g.vram, 0);
  let t = 0;
  for (const g of gpus) t += (bytes * (g.vram / totalVram)) / (g.bw * GB);
  return t;
}

/* 主入口。cfg: {model, fw, quant, kvQuant, ctx, conc, promptLen, gpus, cpus, rams} */
function analyze(cfg) {
  const { model, fw, quant, kvQuant } = cfg;
  const ctx = cfg.ctx || 1024, conc = Math.max(1, cfg.conc || 1);
  const promptLen = cfg.promptLen || 1024;
  const gpus = cfg.gpus || [], cpus = cfg.cpus || [], rams = cfg.rams || [];
  const checks = [];
  const addCheck = (level, msg) => checks.push({ level, msg });

  const p = computeParams(model);
  const headDim = p.headDim || model.headDim || model.hidden / model.heads;

  /* ---------- 基础尺寸 ---------- */
  const weightsB = p.total * quant.bpw / 8;         // 权重体积
  const activeB = p.active * quant.bpw / 8;         // 每 token 实际读取的权重
  const kvTok = kvPerToken(model, kvQuant.bpw);     // 每 token KV 字节
  const kvB = kvTok * ctx * conc;                   // 全部并发×上下文的 KV
  const overheadB = fw.fixedOH * GB + fw.fracOH * weightsB; // 框架/激活开销
  const requiredB = weightsB + kvB + overheadB;
  const totalVramB = gpus.reduce((s, g) => s + g.vram * GB, 0);
  const minVramB = gpus.length ? Math.min(...gpus.map(g => g.vram)) * GB : 0;
  const allNvlink = gpus.length > 0 && gpus.every(g => g.nvlink);

  /* ---------- CPU / 内存侧 ---------- */
  const ramTotalB = rams.reduce((s, r) => s + r.gb * GB, 0);
  const channels = cpus.reduce((s, c) => s + c.channels, 0);
  const ramModuleBw = rams.reduce((s, r) => s + r.bw, 0);
  const cpuBwBps = Math.min(ramModuleBw, channels * DDR5_PER_CHANNEL) * GB;
  const cpuTf = cpus.reduce((s, c) => s + c.cores * 80e9, 0) / 1e12; // AVX-512 估算

  /* ---------- 检查清单 ---------- */
  if (!gpus.length && !cpus.length && !rams.length) {
    addCheck('fail', '画板上还没有任何硬件:从下方硬件库添加 GPU(或 CPU 平台+内存)。');
  }
  if (!gpus.length && !fw.cpuOnly) {
    addCheck('fail', fw.name + ' 需要 GPU 才能运行(纯 CPU 部署可换 llama.cpp / HF Transformers)。');
  }
  for (const g of gpus) {
    if (g.vendor === 'nvidia') {
      if (g.cc < fw.minCC) {
        addCheck('fail', g.name + ' 架构过低:' + fw.name + ' 需要计算能力 ≥ ' + fw.minCC + ',该卡仅 CC ' + g.cc + '。');
      }
      if (quant.needCC && g.cc < quant.needCC) {
        addCheck('fail', quant.name + ' 需要 CC ≥ ' + quant.needCC + '(Ada/Hopper),' + g.name + ' 仅 CC ' + g.cc + ',请改用 FP16/INT4。');
      } else if (quant.bf16 && g.cc < 8.0) {
        addCheck('warn', g.name + '(Volta)不原生支持 BF16,建议改用 FP16。');
      }
      if (kvQuant.needCC && g.cc < kvQuant.needCC) {
        addCheck('fail', 'KV 缓存 ' + kvQuant.name + ' 需要 CC ≥ ' + kvQuant.needCC + '。');
      }
    } else if (g.vendor === 'amd') {
      if (!fw.rocm) addCheck('fail', fw.name + ' 不支持 AMD GPU(' + g.name + '),可换 llama.cpp/vLLM(ROCm)。');
      else if (quant.fp8) addCheck('fail', 'FP8 量化仅支持 NVIDIA Ada/Hopper,MI250X 不支持。');
      else addCheck('warn', g.name + ' 走 ROCm 路径,部分特性/版本支持可能滞后。');
    } else if (g.vendor === 'apple') {
      if (!fw.metal) addCheck('fail', g.name + '(Metal)仅 llama.cpp / Transformers 支持。');
    }
  }
  if (new Set(gpus.map(g => g.vendor)).size > 1) addCheck('fail', '不支持混合厂商 GPU 部署。');
  if (fw.linux) addCheck('info', fw.name + ' 需要 Linux 环境(Windows 可用 WSL2)。');
  if (ctx > model.maxCtx) addCheck('warn', '上下文 ' + fmtK(ctx) + ' 超出模型原生上限 ' + fmtK(model.maxCtx) + ',需 RoPE 外推,精度可能下降。');
  if (gpus.length > 1 && fw.tp === 'tensor') {
    if (new Set(gpus.map(g => g.id)).size > 1) addCheck('warn', '张量并行建议使用相同型号 GPU,混合型号会导致算力浪费。');
    if (!allNvlink) addCheck('warn', '多卡之间无 NVLink,张量并行通信走 PCIe 将明显拖慢速度。');
  }

  /* ---------- 部署模式判定 ---------- */
  const fitsAll = requiredB <= totalVramB;
  const gpuAvailForWeights = Math.max(totalVramB - overheadB - kvB, 0);
  const onCpuWeightsB = fitsAll ? 0 : Math.max(weightsB - gpuAvailForWeights, 0);
  let mode; // gpu: 全GPU | offload: GPU+CPU混合 | cpu: 纯CPU | fail
  const memOK = (bytes) => ramTotalB >= bytes * 1.1 + 2 * GB;

  if (fitsAll && gpus.length) {
    mode = 'gpu';
  } else if (fw.cpuOnly || fw.offload) {
    if (!gpus.length) {
      if (memOK(weightsB)) mode = 'cpu';
      else { mode = 'fail'; addCheck('fail', '纯 CPU 部署内存不足:权重需约 ' + gb(weightsB * 1.1 + 2 * GB) + ' 内存,当前仅 ' + gb(ramTotalB) + '。'); }
    } else if (memOK(onCpuWeightsB)) {
      mode = 'offload';
    } else {
      mode = 'fail';
      addCheck('fail', '显存不够且 CPU 内存也不足:需卸载约 ' + gb(onCpuWeightsB) + ' 到内存(建议 ≥ ' + gb(onCpuWeightsB * 1.1 + 2 * GB) + '),当前 ' + gb(ramTotalB) + '。');
    }
  } else {
    mode = 'fail';
  }
  if (mode === 'offload' || mode === 'cpu') {
    if (cpuBwBps <= 0) { mode = 'fail'; addCheck('fail', 'CPU 卸载需要画板上放置 CPU 平台与内存条。'); }
  }

  if (fitsAll && gpus.length) {
    addCheck('ok', '显存满足:需求 ' + gb(requiredB) + ' / 共 ' + gb(totalVramB) + ',余量 ' + gb(totalVramB - requiredB) + '。');
    if (fw.tp === 'tensor' && gpus.length > 1 && minVramB < requiredB / gpus.length) {
      addCheck('warn', '张量并行均分后单卡需 ' + gb(requiredB / gpus.length) + ',最小卡仅 ' + gb(minVramB) + ',可能放不下。');
    }
  } else if (mode === 'fail' && !checks.some(c => c.level === 'fail')) {
    addCheck('fail', '显存不足:需求 ' + gb(requiredB) + ' / 共 ' + gb(totalVramB) + ',缺 ' + gb(requiredB - totalVramB) + '。');
  }

  const offloadFrac = mode === 'cpu' ? 1 : (mode === 'offload' ? onCpuWeightsB / weightsB : 0);
  const isMoe = (+model.experts || 1) > 1;
  // 解码效率:batch=1 下 MoE 小专家 GEMM 带宽利用率低 + 路由开销;TP 逐层 allreduce 延迟
  let decodeEff = fw.decodeEff;
  if (isMoe) decodeEff *= 0.55;
  if (isMoe && fw.tp === 'tensor' && gpus.length > 1) decodeEff *= 0.75;

  /* ---------- 速度估算 ---------- */
  const ctxAvg = Math.max(Math.round(ctx / 2), 64); // 解码按半满上下文估算
  const kvReadB = kvTok * ctxAvg;
  const bytesPerTok = activeB + kvReadB;
  const bytesPerTok0 = activeB + kvTok * 256; // 接近空上下文的最好情况

  const decodeTime = (bytes) => {
    const gpuBytes = bytes * (1 - offloadFrac);
    const cpuBytes = bytes * offloadFrac;
    let t = 0;
    if (gpus.length) t += splitTime(gpuBytes, gpus, fw.tp, allNvlink);
    if (cpuBytes > 0) t += cpuBytes / Math.max(cpuBwBps, 1);
    return t / decodeEff;
  };

  let speed = null;
  const hasFail = checks.some(c => c.level === 'fail');
  const canRun = !hasFail && mode !== 'fail' && (gpus.length > 0 || cpuBwBps > 0);
  if (canRun) {
    const decodeTps = 1 / decodeTime(bytesPerTok);
    const decodeTpsBest = 1 / decodeTime(bytesPerTok0);
    // prefill 是算力受限:FLOPs/token ≈ 2×激活参数 + 注意力二次项
    const flopsPerTok = 2 * p.active + 4 * model.layers * (model.kvHeads || model.heads) * headDim * ctxAvg;
    const gpuTf = gpus.reduce((s, g) => s + g.tf, 0) * (gpus.length > 1 ? (allNvlink ? 1 : 0.85) : 1);
    const gpuFlops = flopsPerTok * (1 - offloadFrac), cpuFlops = flopsPerTok * offloadFrac;
    let tp = 0;
    if (gpuTf > 0 && gpuFlops > 0) tp += gpuFlops / (gpuTf * 1e12 * fw.mfu);
    if (cpuFlops > 0) tp += cpuFlops / Math.max(cpuTf * 1e12 * 0.3, 1);
    const prefillTps = tp > 0 ? 1 / tp : null;
    const ttftS = prefillTps ? promptLen / prefillTps : null;

    const freeForKv = totalVramB - weightsB - overheadB;
    const maxConc = (mode === 'gpu' && freeForKv > 0) ? Math.floor(freeForKv / (kvTok * ctx)) : (mode === 'gpu' ? 0 : null);

    let aggTps = null;
    if (mode === 'gpu' && conc > 1) {
      const stepB = activeB + conc * kvReadB;
      aggTps = conc / (splitTime(stepB, gpus, fw.tp, allNvlink) / decodeEff);
    } else if (conc > 1) {
      aggTps = decodeTps; // 卸载/CPU模式下多路并发收益极小,近似单流
    }

    const sumBw = gpus.reduce((s, g) => s + g.bw, 0);
    speed = {
      decodeTps, decodeTpsBest, prefillTps, ttftS, maxConc, aggTps,
      offloadFrac,
      gpuLayers: model.layers ? Math.round(model.layers * (1 - offloadFrac)) + '/' + model.layers : null,
      bytesPerTok, cpuBwBps, bwUtil: gpus.length ? (bytesPerTok * decodeTps) / (sumBw * GB) : 0,
    };
  }

  /* ---------- 建议 ---------- */
  const suggestions = [];
  if (!fitsAll && gpus.length) {
    for (const q of fw.quants) {
      if (q.id === quant.id) continue;
      const w = p.total * q.bpw / 8;
      const req = w + kvB + fw.fixedOH * GB + fw.fracOH * w;
      if (req <= totalVramB) { suggestions.push('量化改用 ' + q.name + ':总需求降至 ' + gb(req) + ',可完整放进 GPU。'); break; }
    }
    const more = gpus.length ? Math.max(1, Math.ceil(requiredB / Math.max(totalVramB / gpus.length, 1)) - gpus.length) : 0;
    if (more > 0) suggestions.push('再添加 ' + more + ' 块同规格 GPU(' + gpus[0].name + '),或换大显存卡。');
    if (!fw.offload) suggestions.push('换 llama.cpp:支持把放不下的层卸载到 CPU 内存。');
    if (weightsB + overheadB <= totalVramB && kvB > totalVramB - weightsB - overheadB) {
      const fitCtx = Math.max(1, Math.floor((totalVramB - weightsB - overheadB) / (kvTok * conc)));
      suggestions.push('KV 缓存占 ' + gb(kvB) + ':把上下文降到 ~' + fmtK(fitCtx) + ' 或减少并发可塞下。');
    }
  }
  if (mode === 'offload') {
    suggestions.push('约 ' + Math.round(offloadFrac * 100) + '% 权重在 CPU 上,解码速度将主要由内存带宽(~' + (cpuBwBps / GB).toFixed(0) + ' GB/s)决定,比纯 GPU 慢很多。');
  }
  if (mode === 'gpu' && speed && speed.maxConc !== null && speed.maxConc < conc) {
    suggestions.push('当前显存最多支持 ' + speed.maxConc + ' 路 ' + fmtK(ctx) + ' 并发,超出部分会 OOM,建议加卡或降并发。');
  }
  if (hasFail && suggestions.length === 0 && !fitsAll) suggestions.push('从左侧硬件库添加 GPU 到画板后再评估。');

  const verdict = hasFail ? 'fail' : (mode === 'gpu' ? 'ok' : 'warn');
  const firstFail = checks.find(c => c.level === 'fail');

  return {
    verdict,
    bannerText: verdict === 'fail' ? '❌ 无法部署 — ' + (firstFail ? firstFail.msg : '存在不满足的条件')
      : verdict === 'warn' ? (mode === 'cpu' ? '⚠️ 可以部署 — 纯 CPU 模式,速度受限' : '⚠️ 可以部署 — 需 CPU 卸载,速度有折损')
      : '✅ 可以部署 — 权重与 KV 缓存可完整放进显存',
    mode, checks, suggestions, speed,
    mem: {
      weightsB, kvB, overheadB, requiredB, totalVramB, kvTok,
      paramsTotal: p.total, paramsActive: p.active,
    },
  };
}
