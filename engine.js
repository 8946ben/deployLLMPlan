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
function gb2(x) { const v = x / GB; return (v >= 100 ? v.toFixed(1) : v.toFixed(2)) + ' GB'; }
function kb(x) { return (x / 1024).toFixed(1).replace(/\.0$/, '') + ' KB'; }
function ms(t) { return t >= 1 ? t.toFixed(2) + ' s' : (t * 1000).toFixed(1) + ' ms'; }
function tpsF(v) { return (v >= 100 ? v.toFixed(0) : v.toFixed(2)) + ' tok/s'; }

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
  return { total, active, headDim: d,
    parts: { attn: attn * L, mlp: mlpTotal * L, router: router * L, embed, lmHead, extra: Math.round(+m.extraParams || 0) } };
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
   tensor(张量并行):各卡并行读各自分片,取最慢;通信开销由 tpCommTime 按互联带宽单独计入。
   pipeline(层切分,llama.cpp式):各卡按显存比例分担,逐卡串行。 */
function splitTime(bytes, gpus, tpMode) {
  if (!gpus.length || bytes <= 0) return 0;
  if (tpMode === 'tensor' && gpus.length > 1) {
    const share = bytes / gpus.length;
    let t = 0;
    for (const g of gpus) t = Math.max(t, share / (g.bw * GB));
    return t;
  }
  const totalVram = gpus.reduce((s, g) => s + g.vram, 0);
  let t = 0;
  for (const g of gpus) t += (bytes * (g.vram / totalVram)) / (g.bw * GB);
  return t;
}

/* 互联推断:同代 NVLink 的 NVIDIA 卡 → 对应 NVLink;AMD 全 xGMI → Infinity Fabric;
   其余(消费卡/PCIe版/混合)→ PCIe 取最低代。可被用户显式选择覆盖。 */
function resolveInterconnect(gpus, icId) {
  if (icId && icId !== 'auto') return INTERCONNECTS.find(i => i.id === icId) || null;
  if (!gpus || !gpus.length) return null;
  const vendors = new Set(gpus.map(g => g.vendor));
  const pcieOf = (list) => {
    const gen = Math.min(...list.map(g => g.pcieGen || 3));
    return INTERCONNECTS.find(i => i.id === (gen >= 5 ? 'pcie5' : gen >= 4 ? 'pcie4' : 'pcie3'));
  };
  if (vendors.has('apple')) return INTERCONNECTS.find(i => i.id === 'unified');
  if (vendors.has('fpga')) return INTERCONNECTS.find(i => i.id === 'eth1g'); // FPGA 开发板经千兆网接主机(与 LLM_FPGA 工程实测链路一致)
  if (vendors.size > 1) return pcieOf(gpus);
  if (vendors.has('nvidia')) {
    const gen = Math.min(...gpus.map(g => g.nvGen || 0));
    if (gpus.every(g => g.nvlink) && gen >= 2) {
      return INTERCONNECTS.find(i => i.id === 'nvlink' + gen) || INTERCONNECTS.find(i => i.id === 'nvlink4');
    }
    return pcieOf(gpus);
  }
  if (vendors.has('amd') && gpus.every(g => g.xgmi)) return INTERCONNECTS.find(i => i.id === 'xgmi');
  return pcieOf(gpus);
}

/* 张量并行每 token 的 allreduce 通信耗时 = 带宽项(环形 allreduce 2(N-1)/N) + 延迟项(每层2次) */
function tpCommTime(ic, model, nGpus) {
  if (!ic || !ic.bw || nGpus < 2 || model.hidden < 2 || model.layers < 2) return 0;
  const bytesAR = 2 * model.layers * model.hidden * 2 * 2 * (nGpus - 1) / nGpus;
  return bytesAR / (ic.bw * GB) + 2 * model.layers * (ic.lat || 10) * 1e-6;
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
  const ic = cfg.ic || resolveInterconnect(gpus);
  const tpCommS = fw.tp === 'tensor' && gpus.length > 1 && ic ? tpCommTime(ic, model, gpus.length) : 0;

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
  if (gpus.some(g => g.vendor === 'fpga')) {
    if (!fw.fpga) addCheck('fail', 'FPGA 推理走自定义 HLS 运行时(如 FlightLLM 类),' + fw.name + ' 不直接支持;估算请改用 llama.cpp / HF Transformers 作带宽受限代理。');
    else addCheck('warn', 'FPGA 为自定义 HLS 运行时估算:解码按板载 DDR4 带宽、预填充按 INT8 等效算力(当前串行核未并行化),均为上界参考。');
  }
  if (fw.linux) addCheck('info', fw.name + ' 需要 Linux 环境(Windows 可用 WSL2)。');
  if (ctx > model.maxCtx) addCheck('warn', '上下文 ' + fmtK(ctx) + ' 超出模型原生上限 ' + fmtK(model.maxCtx) + ',需 RoPE 外推,精度可能下降。');
  if (gpus.length > 1 && fw.tp === 'tensor') {
    const icText = ic ? (ic.bw ? ic.name + ' · ' + ic.bw + ' GB/s' : ic.name) : '';
    if (new Set(gpus.map(g => g.id)).size > 1) addCheck('warn', '张量并行建议使用相同型号 GPU,混合型号会导致算力浪费。');
    if (ic && ic.type === 'fabric') addCheck('ok', '互联 ' + icText + ':TP allreduce 开销 ≈' + (tpCommS * 1e6).toFixed(0) + 'µs/token,可接受。');
    else if (ic && ic.bw > 0) addCheck('warn', '张量并行走 ' + icText + ':TP 通信开销 ≈' + (tpCommS * 1e6).toFixed(0) + 'µs/token(含延迟项),批量小/延迟敏感场景吞吐明显受限,建议 NVLink 机型或减小 TP 度。');
  } else if (gpus.length > 1 && ic && fw.tp === 'pipeline') {
    addCheck('info', '层切分模式:卡间仅传递激活向量,' + (ic.bw ? ic.name + '(' + ic.bw + ' GB/s)' : ic.name) + ' 完全足够,多卡主要用于扩容而非提速。');
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
  const kvLayersN = +model.kvAttnLayers || +model.layers;
  const kvHeadsN = +model.kvHeads || +model.heads;
  // 解码效率:batch=1 下 MoE 小专家 GEMM 带宽利用率低 + 路由开销;多卡 MoE 还有 EP 分发/all-to-all 额外开销
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
    if (gpus.length) t += splitTime(gpuBytes, gpus, fw.tp);
    if (cpuBytes > 0) t += cpuBytes / Math.max(cpuBwBps, 1);
    if (tpCommS > 0) t += tpCommS;
    return t / decodeEff;
  };

  let speed = null;
  let speedDetails = null;
  const hasFail = checks.some(c => c.level === 'fail');
  const canRun = !hasFail && mode !== 'fail' && (gpus.length > 0 || cpuBwBps > 0);
  if (canRun) {
    const decodeTps = 1 / decodeTime(bytesPerTok);
    const decodeTpsBest = 1 / decodeTime(bytesPerTok0);
    // prefill 是算力受限:FLOPs/token ≈ 2×激活参数 + 注意力二次项
    const flopsPerTok = 2 * p.active + 4 * model.layers * (model.kvHeads || model.heads) * headDim * ctxAvg;
    // 张量并行各卡同时算同一层 → 算力累加;层切分(llama.cpp)逐层串行 → 吞吐受最慢单卡限制
    const gpuTf = fw.tp === 'pipeline' && gpus.length > 1
      ? Math.max(...gpus.map(g => g.tf || 0))
      : gpus.reduce((s, g) => s + g.tf, 0) * (gpus.length > 1 ? (ic && ic.type === 'fabric' ? 1 : 0.85) : 1);
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
      aggTps = conc / ((splitTime(stepB, gpus, fw.tp) + tpCommS) / decodeEff);
    } else if (conc > 1) {
      aggTps = decodeTps; // 卸载/CPU模式下多路并发收益极小,近似单流
    }

    const sumBw = gpus.reduce((s, g) => s + g.bw, 0);
    speed = {
      decodeTps, decodeTpsBest, prefillTps, ttftS, maxConc, aggTps,
      offloadFrac, tpCommMs: tpCommS * 1000, icName: ic ? ic.name : null, icBw: ic ? ic.bw : 0,
      gpuLayers: model.layers ? Math.round(model.layers * (1 - offloadFrac)) + '/' + model.layers : null,
      bytesPerTok, cpuBwBps, bwUtil: gpus.length ? (bytesPerTok * decodeTps) / (sumBw * GB) : 0,
    };

    /* ---------- 速度各项的计算细节 ---------- */
    const gpuPart = bytesPerTok * (1 - offloadFrac), cpuPart = bytesPerTok * offloadFrac;
    const timeParts = [];
    if (gpus.length && gpuPart > 0) timeParts.push('GPU 读 ' + gb2(gpuPart) + ' ≈ ' + ms(splitTime(gpuPart, gpus, fw.tp)));
    if (cpuPart > 0 && cpuBwBps > 0) timeParts.push('CPU 读 ' + gb2(cpuPart) + ' ÷ 内存带宽 ' + (cpuBwBps / GB).toFixed(0) + ' GB/s ≈ ' + ms(cpuPart / cpuBwBps));
    if (tpCommS > 0) {
      const arMb = 2 * model.layers * model.hidden * 2 * 2 * (gpus.length - 1) / gpus.length / 1e6;
      timeParts.push('TP 通信 ≈ ' + ms(tpCommS) + '(' + ic.name + ' ' + ic.bw + ' GB/s:每 token allreduce ' + arMb.toFixed(2) + ' MB ÷ 带宽 + ' + (2 * model.layers) + ' 次 × ' + ic.lat + 'µs 延迟)');
    }
    const tpDesc = gpus.length > 1
      ? (fw.tp === 'tensor'
        ? '张量并行:每卡读 1/' + gpus.length + ' 分片,耗时取最慢卡;通信开销按互联带宽单独计入'
        : '层切分:各卡按显存比例分担权重,串行读取耗时累加(多卡主要扩容,提速有限)')
      : '';
    const effDesc = '基础 ' + fw.decodeEff + (isMoe ? ' × 0.55(MoE batch=1 小专家带宽利用率低)' : '') + (isMoe && fw.tp === 'tensor' && gpus.length > 1 ? ' × 0.75(EP 路由/all-to-all)' : '');
    speedDetails = {
      decode: [
        '每 token 读取 = 激活权重 ' + gb2(activeB) + '(' + fmtB(p.active) + ' 激活参数 × ' + quant.bpw + 'bit ÷ 8) + KV ' + kb(kvTok) + '/token × ' + fmtK(ctxAvg) + '(按半满上下文估) = ' + gb2(bytesPerTok),
        ...(tpDesc ? [tpDesc] : []),
        ...(timeParts.length ? ['解码时间 = ' + timeParts.join(' + ') + ',再 ÷ 效率系数(' + effDesc + ') ≈ ' + ms(decodeTime(bytesPerTok))] : []),
        '解码速度 = 1 ÷ 时间 ≈ ' + tpsF(decodeTps),
      ],
      best: ['最好情况(接近空上下文):KV 读取按 256 token 计 → 每 token 读取 ' + gb2(bytesPerTok0) + ' → 速度上限 ≈ ' + tpsF(decodeTpsBest)],
      prefill: (() => {
        const lines = ['每 token 计算量 ≈ 2 × 激活参数 ' + fmtB(p.active) + ' + 注意力二次项 4 × ' + model.layers + ' 层 × ' + kvHeadsN + ' KV头 × ' + headDim + ' 维 × ' + fmtK(ctxAvg) + '(半满上下文) ≈ ' + (flopsPerTok / 1e12).toFixed(2) + ' TFLOP'];
        if (gpuTf > 0 && gpuFlops > 0) lines.push('GPU 部分 ÷ (' + gpuTf.toFixed(0) + ' TFLOPS 峰值 × MFU ' + fw.mfu + ') ≈ ' + ms(gpuFlops / (gpuTf * 1e12 * fw.mfu)));
        if (cpuFlops > 0) lines.push('CPU 部分 ÷ (' + cpuTf.toFixed(0) + ' TFLOPS × 利用率 0.30) ≈ ' + ms(cpuFlops / Math.max(cpuTf * 1e12 * 0.3, 1)));
        lines.push('预填充速度 ≈ ' + tpsF(prefillTps) + '(算力受限,与解码的带宽受限不同)');
        return lines;
      })(),
      ttft: ['TTFT = 提示长度 ' + fmtK(promptLen) + ' tokens ÷ 预填充速度 ' + tpsF(prefillTps) + ' ≈ ' + (ttftS >= 1 ? ttftS.toFixed(2) + ' s' : (ttftS * 1000).toFixed(0) + ' ms')],
      maxConc: mode === 'gpu'
        ? ['最大并发 = (显存 ' + gb(totalVramB) + ' − 权重 ' + gb2(weightsB) + ' − 开销 ' + gb2(overheadB) + ') = 可用 ' + gb2(Math.max(freeForKv, 0)),
           '每路 KV = ' + kb(kvTok) + ' × ' + fmtK(ctx) + ' = ' + gb2(kvTok * ctx) + ' → 可容纳 ' + (maxConc > 0 ? maxConc + ' 路' : '0 路(剩余空间放不下一条完整上下文)')]
        : ['卸载/纯 CPU 模式不估算最大并发:容量由内存决定,多路并发共享带宽,总吞吐近似单流'],
      agg: conc > 1
        ? (mode === 'gpu'
          ? ['每步读取 = 激活权重 ' + gb2(activeB) + '(各路共享) + ' + conc + ' 路 × KV 读取 ' + gb2(kvReadB) + ' = ' + gb2(activeB + conc * kvReadB),
             '总吞吐 = ' + conc + ' 路 ÷ 单步时间 ≈ ' + tpsF(aggTps)]
          : ['卸载/CPU 模式下多路并发共享内存带宽,收益极小:总吞吐 ≈ 单流速度'])
        : ['并发 = 1:总吞吐 = 单流解码速度'],
      layers: mode === 'offload'
        ? ['显存可容纳权重 = 显存 ' + gb(totalVramB) + ' − 开销 ' + gb2(overheadB) + ' − KV ' + gb2(kvB) + ' = ' + gb2(gpuAvailForWeights),
           '权重共 ' + gb2(weightsB) + ' → 约 ' + Math.round(offloadFrac * 100) + '%(' + gb2(onCpuWeightsB) + ')放在 CPU 内存']
        : mode === 'cpu' ? ['纯 CPU 模式:100% 权重放在内存,解码速度由内存带宽决定'] : ['全部权重放进 GPU 显存,无需卸载'],
      bneck: ['带宽利用率 = 每 token 读取 ' + gb2(bytesPerTok) + ' × 解码 ' + tpsF(decodeTps) + ' ÷ 总带宽 ' + sumBw + ' GB/s ≈ ' + Math.round((bytesPerTok * decodeTps) / (sumBw * GB) * 100) + '%',
        mode === 'gpu' ? '解码阶段是显存带宽受限(读权重+KV),算力利用率低是正常现象' : '瓶颈在内存带宽,远低于显存带宽'],
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

  /* ---------- 显存/判定各项的计算细节 ---------- */
  const gpuGroup = {};
  for (const g of gpus) { const k = g.name; (gpuGroup[k] = gpuGroup[k] || { n: 0, vram: 0 }); gpuGroup[k].n++; gpuGroup[k].vram += g.vram; }
  const vramDesc = gpus.length
    ? Object.keys(gpuGroup).map(k => k + ' ×' + gpuGroup[k].n + '(' + gpuGroup[k].vram + 'GB)').join(' + ') + ' = ' + gb(totalVramB)
    : '画板上没有 GPU';
  const pt = p.parts;
  const paramLine = '参数量 = 每层(注意力 ' + fmtB(pt.attn) + ' + MLP ' + (isMoe ? model.experts + '专家 ' : '') + fmtB(pt.mlp) + (pt.router ? ' + 路由 ' + fmtB(pt.router) : '') + ') × ' + model.layers +
    ' 层 + 词嵌入 ' + fmtB(pt.embed) + (model.tied ? '(LM头共享)' : ' + LM头 ' + fmtB(pt.lmHead)) + (pt.extra ? ' + 附加参数 ' + fmtB(pt.extra) : '') + ' ≈ ' + fmtB(p.total);
  const details = {
    weights: [
      paramLine,
      '权重体积 = ' + fmtB(p.total) + ' 参数 × ' + quant.bpw + ' bit ÷ 8(' + quant.name + ') = ' + gb2(weightsB),
      ...(isMoe ? ['MoE:每 token 只读激活参数 ' + fmtB(p.active) + ' 对应的权重(决定解码速度),但全部专家都要放进显存'] : []),
    ],
    kv: +model.kvBytesPerTok > 0
      ? ['每 token KV = 覆盖值 ' + model.kvBytesPerTok + ' B(FP16 基准,MLA/压缩注意力架构) × ' + kvQuant.bpw + ' ÷ 16(' + kvQuant.name + ') = ' + kb(kvTok),
         '总量 = ' + kb(kvTok) + ' × ' + fmtK(ctx) + ' 上下文 × ' + conc + ' 路并发 = ' + gb2(kvB)]
      : ['每 token KV = 2(K+V) × ' + kvLayersN + ' 个 KV 层 × ' + kvHeadsN + ' 个 KV 头 × ' + headDim + ' 头维度 × ' + (kvQuant.bpw / 8) + ' 字节(' + kvQuant.name + ') = ' + kb(kvTok),
         ...(kvLayersN < +model.layers ? ['混合注意力:' + model.layers + ' 层中仅 ' + kvLayersN + ' 层全注意力产生逐 token KV,其余线性注意力层状态恒定'] : []),
         '总量 = ' + kb(kvTok) + ' × ' + fmtK(ctx) + ' 上下文 × ' + conc + ' 路并发 = ' + gb2(kvB)],
    overhead: ['框架开销 = ' + fw.name + ' 固定 ' + fw.fixedOH + ' GB + 权重 ' + gb2(weightsB) + ' × ' + Math.round(fw.fracOH * 100) + '% = ' + gb2(overheadB) + '(激活值/运行时/显存碎片)'],
    required: ['合计 = 权重 ' + gb2(weightsB) + ' + KV ' + gb2(kvB) + ' + 开销 ' + gb2(overheadB) + ' = ' + gb2(requiredB)],
    vram: ['显存总量 = ' + vramDesc],
    surplus: fitsAll
      ? ['剩余 = 显存 ' + gb(totalVramB) + ' − 需求 ' + gb2(requiredB) + ' = ' + gb(totalVramB - requiredB)]
      : ['缺口 = 需求 ' + gb2(requiredB) + ' − 显存 ' + gb(totalVramB) + ' = ' + gb(requiredB - totalVramB)],
    verdict: (() => {
      const lines = ['总需求 = 权重 ' + gb2(weightsB) + ' + KV ' + gb2(kvB) + ' + 开销 ' + gb2(overheadB) + ' = ' + gb2(requiredB) + ';显存总量 ' + gb(totalVramB)];
      if (fitsAll && gpus.length) lines.push('需求 ≤ 显存 → 可完整放进 GPU,无需卸载');
      else if (!gpus.length) lines.push('画板无 GPU:' + (fw.cpuOnly ? '走纯 CPU 路线,权重需放进内存' : fw.name + ' 必须运行在 GPU 上 → 判定失败'));
      else {
        lines.push('需求 > 显存,差 ' + gb(requiredB - totalVramB));
        if (fw.offload) lines.push('框架支持 CPU 卸载:显存可容纳权重 ' + gb2(gpuAvailForWeights) + '(= 显存 − 开销 − KV),需卸载 ' + gb2(onCpuWeightsB) + ' 到内存;内存判定 = 卸载量 × 1.1 + 2 GB ≈ ' + gb2(onCpuWeightsB * 1.1 + 2 * GB) + ',当前内存 ' + gb(ramTotalB) + (memOK(onCpuWeightsB) ? ' → 够用' : ' → 也不足'));
        else lines.push(fw.name + ' 不支持把权重卸载到 CPU → 无法部署');
      }
      if (mode === 'cpu') lines.push('纯 CPU 模式:权重全部放内存,解码受内存带宽限制');
      lines.push('结论:' + (verdict === 'ok' ? '✅ 可以部署' : verdict === 'warn' ? '⚠️ 可以部署(有折损)' : '❌ 无法部署(全部原因见检查清单)'));
      return lines;
    })(),
    speed: speedDetails,
  };

  return {
    verdict,
    bannerText: verdict === 'fail' ? '❌ 无法部署 — ' + (firstFail ? firstFail.msg : '存在不满足的条件')
      : verdict === 'warn' ? (mode === 'cpu' ? '⚠️ 可以部署 — 纯 CPU 模式,速度受限' : '⚠️ 可以部署 — 需 CPU 卸载,速度有折损')
      : '✅ 可以部署 — 权重与 KV 缓存可完整放进显存',
    mode, checks, suggestions, speed, details,
    mem: {
      weightsB, kvB, overheadB, requiredB, totalVramB, kvTok,
      paramsTotal: p.total, paramsActive: p.active,
    },
  };
}
