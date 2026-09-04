'use strict';
/* 数据目录:模型预设 / 推理框架 / GPU / CPU / 内存
   规格 sourced 自公开资料,均为规划用近似值。 */

const MODELS = [
  /* Qwen3.8-27B: 官方 config.json 实测规格。混合注意力:64层中仅16层全注意力(产生KV缓存),
     其余48层为线性注意力(常数状态);估算语言主体 ≈24.4B,官方27B含视觉塔与MTP头。 */
  { id: 'qwen3.8-27b',      name: 'Qwen3.8-27B',            layers: 64, hidden: 5120, heads: 24, kvHeads: 4, headDim: 256, kvAttnLayers: 16, inter: 17408, vocab: 248320, tied: false, maxCtx: 262144 },
  /* 2026 新架构 MoE,按官方 config.json 录入。MLA/KDA 等压缩注意力用 attnPerLayer / kvBytesPerTok
     覆盖近似;extraParams 计入权重但不参与每 token 读取(n-gram 嵌入表等)。 */
  { id: 'dsv4-flash',       name: 'DeepSeek-V4-Flash',      layers: 43, hidden: 4096, heads: 64, kvHeads: 1, headDim: 512, inter: 2048, experts: 257, expertsActive: 7, vocab: 129280, tied: false, maxCtx: 1048576, attnPerLayer: 1.18e8, kvBytesPerTok: 49536 },
  { id: 'qwen3.8-flash-next', name: 'Qwen3.8-Flash-Next',   layers: 48, hidden: 2560, heads: 24, kvHeads: 2, headDim: 256, kvAttnLayers: 12, inter: 640, experts: 513, expertsActive: 11, vocab: 248320, tied: false, maxCtx: 262144, extraParams: 5.12e10 },
  { id: 'glm-5.3-flash',    name: 'GLM-5.3-Flash',          layers: 45, hidden: 4096, heads: 64, kvHeads: 64, headDim: 256, kvAttnLayers: 11, inter: 2048, experts: 289, expertsActive: 9, vocab: 154880, tied: false, maxCtx: 1048576, attnPerLayer: 1.2e8, kvBytesPerTok: 11264 },
  { id: 'qwen3-32b',        name: 'Qwen3-32B',              layers: 64, hidden: 5120, heads: 40, kvHeads: 8, inter: 27648, vocab: 151936, tied: false, maxCtx: 40960 },
  { id: 'qwen3-30b-a3b',    name: 'Qwen3-30B-A3B (MoE)',    layers: 48, hidden: 2048, heads: 16, kvHeads: 8, headDim: 128, inter: 768, experts: 128, expertsActive: 8, vocab: 151936, tied: true, maxCtx: 40960 },
  { id: 'qwen2.5-32b',      name: 'Qwen2.5-32B-Instruct',   layers: 64, hidden: 5120, heads: 40, kvHeads: 8, inter: 27648, vocab: 152064, tied: false, maxCtx: 32768 },
  { id: 'qwen2.5-0.5b',     name: 'Qwen2.5-0.5B-Instruct',  layers: 24, hidden: 896,  heads: 14, kvHeads: 2, inter: 4864,  vocab: 151936, tied: true,  maxCtx: 32768 },
  { id: 'qwen2.5-7b',       name: 'Qwen2.5-7B-Instruct',    layers: 28, hidden: 3584, heads: 28, kvHeads: 4, inter: 18944, vocab: 152064, tied: true,  maxCtx: 32768 },
  { id: 'qwen2.5-72b',      name: 'Qwen2.5-72B-Instruct',   layers: 80, hidden: 8192, heads: 64, kvHeads: 8, inter: 29696, vocab: 152064, tied: false, maxCtx: 32768 },
  { id: 'llama-3.1-8b',     name: 'Llama-3.1-8B-Instruct',  layers: 32, hidden: 4096, heads: 32, kvHeads: 8, inter: 14336, vocab: 128256, tied: false, maxCtx: 131072 },
  { id: 'llama-3.1-70b',    name: 'Llama-3.1-70B-Instruct', layers: 80, hidden: 8192, heads: 64, kvHeads: 8, inter: 28672, vocab: 128256, tied: false, maxCtx: 131072 },
];

/* GPU: vram(GB) bw(GB/s,显存带宽) cc(CUDA计算能力,nvidia专有) tf(FP16 Tensor TFLOPS) nvlink fp8(支持FP8张量核心)
   nvGen(NVLink代数,卡间聚合带宽见 INTERCONNECTS) pcieGen(PCIe代数) xgmi(AMD Infinity Fabric) */
const GPUS = [
  { id: 'v100-16',     name: 'NVIDIA V100-SXM2 16GB', vram: 16,  bw: 900,  cc: 7.0,  tf: 112, nvlink: true,  fp8: false, vendor: 'nvidia', nvGen: 2, pcieGen: 3, note: 'Volta · 无BF16/FP8' },
  { id: 'v100-32',     name: 'NVIDIA V100-SXM2 32GB', vram: 32,  bw: 900,  cc: 7.0,  tf: 112, nvlink: true,  fp8: false, vendor: 'nvidia', nvGen: 2, pcieGen: 3, note: 'Volta · 无BF16/FP8' },
  { id: 'v100s-pcie',  name: 'NVIDIA Tesla V100S-PCIE 32GB', vram: 32, bw: 900, cc: 7.0, tf: 112, nvlink: false, fp8: false, vendor: 'nvidia', pcieGen: 3, note: 'Volta · 32GB PCIe版' },
  { id: 't4',          name: 'NVIDIA Tesla T4 16GB',  vram: 16,  bw: 320,  cc: 7.5,  tf: 65,  nvlink: false, fp8: false, vendor: 'nvidia', pcieGen: 3, note: '低功耗推理卡' },
  { id: 'rtx3090',     name: 'NVIDIA RTX 3090 24GB',  vram: 24,  bw: 936,  cc: 8.6,  tf: 71,  nvlink: false, fp8: false, vendor: 'nvidia', pcieGen: 4, note: 'Ampere 消费级' },
  { id: 'a10',         name: 'NVIDIA A10 24GB',       vram: 24,  bw: 600,  cc: 8.6,  tf: 125, nvlink: false, fp8: false, vendor: 'nvidia', pcieGen: 4, note: 'Ampere 数据中心' },
  { id: 'rtx4090',     name: 'NVIDIA RTX 4090 24GB',  vram: 24,  bw: 1008, cc: 8.9,  tf: 165, nvlink: false, fp8: true,  vendor: 'nvidia', pcieGen: 4, note: 'Ada · 支持FP8' },
  { id: 'rtx5090',     name: 'NVIDIA RTX 5090 32GB',  vram: 32,  bw: 1792, cc: 12.0, tf: 209, nvlink: false, fp8: true,  vendor: 'nvidia', pcieGen: 5, note: 'Blackwell · 需新软件栈' },
  { id: 'l40s',        name: 'NVIDIA L40S 48GB',      vram: 48,  bw: 864,  cc: 8.9,  tf: 181, nvlink: false, fp8: true,  vendor: 'nvidia', pcieGen: 4, note: 'Ada · 大显存推理卡' },
  { id: 'a100-40',     name: 'NVIDIA A100-SXM4 40GB', vram: 40,  bw: 1555, cc: 8.0,  tf: 312, nvlink: true,  fp8: false, vendor: 'nvidia', nvGen: 3, pcieGen: 4, note: 'Ampere · 原生BF16' },
  { id: 'a100-80',     name: 'NVIDIA A100-SXM4 80GB', vram: 80,  bw: 2039, cc: 8.0,  tf: 312, nvlink: true,  fp8: false, vendor: 'nvidia', nvGen: 3, pcieGen: 4, note: 'Ampere · 原生BF16' },
  { id: 'a100-80-pcie',name: 'NVIDIA A100-PCIe 80GB', vram: 80,  bw: 1935, cc: 8.0,  tf: 312, nvlink: false, fp8: false, vendor: 'nvidia', nvGen: 3, pcieGen: 4, note: 'PCIe版 · 卡间无NVLink' },
  { id: 'h100',        name: 'NVIDIA H100-SXM5 80GB', vram: 80,  bw: 3350, cc: 9.0,  tf: 990, nvlink: true,  fp8: true,  vendor: 'nvidia', nvGen: 4, pcieGen: 5, note: 'Hopper · FP8 · NVSwitch' },
  { id: 'h100-pcie',   name: 'NVIDIA H100-PCIe 80GB', vram: 80,  bw: 2000, cc: 9.0,  tf: 756, nvlink: false, fp8: true,  vendor: 'nvidia', nvGen: 4, pcieGen: 5, note: 'Hopper PCIe版' },
  { id: 'h200',        name: 'NVIDIA H200 SXM 141GB', vram: 141, bw: 4800, cc: 9.0,  tf: 990, nvlink: true,  fp8: true,  vendor: 'nvidia', nvGen: 4, pcieGen: 5, note: 'Hopper · 141GB HBM3e' },
  { id: 'b200',        name: 'NVIDIA B200 SXM 180GB', vram: 180, bw: 8000, cc: 10.0, tf: 2250, nvlink: true,  fp8: true,  vendor: 'nvidia', nvGen: 5, pcieGen: 5, note: 'Blackwell · FP8/FP4 · NVSwitch' },
  { id: 'b300',        name: 'NVIDIA B300 (Blackwell Ultra) 288GB', vram: 288, bw: 8000, cc: 10.0, tf: 2250, nvlink: true, fp8: true, vendor: 'nvidia', nvGen: 5, pcieGen: 5, note: 'Blackwell Ultra · 288GB HBM3e' },
  { id: 'rtxpro6000',  name: 'NVIDIA RTX PRO 6000 96GB', vram: 96, bw: 1600, cc: 12.0, tf: 500, nvlink: false, fp8: true,  vendor: 'nvidia', pcieGen: 5, note: 'Blackwell 工作站 · 96GB GDDR7' },
  { id: 'mi250x',      name: 'AMD MI250X 64GB',       vram: 64,  bw: 1638, cc: null, tf: 383, nvlink: true,  fp8: false, vendor: 'amd',    xgmi: true, pcieGen: 4, note: 'CDNA2 · ROCm · xGMI' },
  { id: 'm2ultra',     name: 'Apple M2 Ultra 192GB',  vram: 192, bw: 819,  cc: null, tf: 27,  nvlink: false, fp8: false, vendor: 'apple', note: '统一内存 · 仅llama.cpp' },
  { id: 'dspark',      name: 'NVIDIA DGX Spark 128GB', vram: 128, bw: 273, cc: 12.0, tf: 125, nvlink: false, fp8: true,  vendor: 'nvidia', pcieGen: 5, note: 'GB10 统一内存 · 桌面级' },
];

/* 互联方式: bw = 卡间有效带宽 GB/s(单向),lat = 每次 allreduce 近似延迟 µs
   fabric: NVLink/NVSwitch 类高速域内互联;pcie: 走主机 PCIe;net: 跨节点网络 */
const INTERCONNECTS = [
  { id: 'auto',    name: '自动(按GPU推断)',      bw: 0,    lat: 0,  type: 'auto' },
  { id: 'nvlink2', name: 'NVLink 2.0 (V100)',   bw: 300,  lat: 6,  type: 'fabric' },
  { id: 'nvlink3', name: 'NVLink 3.0 (A100)',   bw: 600,  lat: 5,  type: 'fabric' },
  { id: 'nvlink4', name: 'NVLink 4.0 (H100)',   bw: 900,  lat: 5,  type: 'fabric' },
  { id: 'nvlink5', name: 'NVLink 5.0 (B200)',   bw: 1800, lat: 4,  type: 'fabric' },
  { id: 'xgmi',    name: 'xGMI/Infinity Fabric', bw: 800,  lat: 6,  type: 'fabric' },
  { id: 'pcie3',   name: 'PCIe 3.0 x16',        bw: 16,   lat: 25, type: 'pcie' },
  { id: 'pcie4',   name: 'PCIe 4.0 x16',        bw: 32,   lat: 20, type: 'pcie' },
  { id: 'pcie5',   name: 'PCIe 5.0 x16',        bw: 64,   lat: 15, type: 'pcie' },
  { id: 'ib400',   name: 'InfiniBand NDR 400G', bw: 50,   lat: 10, type: 'net' },
  { id: 'eth100',  name: 'Ethernet 100GbE',     bw: 12.5, lat: 30, type: 'net' },
  { id: 'unified', name: '统一内存(片上)',       bw: 0,    lat: 0,  type: 'unified' },
];

/* CPU平台: cores(核心数) channels(内存通道数,每通道DDR5-4800约38.4GB/s) */
const CPUS = [
  { id: 'epyc-9654', name: 'AMD EPYC 9654 (96核)',  cores: 96, channels: 12, note: '12通道 DDR5 · 服务器' },
  { id: 'xeon-8480', name: 'Intel Xeon 8480+ (56核)', cores: 56, channels: 8, note: '8通道 DDR5 · 服务器' },
  { id: 'ryzen',     name: 'Ryzen 9 7950X (16核)',   cores: 16, channels: 2, note: '双通道 DDR5 · 消费级' },
  { id: 'xeon-e5v4', name: 'Intel Xeon E5-2698 v4 ×2 (40核)', cores: 40, channels: 12, note: 'DGX-1 原装 · 双路 Broadwell · DDR4' },
  { id: 'xeon-gold-6326', name: 'Intel Xeon Gold 6326 ×2 (32核)', cores: 32, channels: 16, note: '思腾合力原装 · 双路 Ice Lake · DDR4-3200' },
];

const RAMS = [
  { id: 'ddr5-32', name: 'DDR5-4800 32GB',  gb: 32, bw: 38.4, note: '单条 · 单通道带宽38.4GB/s' },
  { id: 'ddr5-64', name: 'DDR5-4800 64GB',  gb: 64, bw: 38.4, note: '单条 · RDIMM' },
];

/* 整机:预配置一体机,添加到画板时自动展开为组成部件(GPU/CPU/内存)。
   两台成员配置均已 SSH 实测核实(nvidia-smi / lscpu / free,2026-09-05) */
const APPLIANCES = [
  {
    id: 'dgx-1', name: 'NVIDIA DGX-1 · 8×V100-SXM2 256GB', vendor: 'nvidia',
    spec: { gpus: { 'v100-32': 8 }, cpus: { 'xeon-e5v4': 1 }, rams: { 'ddr5-64': 8 } },
    note: '实测:8×V100-SXM2-32GB · 双路E5-2698 v4 · 512GB · NVLink2 全互联(ssh 124.16.70.19:7626)',
  },
  {
    id: 'siton-gpu', name: '思腾合力 GPU 服务器 · 3×V100S-PCIE 96GB', vendor: 'nvidia',
    spec: { gpus: { 'v100s-pcie': 3 }, cpus: { 'xeon-gold-6326': 1 }, rams: { 'ddr5-64': 4 } },
    note: '实测:3×V100S-PCIE-32GB · 双路Gold 6326 · 256GB · 无NVLink桥接,走PCIe(ssh 124.16.71.7:7626)',
  },
];

/* 推理框架: minCC(最低CUDA计算能力) linux(需Linux/WSL2) tp(多卡策略 tensor|pipeline)
   offload(支持权重部分卸载到CPU) cpuOnly(支持纯CPU) decodeEff(解码带宽效率) mfu(prefill算力利用率)
   fixedOH/fracOH: 框架显存开销 = fixedOH GB + fracOH×权重 */
const Q_COMMON_AWQ = { id: 'awq',  name: 'AWQ INT4 (W4A16)', bpw: 4.5 };
const Q_COMMON_GPTQ = { id: 'gptq', name: 'GPTQ INT4 (W4A16)', bpw: 4.5 };
const Q_COMMON_FP8 = { id: 'fp8', name: 'FP8 (E4M3)', bpw: 8, needCC: 8.9, fp8: true };

const FRAMEWORKS = [
  {
    id: 'vllm', name: 'vLLM', tag: '高吞吐服务引擎 · PagedAttention · 连续批处理',
    minCC: 7.0, linux: true, tp: 'tensor', offload: false, cpuOnly: false, rocm: true, metal: false,
    quants: [
      { id: 'fp16', name: 'FP16 (2字节)', bpw: 16 },
      { id: 'bf16', name: 'BF16 (2字节)', bpw: 16, bf16: true },
      Q_COMMON_AWQ, Q_COMMON_GPTQ, Q_COMMON_FP8,
    ],
    kvQuants: [
      { id: 'fp16', name: 'FP16', bpw: 16 },
      { id: 'fp8',  name: 'FP8 (需CC≥8.9)', bpw: 8, needCC: 8.9 },
    ],
    fixedOH: 1.5, fracOH: 0.05, decodeEff: 0.75, mfu: 0.45,
    desc: '生产级服务首选,并发吞吐高;仅Linux,需GPU,支持TP多卡。',
  },
  {
    id: 'llamacpp', name: 'llama.cpp', tag: 'CPU/GPU 混合推理 · GGUF 量化 · 可部分卸载',
    minCC: 6.0, linux: false, tp: 'pipeline', offload: true, cpuOnly: true, rocm: true, metal: true,
    quants: [
      { id: 'f16',  name: 'F16 (2字节)',       bpw: 16 },
      { id: 'q8_0', name: 'Q8_0 (~8.5bit)',    bpw: 8.5 },
      { id: 'q6_k', name: 'Q6_K (~6.6bit)',    bpw: 6.6 },
      { id: 'q5km', name: 'Q5_K_M (~5.7bit)',  bpw: 5.7 },
      { id: 'q4km', name: 'Q4_K_M (~4.9bit)',  bpw: 4.85 },
      { id: 'q3km', name: 'Q3_K_M (~3.9bit)',  bpw: 3.9 },
      { id: 'q2k',  name: 'Q2_K (~3.4bit)',    bpw: 3.4 },
    ],
    kvQuants: [
      { id: 'f16', name: 'F16',   bpw: 16 },
      { id: 'q8',  name: 'Q8_0',  bpw: 8.5 },
      { id: 'q4',  name: 'Q4_0',  bpw: 4.5 },
    ],
    fixedOH: 0.8, fracOH: 0.02, decodeEff: 0.7, mfu: 0.35,
    desc: '单机/边缘部署神器,显存不够可把部分层放CPU,全平台支持。',
  },
  {
    id: 'sglang', name: 'SGLang', tag: 'RadixAttention · 高并发服务 · 仅Ampere+',
    minCC: 8.0, linux: true, tp: 'tensor', offload: false, cpuOnly: false, rocm: true, metal: false,
    quants: [
      { id: 'fp16', name: 'FP16 (2字节)', bpw: 16 },
      { id: 'bf16', name: 'BF16 (2字节)', bpw: 16, bf16: true },
      Q_COMMON_AWQ, Q_COMMON_FP8,
    ],
    kvQuants: [
      { id: 'fp16', name: 'FP16', bpw: 16 },
      { id: 'fp8',  name: 'FP8 (需CC≥8.9)', bpw: 8, needCC: 8.9 },
    ],
    fixedOH: 2.0, fracOH: 0.05, decodeEff: 0.78, mfu: 0.48,
    desc: 'vLLM 的有力竞争者,复杂提示场景吞吐更优;需要较新GPU。',
  },
  {
    id: 'trtllm', name: 'TensorRT-LLM', tag: 'NVIDIA 极致优化 · 工程成本高',
    minCC: 8.0, linux: true, tp: 'tensor', offload: false, cpuOnly: false, rocm: false, metal: false,
    quants: [
      { id: 'fp16', name: 'FP16 (2字节)', bpw: 16 },
      { id: 'bf16', name: 'BF16 (2字节)', bpw: 16, bf16: true },
      { id: 'int8', name: 'INT8 (W8A8)',  bpw: 8 },
      Q_COMMON_AWQ, Q_COMMON_FP8,
    ],
    kvQuants: [ { id: 'fp16', name: 'FP16', bpw: 16 } ],
    fixedOH: 2.0, fracOH: 0.04, decodeEff: 0.85, mfu: 0.55,
    desc: '延迟最低,需模型转换与调优,适合追求极致的NVIDIA集群。',
  },
  {
    id: 'transformers', name: 'HF Transformers', tag: '基线实现 · 灵活但慢 · 可CPU/磁盘卸载',
    minCC: 6.0, linux: false, tp: 'pipeline', offload: true, cpuOnly: true, rocm: true, metal: true,
    quants: [
      { id: 'fp32', name: 'FP32 (4字节)',          bpw: 32 },
      { id: 'fp16', name: 'FP16 (2字节)',          bpw: 16 },
      { id: 'bf16', name: 'BF16 (2字节)',          bpw: 16, bf16: true },
      { id: 'int8', name: 'INT8 (bitsandbytes)',   bpw: 8 },
      { id: 'int4', name: 'INT4 NF4 (bitsandbytes)', bpw: 4.5 },
    ],
    kvQuants: [ { id: 'fp16', name: 'FP16', bpw: 16 } ],
    fixedOH: 1.0, fracOH: 0.02, decodeEff: 0.5, mfu: 0.22,
    desc: ' transformers 原生 pipeline,适合验证与实验,不适合高吞吐服务。',
  },
];

/* 快速场景示例 */
const SCENARIOS = [
  {
    id: 'demo-v100', name: '示例A: Qwen3-32B + 1×V100-32GB · vLLM FP16',
    model: 'qwen3-32b', fw: 'vllm', quant: 'fp16', kvQuant: 'fp16', ctx: 4096, conc: 1,
    board: [['gpu', 'v100-32']],
  },
  {
    id: 'demo-v100-q4', name: '示例B: Qwen3-32B Q4 + 1×V100-32GB · llama.cpp',
    model: 'qwen3-32b', fw: 'llamacpp', quant: 'q4km', kvQuant: 'f16', ctx: 4096, conc: 1,
    board: [['gpu', 'v100-32']],
  },
  {
    id: 'demo-offload', name: '示例C: Qwen3-32B Q4 + V100-16GB + EPYC · CPU卸载',
    model: 'qwen3-32b', fw: 'llamacpp', quant: 'q4km', kvQuant: 'f16', ctx: 4096, conc: 1,
    board: [['gpu', 'v100-16'], ['cpu', 'epyc-9654'], ['ram', 'ddr5-64'], ['ram', 'ddr5-64'], ['ram', 'ddr5-64'], ['ram', 'ddr5-64'], ['ram', 'ddr5-64'], ['ram', 'ddr5-64']],
  },
  {
    id: 'demo-h100', name: '示例D: Qwen2.5-72B FP8 + 4×H100 · vLLM TP4',
    model: 'qwen2.5-72b', fw: 'vllm', quant: 'fp8', kvQuant: 'fp16', ctx: 8192, conc: 8,
    board: [['gpu', 'h100'], ['gpu', 'h100'], ['gpu', 'h100'], ['gpu', 'h100']],
  },
  {
    id: 'demo-moe', name: '示例E: Qwen3-30B-A3B Q4 + M2 Ultra · llama.cpp',
    model: 'qwen3-30b-a3b', fw: 'llamacpp', quant: 'q4km', kvQuant: 'f16', ctx: 8192, conc: 1,
    board: [['gpu', 'm2ultra']],
  },
  {
    id: 'demo-cpu', name: '示例F: Qwen3-32B Q4 + EPYC 纯CPU · llama.cpp',
    model: 'qwen3-32b', fw: 'llamacpp', quant: 'q4km', kvQuant: 'q8', ctx: 2048, conc: 1,
    board: [['cpu', 'epyc-9654'], ['ram', 'ddr5-64'], ['ram', 'ddr5-64'], ['ram', 'ddr5-64'], ['ram', 'ddr5-64'], ['ram', 'ddr5-64'], ['ram', 'ddr5-64'], ['ram', 'ddr5-64'], ['ram', 'ddr5-64'], ['ram', 'ddr5-64'], ['ram', 'ddr5-64'], ['ram', 'ddr5-64']],
  },
  {
    id: 'demo-glm53', name: '示例G: GLM-5.3-Flash FP8 + 4×H100 · vLLM TP4',
    model: 'glm-5.3-flash', fw: 'vllm', quant: 'fp8', kvQuant: 'fp16', ctx: 32768, conc: 8,
    board: [['gpu', 'h100'], ['gpu', 'h100'], ['gpu', 'h100'], ['gpu', 'h100']],
  },
  {
    id: 'demo-qwen38next', name: '示例H: Qwen3.8-Flash-Next Q4 + M2 Ultra · llama.cpp',
    model: 'qwen3.8-flash-next', fw: 'llamacpp', quant: 'q4km', kvQuant: 'f16', ctx: 16384, conc: 1,
    board: [['gpu', 'm2ultra']],
  },
  {
    id: 'demo-4090x2', name: '示例I: Qwen3-32B AWQ + 2×RTX4090 TP2 · vLLM (PCIe)',
    model: 'qwen3-32b', fw: 'vllm', quant: 'awq', kvQuant: 'fp16', ctx: 8192, conc: 4,
    board: [['gpu', 'rtx4090'], ['gpu', 'rtx4090']],
  },
];
