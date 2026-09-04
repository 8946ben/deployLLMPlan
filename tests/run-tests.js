'use strict';
/* 引擎单元/场景测试:把 data.js + engine.js 拼接后用 node 运行断言。 */
const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync } = require('child_process');

const src = ['data.js', 'engine.js']
  .map(f => fs.readFileSync(path.join(__dirname, '..', f), 'utf8'))
  .join('\n');

const testBody = `
'use strict';
let failed = 0;
function assert(cond, msg) {
  if (cond) { console.log('  ok - ' + msg); }
  else { console.error('  FAIL - ' + msg); failed++; }
}
function near(x, y, tolPct, msg) {
  const ok = Math.abs(x - y) / y <= tolPct / 100;
  assert(ok, msg + ' (got ' + x + ', expect ~' + y + ')');
}
const fw = id => FRAMEWORKS.find(f => f.id === id);
const gpu = id => GPUS.find(g => g.id === id);
const cpu = id => CPUS.find(c => c.id === id);
const ram = () => RAMS.find(r => r.id === 'ddr5-64');
const model = id => MODELS.find(m => m.id === id);
function run(m, f, q, kvq, ctx, conc, gpus, cpus, rams, promptLen) {
  return analyze({
    model: m, fw: fw(f), quant: fw(f).quants.find(x => x.id === q) || fw(f).quants[0],
    kvQuant: fw(f).kvQuants.find(x => x.id === kvq) || fw(f).kvQuants[0],
    ctx, conc, promptLen: promptLen || 1024,
    gpus: (gpus || []).map(gpu), cpus: (cpus || []).map(cpu), rams: (rams || []).map(() => ram()),
  });
}

console.log('== 参数量估算 ==');
const q38 = computeParams(model('qwen3.8-27b'));
near(q38.total / 1e9, 24.4, 5, 'Qwen3.8-27B 语言主体 ≈24.4B (官方27B含视觉塔/MTP)');
assert(kvPerToken(model('qwen3.8-27b'), 16) === 2 * 16 * 4 * 256 * 2, '混合注意力:仅16/64层产生KV,64KB/token@FP16');
const o1 = run(model('qwen3.8-27b'), 'vllm', 'fp16', 'fp16', 4096, 1, ['v100-32']);
assert(o1.verdict === 'fail' && o1.checks.some(c => c.level === 'fail' && c.msg.includes('显存不足')), 'Qwen3.8-27B FP16 单卡V100 不可部署(需求 ' + (o1.mem.requiredB / 1e9).toFixed(1) + 'GB)');
const o2 = run(model('qwen3.8-27b'), 'llamacpp', 'q4km', 'f16', 32768, 1, ['v100-32']);
assert(o2.verdict === 'ok' && o2.speed && o2.speed.decodeTps > 25, 'Q4 + 32K上下文 单卡V100 可部署,解码 ' + (o2.speed ? o2.speed.decodeTps.toFixed(1) : '?') + ' tok/s (混合注意力KV小)');
const q32 = computeParams(model('qwen3-32b'));
near(q32.total / 1e9, 32.8, 4, 'Qwen3-32B 总参数 ≈32.8B');
assert(q32.active === q32.total, 'Qwen3-32B 稠密模型 active=total');
const moe = computeParams(model('qwen3-30b-a3b'));
near(moe.total / 1e9, 30.5, 5, 'Qwen3-30B-A3B 总参数 ≈30.5B');
near(moe.active / 1e9, 3.0, 15, 'Qwen3-30B-A3B 激活参数 ≈3.0B (官方口径3.3B含共享专家)');
const l8 = computeParams(model('llama-3.1-8b'));
near(l8.total / 1e9, 8.0, 3, 'Llama-3.1-8B 总参数 ≈8B');
const q05 = computeParams(model('qwen2.5-0.5b'));
near(q05.total / 1e9, 0.494, 5, 'Qwen2.5-0.5B 总参数 ≈0.49B');
assert(kvPerToken(model('qwen2.5-0.5b'), 16) === 2 * 24 * 2 * 64 * 2, 'Qwen2.5-0.5B KV缓存 12KB/token@FP16');

console.log('== 2026 新架构 MoE 模型 ==');
const dsv4 = computeParams(model('dsv4-flash'));
near(dsv4.total / 1e9, 284, 4, 'DeepSeek-V4-Flash 总参数 ≈284B');
near(dsv4.active / 1e9, 13.8, 8, 'DeepSeek-V4-Flash 激活参数 ≈13.8B');
assert(kvPerToken(model('dsv4-flash'), 16) === 49536, 'DeepSeek-V4-Flash MLA压缩KV ≈48KB/token');
const qf = computeParams(model('qwen3.8-flash-next'));
near(qf.total / 1e9, 175, 4, 'Qwen3.8-Flash-Next 总参数 ≈175B (含51B n-gram表)');
near(qf.active / 1e9, 5.6, 12, 'Qwen3.8-Flash-Next 激活参数 ≈5.6B');
assert(kvPerToken(model('qwen3.8-flash-next'), 16) === 24576, 'Qwen3.8-Flash-Next 12层全注意力KV 24KB/token');
const glm = computeParams(model('glm-5.3-flash'));
near(glm.total / 1e9, 334, 4, 'GLM-5.3-Flash 总参数 ≈334B (官方320B,3层稠密MLP未区分)');
near(glm.active / 1e9, 16.9, 10, 'GLM-5.3-Flash 激活参数 ≈17B (官方18B)');
assert(kvPerToken(model('glm-5.3-flash'), 16) === 11264, 'GLM-5.3-Flash MLA潜空间KV ≈11KB/token');
const g1 = run(model('glm-5.3-flash'), 'vllm', 'fp8', 'fp16', 32768, 8, ['h100','h100','h100','h100']);
assert(g1.verdict === 'fail' && g1.checks.some(c => c.level === 'fail' && c.msg.includes('显存不足')), 'GLM-5.3-Flash FP8 4×H100 差一点: ' + (g1.mem.requiredB / 1e9).toFixed(0) + 'GB > 320GB');
const g2 = run(model('glm-5.3-flash'), 'vllm', 'fp8', 'fp16', 32768, 8, ['h100','h100','h100','h100','h100']);
assert(g2.verdict === 'ok' && g2.speed && g2.speed.decodeTps > 100, 'GLM-5.3-Flash FP8 5×H100 可部署,解码 ' + (g2.speed ? g2.speed.decodeTps.toFixed(0) : '?') + ' tok/s');
const qn = run(model('qwen3.8-flash-next'), 'llamacpp', 'q4km', 'f16', 16384, 1, ['m2ultra']);
assert(qn.verdict === 'ok' && qn.speed && qn.speed.decodeTps > 80, 'Qwen3.8-Flash-Next Q4 + M2 Ultra 可部署,解码 ' + (qn.speed ? qn.speed.decodeTps.toFixed(0) : '?') + ' tok/s');
const dv = run(model('dsv4-flash'), 'vllm', 'fp8', 'fp16', 8192, 8, ['h100','h100','h100','h100']);
assert(dv.verdict === 'ok' && dv.speed && dv.speed.decodeTps > 100, 'DeepSeek-V4-Flash FP8 4×H100 可部署,解码 ' + (dv.speed ? dv.speed.decodeTps.toFixed(0) : '?') + ' tok/s (roofline上限)');

console.log('== 场景A: Qwen3-32B FP16 + 1xV100-32GB + vLLM (应失败:显存不足) ==');
const a = run(model('qwen3-32b'), 'vllm', 'fp16', 'fp16', 4096, 1, ['v100-32']);
assert(a.verdict === 'fail', '判定为不可部署');
assert(a.checks.some(c => c.level === 'fail' && c.msg.includes('显存不足')), '失败原因是显存不足');
assert(a.speed === null, '无速度估算');
assert(a.suggestions.length > 0, '给出建议: ' + a.suggestions[0]);

console.log('== 场景B: Qwen3-32B Q4_K_M + 1xV100-32GB + llama.cpp (应可部署) ==');
const b = run(model('qwen3-32b'), 'llamacpp', 'q4km', 'f16', 4096, 1, ['v100-32']);
assert(b.verdict === 'ok', '判定为可部署 (verdict=' + b.verdict + ')');
assert(b.mode === 'gpu', '全GPU部署');
assert(b.speed && b.speed.decodeTps > 15 && b.speed.decodeTps < 45, '解码速度合理: ' + (b.speed && b.speed.decodeTps.toFixed(1)) + ' tok/s');
assert(b.speed && b.speed.prefillTps > 200, '预填充速度合理: ' + (b.speed ? b.speed.prefillTps.toFixed(0) : 'null') + ' tok/s');

console.log('== 场景C: Qwen3-32B Q4_K_M + 1xV100-16GB + EPYC (应CPU卸载) ==');
const c = run(model('qwen3-32b'), 'llamacpp', 'q4km', 'f16', 4096, 1, ['v100-16'], ['epyc-9654'], [1,2,3,4,5,6,7,8]);
assert(c.verdict === 'warn', '判定为警告(可部署有折损)');
assert(c.mode === 'offload', '模式=offload');
assert(c.speed && c.speed.offloadFrac > 0.1 && c.speed.offloadFrac < 0.6, '卸载比例合理: ' + (c.speed && (c.speed.offloadFrac * 100).toFixed(0)) + '%');
assert(c.speed && c.speed.decodeTps > 5 && c.speed.decodeTps < 35, '混合推理速度合理: ' + (c.speed && c.speed.decodeTps.toFixed(1)) + ' tok/s');

console.log('== 场景D: Qwen2.5-72B FP8 + 4xH100 + vLLM TP4 (应可部署) ==');
const d = run(model('qwen2.5-72b'), 'vllm', 'fp8', 'fp16', 8192, 8, ['h100','h100','h100','h100']);
assert(d.verdict === 'ok', '判定可部署');
assert(d.speed && d.speed.decodeTps > 60 && d.speed.decodeTps < 250, 'TP4解码: ' + (d.speed && d.speed.decodeTps.toFixed(0)) + ' tok/s');
assert(d.speed && d.speed.aggTps > d.speed.decodeTps * 1.5, '并发总吞吐高于单流: ' + (d.speed && d.speed.aggTps.toFixed(0)));

console.log('== 场景E: BF16 在 V100 上 (应警告建议FP16) ==');
const e = run(model('qwen3-32b'), 'vllm', 'bf16', 'fp16', 2048, 1, ['a100-40']);
assert(!e.checks.some(x => x.msg.includes('BF16')), 'A100 上 BF16 不应告警');
const e2 = run(model('qwen3-32b'), 'vllm', 'bf16', 'fp16', 2048, 1, ['v100-32']);
assert(e2.checks.some(x => x.level === 'warn' && x.msg.includes('BF16') && x.msg.includes('FP16')), 'V100+BF16 → 建议改FP16');

console.log('== 场景F: FP8 在 A100 上 (应失败, 需CC>=8.9) ==');
const f = run(model('qwen3-32b'), 'vllm', 'fp8', 'fp16', 2048, 1, ['a100-80']);
assert(f.verdict === 'fail' && f.checks.some(x => x.level === 'fail' && x.msg.includes('FP8')), 'A100 不支持 FP8 被正确拦截');

console.log('== 场景G: SGLang 在 V100 上 (应失败, 需CC>=8.0) ==');
const g = run(model('qwen3-32b'), 'sglang', 'fp16', 'fp16', 2048, 1, ['v100-32']);
assert(g.verdict === 'fail' && g.checks.some(x => x.level === 'fail' && x.msg.includes('计算能力')), 'SGLang 拦截 CC7.0');

console.log('== 场景H: M2 Ultra + llama.cpp MoE (应可部署且快) ==');
const h = run(model('qwen3-30b-a3b'), 'llamacpp', 'q4km', 'f16', 8192, 1, ['m2ultra']);
assert(h.verdict === 'ok', '统一内存可部署');
assert(h.speed && h.speed.decodeTps > 80, 'MoE 激活参数少,速度高: ' + (h.speed && h.speed.decodeTps.toFixed(0)) + ' tok/s');

console.log('== 场景I: M2 Ultra + vLLM (应失败) ==');
const i = run(model('qwen3-30b-a3b'), 'vllm', 'fp16', 'fp16', 2048, 1, ['m2ultra']);
assert(i.verdict === 'fail' && i.checks.some(x => x.msg.includes('Metal')), 'Apple 硬件只支持 llama.cpp');

console.log('== 场景J: 纯 CPU EPYC 部署 (llama.cpp) ==');
const j = run(model('qwen3-32b'), 'llamacpp', 'q4km', 'q8', 2048, 1, [], ['epyc-9654'], [1,2,3,4,5,6,7,8,9,10,11,12]);
assert(j.verdict === 'warn' && j.mode === 'cpu', '纯CPU模式');
assert(j.speed && j.speed.decodeTps > 8 && j.speed.decodeTps < 30, 'CPU速度合理: ' + (j.speed && j.speed.decodeTps.toFixed(1)) + ' tok/s');

console.log('== 场景K: 纯 CPU 但 vLLM (应失败) ==');
const k = run(model('qwen3-32b'), 'vllm', 'fp16', 'fp16', 2048, 1, [], ['epyc-9654'], [1,2,3,4,5,6,7,8]);
assert(k.verdict === 'fail' && k.checks.some(x => x.msg.includes('GPU')), 'vLLM 无 GPU 被拦截');

console.log('== 场景L: 内存也装不下的卸载 (72B F16 + T4 + 128GB RAM, 应失败) ==');
const l = run(model('qwen2.5-72b'), 'llamacpp', 'f16', 'f16', 4096, 1, ['t4'], ['ryzen'], [1,2]);
assert(l.verdict === 'fail' && l.checks.some(x => x.level === 'fail' && (x.msg.includes('内存不足') || x.msg.includes('内存也不足'))), '内存不足被拦截');

console.log('== 场景M: 并发 KV 爆显存 → 建议降并发/缩上下文 ==');
const m = run(model('qwen3-32b'), 'vllm', 'awq', 'fp16', 32768, 32, ['a100-40']);
assert(m.verdict === 'fail', '32路×32K上下文应爆显存');
assert(m.suggestions.some(s => s.includes('KV') || s.includes('上下文')), '给出KV相关建议');

console.log('== 场景N: 多卡不同型号 TP 警告 ==');
const n = run(model('llama-3.1-8b'), 'vllm', 'fp16', 'fp16', 4096, 1, ['a100-40', 'rtx4090']);
assert(n.verdict === 'ok' || n.verdict === 'warn', '混合卡可运行');
assert(n.checks.some(x => x.level === 'warn' && x.msg.includes('相同型号')), '提示使用相同型号');

console.log('== 互联与集群拓扑 ==');
assert(resolveInterconnect([gpu('v100-32'), gpu('v100-32')]).id === 'nvlink2', 'V100×2 自动 → NVLink2 (300GB/s)');
assert(resolveInterconnect([gpu('h100'), gpu('h100'), gpu('h100'), gpu('h100')]).id === 'nvlink4', 'H100×4 自动 → NVLink4 (900GB/s)');
assert(resolveInterconnect([gpu('rtx4090'), gpu('rtx4090')]).id === 'pcie4', '4090×2 自动 → PCIe4 (32GB/s)');
assert(resolveInterconnect([gpu('a100-80'), gpu('rtx4090')]).id === 'pcie4', 'A100+4090 混合 → PCIe4');
assert(resolveInterconnect([gpu('mi250x'), gpu('mi250x')]).id === 'xgmi', 'MI250X×2 → xGMI');
assert(resolveInterconnect([gpu('m2ultra')]).id === 'unified', 'Apple → 统一内存');
const p4 = run(model('llama-3.1-8b'), 'vllm', 'fp16', 'fp16', 4096, 1, ['rtx4090', 'rtx4090', 'rtx4090', 'rtx4090']);
assert(p4.checks.some(c => c.level === 'warn' && c.msg.includes('PCIe')), 'PCIe TP 触发通信告警: ' + ((p4.checks.find(c => c.msg.includes('PCIe')) || {}).msg || '').slice(0, 40) + '...');
const icNv4 = INTERCONNECTS.find(i => i.id === 'nvlink4');
const n4 = analyze({ model: model('llama-3.1-8b'), fw: fw('vllm'), quant: fw('vllm').quants[0], kvQuant: fw('vllm').kvQuants[0], ctx: 4096, conc: 1, promptLen: 1024, gpus: ['rtx4090', 'rtx4090', 'rtx4090', 'rtx4090'].map(gpu), cpus: [], rams: [], ic: icNv4 });
assert(n4.speed.decodeTps > p4.speed.decodeTps * 1.15, '假设 NVLink4 后 TP4 解码显著快于 PCIe: ' + n4.speed.decodeTps.toFixed(0) + ' vs ' + p4.speed.decodeTps.toFixed(0) + ' tok/s');
assert(p4.speed.tpCommMs > n4.speed.tpCommMs * 2, 'PCIe 通信开销远大于 NVLink: ' + (p4.speed.tpCommMs * 1000).toFixed(0) + 'µs vs ' + (n4.speed.tpCommMs * 1000).toFixed(0) + 'µs');
const pipeIc = run(model('qwen3-32b'), 'llamacpp', 'q4km', 'f16', 4096, 1, ['v100-32', 'v100-32']);
assert(pipeIc.checks.some(c => c.level === 'info' && c.msg.includes('层切分')), '层切分模式显示互联 info 说明');

console.log(failed === 0 ? '\\nALL PASSED' : '\\n' + failed + ' FAILED');
process.exit(failed === 0 ? 0 : 1);
`;

const tmp = path.join(os.tmpdir(), 'llmplan-test-' + Date.now() + '.js');
fs.writeFileSync(tmp, src + '\n' + testBody);
try {
  execSync('node ' + JSON.stringify(tmp), { stdio: 'inherit' });
} finally {
  try { fs.unlinkSync(tmp); } catch (e) { /* ignore */ }
}
