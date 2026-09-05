# LLM 推理部署规划器

一个纯前端(零依赖、离线可用)的 LLM 推理部署模拟工具:导入模型配置 → 选择推理框架与量化 → 把硬件拖到画板上 → 实时得到**能否部署、瓶颈在哪、速度多少**。

## 启动方式

- **双击 `index.html`** 即可用(无需构建、无外部依赖、可离线);或
- `node serve.js [端口]` → 浏览器打开 `http://127.0.0.1:8765`

## 功能

| 步骤 | 说明 |
|---|---|
| ① 模型配置 | 内置 Qwen3.8-27B / Qwen3.8-Flash-Next / DeepSeek-V4-Flash / GLM-5.3-Flash / Qwen3 / Qwen2.5 / Llama3 等预设(新架构按官方 config.json 录入);可导入 HuggingFace `config.json`;支持 MoE、混合注意力(KV 缓存层数)、MLA 覆盖参数 |
| ② 推理框架 | vLLM / llama.cpp / SGLang / TensorRT-LLM / HF Transformers,各自的量化格式(含 GGUF Q2~Q8、AWQ/GPTQ INT4、FP8)、KV 缓存精度、多卡策略(TP/层切分)、CPU 卸载能力 |
| ③ 服务场景 | 服务上下文长度、并发路数、提示词长度(TTFT 估算用);KV 缓存按服务上下文计算,选中预设模型/导入 config.json 时**默认取模型原生上下文**,可随时修改 |
| ④ 硬件库 | 按品牌/代际分组陈列、代际内按性能从强到弱排序:19 款 GPU 卡(NVIDIA 的 V/T/A/40/H/B 系列与 AMD MI)、5 款 CPU 平台(Intel/AMD)、DDR5 内存条、**FPGA**(Alinx AXU9EGB,规格取自 LLM_FPGA 实测工程)、**整机 4 成员**(DGX-1、思腾合力、Mac Studio M2 Ultra、DGX Spark——添加时自动展开为部件);点击卡片或拖拽添加,点 **ⓘ** 弹出详细属性面板(含**参考价格与可跳转来源链接**);画板汇总条显示整板参考总价 |
| ⑤ 部署画板 | 画板即集群:**每块 GPU 独立成节点**(8 块 V100 就画 8 张相连的卡),卡间按真实拓扑连线并标注速率——A100/H100 多卡经 NVSwitch 交换背板星型全互联,V100(NVLink2)画混合立方网格,PCIe 卡经主机星型;下方是 CPU 节点与内存节点(×N);**整机呈独立虚框分区**(右上角 ✕ 一键整体移除);互联方式可选/自动推断(NVLink 2/3/4/5、xGMI、PCIe 3/4/5、IB 400G、100GbE、1GbE) |
| 📊 分析报告 | ✅/⚠️/❌ 部署判定 + 原因;显存构成条形图(权重/KV/开销/缺口);架构兼容性检查清单;解码/预填充速度、TTFT、最大并发、总吞吐、瓶颈分析(含 TP 通信开销);可执行的改进建议;报告各项可**点击展开**查看代入实际数值的计算细节 |

右上角「快速场景」内置 9 个示例:V100 跑 32B 失败分析、llama.cpp Q4 成功部署、CPU 卸载混合推理、4×H100 FP8 集群、MoE + Apple 统一内存、纯 CPU 部署、GLM-5.3-Flash 4×H100、Qwen3.8-Flash-Next + M2 Ultra、2×4090 PCIe TP。

## 估算模型(核心公式)

所有容量/带宽按**十进制 GB**(与硬件规格书一致)计算。

- **参数量**:按 transformer 结构累加(Attention Q/K/V/O + MLP ×3 + 词嵌入;GQA 下 K/V 按 KV 头数;MoE 分别算总参数与激活参数)
- **权重显存** = `总参数 × 量化位宽/8`(如 Q4_K_M ≈ 4.85 bit,AWQ ≈ 4.5 bit,FP8 = 8 bit)
- **KV 缓存** = `2 × 带KV缓存的注意力层数 × KV头数 × 头维度 × 每元素字节 × 上下文 × 并发`(混合注意力架构如 Qwen3.8-27B:64 层中仅 16 层全注意力产生逐 token KV,其余线性注意力层状态恒定)
- **框架开销** = 固定量(vLLM ≈1.5GB 等)+ 权重的一定比例
- **解码速度**(带宽受限):`tok/s ≈ 有效带宽 / 每 token 读取字节(激活权重 + 全量KV)`
  - 张量并行(TP):各卡并行读分片,取最慢卡;TP 通信开销按互联带宽单独建模:`allreduce字节(2×层×隐藏维×2B×2(N-1)/N) ÷ 卡间带宽 + 每层2次×链路延迟(NVLink ~5µs / PCIe ~20µs)` —— NVLink 900GB/s 时几乎可忽略,PCIe 4.0 x16(32GB/s)时可达数 ms/token
  - 层切分(llama.cpp 多卡/卸载):各卡串行累加 → 多卡主要扩容不明显提速,卡间仅传激活向量,互联要求低
  - 框架效率系数:vLLM 0.75 / TRT-LLM 0.85 / llama.cpp 0.70 / Transformers 0.50;MoE batch=1 额外 ×0.55(小专家 GEMM 利用率低),多卡 MoE 再 ×0.75(EP 路由/all-to-all)
- **预填充速度**(算力受限):`FLOPs/token ÷ (峰值TFLOPS × MFU)`,MFU 取 0.22~0.55;张量并行算力累加,层切分(llama.cpp 多卡)逐层串行、受最慢单卡限制
- **CPU 卸载**:放不下的层进内存,解码时间 = GPU 部分耗时 + CPU 部分按内存有效带宽(内存通道数 × 38.4GB/s,受插条数限制)的耗时

## 兼容性检查规则

- 计算能力:vLLM ≥7.0,SGLang/TensorRT-LLM ≥8.0;FP8/FP8-KV 需 ≥8.9;BF16 在 Volta(CC 7.0)上提示改用 FP16
- 生态:vLLM/SGLang/TRT-LLM 需 Linux(提示 WSL2);AMD 走 ROCm、Apple 统一内存仅 llama.cpp/Transformers;禁止混插不同厂商 GPU
- 多卡:TP 混插不同型号告警;TP 走 PCIe/网络时按互联速率告警通信开销;TP 均分超过最小卡显存告警;互联方式默认按 GPU 自动推断(同代 NVLink 卡→NVLink,MI250X→xGMI,消费卡/PCIe版→对应代数 PCIe),可手动覆盖做 what-if

## 已知近似(刻意取舍)

- 参数量按公开配置推算,与官方标称可能差 ±5%(MoE 官方激活参数含共享专家,本工具略低 ~10%;Qwen3.8-27B 按官方 config.json 实测规格录入,估算语言主体 ≈24.4B,官方 27B 含视觉塔与 MTP 头,视觉推理请在结果上自行加 ~1GB)
- **2026 新架构 MoE**(DeepSeek-V4-Flash / Qwen3.8-Flash-Next / GLM-5.3-Flash)按官方 config.json 录入,但 MLA 潜空间 KV、DSA 稀疏索引、KDA 线性层用覆盖字段近似:估算总参数分别 ≈284B / 175B(含 51B n-gram 表)/ 334B,激活 ≈13.8B / 5.6B / 16.9B;KV 每 token ≈48KB / 24KB / 11KB(FP16)。MLA 的 KV 为潜空间近似值,实际随缓存实现不同可能更低
- **MoE 解码效率修正**:batch=1 下小专家 GEMM 带宽利用率低 + 路由开销(×0.55),TP 多卡逐层 allreduce 延迟(再 ×0.75);即便如此,大 MoE 的速度估算仍偏上限,建议按“上界”理解
- 速度是数量级正确的工程估算,不含 kernel 调优、CUDA Graph、调度开销等实现细节;实测通常在 ±30% 内
- **实测校准锚点**(2026-09-05,DGX-1 · 单卡 V100-32GB · llama.cpp Q5_K_M · Qwen3.8-27B · 32K ctx):解码 31.3 tok/s、预填充 679 tok/s(1.1K prompt);对应解码带宽利用率 60.5%、预填充 MFU 29.5%。据此校准 llama.cpp 效率系数(decodeEff 0.65 / MFU 0.32)后,估算解码 33.5、预填充 728 tok/s,偏差约 +7%。这是目前唯一真机锚点(基于 V100 一代),Ampere 及之后的卡效率通常更高
- 未建模:MLA(DeepSeek 系)、Speculative Decoding、prefix cache 命中、网络传输

## 测试

```
node tests/run-tests.js
```

31 项断言:参数量校验、14 个部署场景(失败/成功/卸载/纯CPU/兼容性拦截)与速度合理区间。

## 目录结构

```
index.html   页面骨架      style.css  深色主题样式
data.js      硬件/模型/框架目录  engine.js  分析与估算引擎(纯函数)
app.js       UI 交互       serve.js   可选本地服务器
tests/run-tests.js  引擎测试
CHANGELOG.md 更新日志      AGENTS.md  多会话协作规则
tools/lock.js  会话范围锁
```

扩展:在 `data.js` 中按现有格式追加 GPU/模型/框架即可,报告自动生效。
