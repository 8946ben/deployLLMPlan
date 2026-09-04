# 更新日志

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/);每轮功能改动以版本 + 日期记录,对应提交哈希附在段落标题。**所有会话在提交前须把改动写入「未发布」段**(见 AGENTS.md)。

## [未发布]

### 新增
- **整机类硬件**:硬件库新增「整机」分类,添加到画板时自动展开为组成部件(GPU/CPU/内存),接入既有分析与拓扑逻辑;首批成员 **NVIDIA DGX-1**(8×V100-SXM2-32GB · 双路E5-2698 v4 · 512GB · NVLink2 全互联)与**思腾合力 GPU 服务器**(3×V100S-PCIE-32GB · 双路Gold 6326 · 256GB · 无NVLink桥接走 PCIe),两台配置均已 SSH 实测核实(nvidia-smi/lscpu/free)
- GPU 目录新增 Tesla V100S-PCIE 32GB(Volta PCIe 版,共 21 款);CPU 平台新增 Intel Xeon E5-2698 v4 ×2 与 Xeon Gold 6326 ×2(共 5 款)

### 变更
- **部署画板重构为「画板即集群」**:每块 GPU 独立成节点,卡间按真实拓扑连线并标注速率(A100/H100 多卡经 NVSwitch 星型全互联,V100 混合立方网格,PCIe 卡经主机星型),CPU/内存节点画在下方;原独立拓扑面板移除,互联方式选择器移入画板标题栏

### 修复
- README 硬件库行误回退为 15 款 GPU,与 data.js 实际 20 款不符,已恢复

## [0.2.0] - 2026-09-05(591f9c8)

### 新增
- **报告计算细节展开**:分析报告每一项(部署判定流程、显存需求 6 行、性能预估 6 卡、层分布、瓶颈分析)可点击展开,显示代入当前配置实际数值的分步公式;展开状态跨重渲染保留,性能卡片同时只展开一个
- **GPU 互联建模**:NVLink 2.0/3.0/4.0/5.0、xGMI、PCIe 3/4/5、InfiniBand NDR、100GbE、统一内存;按 GPU 自动推断可手动覆盖;TP 通信开销(allreduce 字节 ÷ 卡间带宽 + 链路延迟)计入解码速度
- **集群拓扑图**:GPU ↔ NVSwitch/卡间直连/PCIe ↔ CPU ↔ 内存,每条链路标注速率;3 卡以上 fabric 画交换芯片星型
- **5 款新显卡**(硬件库 15 → 20 款):NVIDIA H200 SXM 141GB、B200 SXM 180GB、B300 (Blackwell Ultra) 288GB、RTX PRO 6000 96GB、DGX Spark 128GB(GB10 统一内存)
- 快速场景示例 I:Qwen3-32B AWQ + 2×RTX4090 PCIe TP2

### 变更
- KV 缓存显示跟随「服务上下文」输入实时联动;选中预设模型或导入 config.json 时,服务上下文默认取模型原生上下文(如 Qwen3.8-27B → 256K)

### 协作
- 新增 `AGENTS.md` 多会话协作规则与 `tools/lock.js` 范围互斥锁(重叠范围互斥、不重叠并行、10 分钟租约、超时接管、释放前强制提交本范围改动)

## [0.1.0] - 2026-09-04(d59a53a)

### 新增
- 首个版本:**LLM 推理部署规划器** — 纯前端、零依赖、离线可用的 LLM 推理部署模拟工具
- 模型配置:12 个预设(Qwen3.8-27B / DeepSeek-V4-Flash / Qwen3.8-Flash-Next / GLM-5.3-Flash / Qwen3 / Qwen2.5 / Llama3 等,2026 新架构按官方 config.json 录入);HuggingFace config.json 导入;MoE、混合注意力(KV 缓存层数)、MLA 覆盖参数
- 推理框架:vLLM / llama.cpp / SGLang / TensorRT-LLM / HF Transformers,各自量化格式(GGUF Q2~Q8、AWQ/GPTQ INT4、FP8)、KV 缓存精度、多卡策略(TP/层切分)、CPU 卸载
- 硬件库:15 款 GPU、3 款 CPU 平台、DDR5 内存条,点击/拖放到部署画板,实时汇总显存与带宽
- 分析报告:✅/⚠️/❌ 部署判定、显存构成条形图、架构兼容性检查清单、解码/预填充速度、TTFT、最大并发、总吞吐、瓶颈分析、可执行建议
- 8 个快速场景示例(失败分析/成功部署/CPU 卸载/TP 集群/MoE 统一内存/纯 CPU)
- 引擎测试:31 项断言(参数量校验、14 个部署场景与速度合理区间)
