---
title: "Kimi K3: Open Frontier Intelligence"
authors:
  - Kimi Team
year: 2026
venue: arXiv preprint
arxiv: https://arxiv.org/abs/2607.24653
version: arXiv v2 (2026-08-07)
url: https://huggingface.co/moonshotai/Kimi-K3
aliases:
  - Kimi K3
  - K3
tags:
  - llm
  - inference
  - vllm
  - moe
  - speculative-decoding
updated: 2026-11-12
---

# Kimi K3: Open Frontier Intelligence

这份笔记围绕 **K3 架构如何影响 vLLM 的缓存、并行和推测解码**展开。论文依据为 [arXiv v2](https://arxiv.org/abs/2607.24653v2)；K3.1 的 DSpark 草稿模型作为后续实现单独比较。

> [!info] 三种不同的结论来源
> **论文事实**描述 K3；**源码事实**限定于文中固定的 vLLM commit 或具体 checkpoint；**工程推断**用于解释显存和性能取舍。K3.1 的 DSpark 配置不属于 K3 论文的实验设置。

## 1. 从问题进入

| 想弄清的问题                               | 阅读位置                                                            |
| ------------------------------------ | --------------------------------------------------------------- |
| MLA 已经压缩 KV，为什么 TP 后缓存还会复制？          | [[MLA-TP-DP-DCP#3. TP 拆分 heads，为什么每张卡还需要完整缓存？]]                 |
| TP4、DP2 是否必须设置 DCP4？                 | [[MLA-TP-DP-DCP#5. TP4、DP2 时，DCP 如何选？]]                         |
| 通用参数合法，为什么 K3.1 专用后端仍会报错？            | [[MLA-TP-DP-DCP#6. 应用于 Kimi-K3.1-Pro，还要看专用路径的限制]]               |
| DSpark 草稿是不是一个输出 head？               | [[DSpark#0. 先回答：这个“草稿模型”是不是只是一个 head？]]                         |
| 草稿骨干并行，token 之间怎样建立依赖？               | [[DSpark#3. Markov head 的实际数学结构]]、[[DSpark#4. 联合分布、因果性与“半自回归”]] |
| Markov 修正如何进入目标验证？                   | [[DSpark#8. Markov 分布怎样接入 block verification？]]                 |
| 五层草稿、共享 LM head、Markov head 分别占多少参数？ | [[DSpark#9. 参数量、共享与串行开销]]                                       |

## 2. 先抓住影响推理的架构

论文 Table 1 和 §2 给出的 K3 主干配置：

| 项目                                    | K3                                    | 对推理的含义                            |
| ------------------------------------- | ------------------------------------- | --------------------------------- |
| 总参数 / 每 token 激活参数                    | 2.78T / 104.2B                        | 总权重容量与每 token 的计算规模要分别考虑          |
| 主干层数                                  | 93                                    | 包含 69 层 KDA、24 层 MLA              |
| Attention heads / hidden size         | 96 / 7168                             | TP 划分计算时需要考虑本地 head 数             |
| 混合 attention                          | 3 层 KDA + 1 层 Gated MLA 周期，最后额外一层 MLA | 同时管理 recurrent state 与随历史增长的缓存    |
| MLA 位置编码                              | NoPE                                  | 目标模型 MLA 不需要额外缓存 RoPE key 分量      |
| Routed experts                        | 896，每 token 激活 16                     | Expert 通信布局不等同于 attention 布局      |
| Shared experts / latent MoE dimension | 2 / 3584                              | 除 routed experts 外仍有投影与共享分支开销     |
| MTP                                   | 1 层                                   | 论文从该层微调 EAGLE-3 风格草稿              |
| 上下文窗口                                 | 1M token                              | 长上下文使缓存布局、prefix cache 和 TTFT 更关键 |

**激活 104.2B 参数不意味着只需存下 104.2B 参数。**在通常将权重驻留 GPU 的部署中，需要为全部权重及运行状态规划容量；每 token 的路由只决定本轮计算哪些专家。

论文 §4.1.4 将 MoE expert 权重做 MXFP4 QAT，激活采用 MXFP8，attention、latent MoE 投影、shared experts 和 router 等保留较高精度。这是论文的训练与部署方案；某个 vLLM checkpoint 的权重格式、KV cache dtype 和 kernel 支持还需分别核对。

## 3. MLA、TP、DP、DCP：哪些东西被切分？

架构公式见 [[MLA]]，位置编码方式见 [[NoPE]]；并行缓存的容量算例见 [[MLA-TP-DP-DCP]]。理解 K3 时最关键的是：

- **MLA 压缩的是每个历史 token 的表示。**同层同请求的各个 heads 从同一个 latent 读取信息。
- **普通 TP 主要切分 head 相关的投影和计算。**在这里引用的实现中，latent 本身在 TP ranks 上复制；切分 heads 不等于切分 latent 历史。
- **DCP 沿历史位置切分缓存。**每个 rank 保存部分 token 的完整 latent，再合并各分片的 attention 结果。
- **DP 为 attention 建立处理不同请求的组。**它不自动将一条请求的缓存跨 DP 组共享；MoE 的权重布局则要另外看 TP/EP 分组。

以 PP=PCP=1、TP4、DP2 为例，总 GPU 数是 8。DCP 复用各自 TP4 组内的 ranks，**不再乘一个 GPU 数量因子**。通用配置规则允许 DCP1/2/4；DCP4 可以消除每个 TP4 组内的 latent 历史重复，但不据此保证最快。

[固定版本的 K3 `latency_specialized` 后端](https://github.com/vllm-project/vllm/blob/a9dcfae9ef0756ce076d0053429d6c9afef067a3/vllm/models/kimi_k3/nvidia/specialized_decode/runtime/__init__.py#L125)有更严格的限制：DCP 只能为 1 或 TP；该路径下 TP4 还要求 EP，TP>1 的 EP 要求 DSA 和 MegaMoE 等条件。因此 **TP4、DCP2 在通用层面合法，不代表该专用路径接受它**。

同一源码还区分 SM100/103、CUDA Graph、KV dtype 和 speculative method 等条件。上述检查只是该版本的必要条件，不构成 TP4、DP2 一定可运行的证明。

> [!note] Attention 的 DP 不等于全部权重复制
> 引用的 [vLLM MoE 部署说明](https://github.com/vllm-project/vllm/blob/a9dcfae9ef0756ce076d0053429d6c9afef067a3/docs/serving/expert_parallel_deployment.md#L69)指出：routed experts 可在 TP×DP ranks 上做 TP；启用 EP 后在这些 ranks 之间分配专家。TP4、DP2 的两个 attention 组，不能直接解释成两份完整的 MoE 权重。

## 4. KDA 的固定状态为什么仍影响推测解码？

论文 §5.1 说明，KDA 用每 head 固定形状的 recurrent state 保存历史；MLA 则保留各历史位置的 latent。**固定大小描述的是 KDA 单个状态的形状，整个混合模型仍有随上下文增长的 MLA 缓存。**

论文 §5.1.2 的 **KDA Context Parallelism（KCP）**通过组合片段的状态转移恢复各段初始状态；vLLM 参数 **DCP**在这里讨论的是 decode 历史缓存分片。两者处理的对象、阶段和通信不同，不能仅因都有“context parallel”就等同。

推测解码中，目标模型一次验证多个候选，KDA 状态会沿这些候选推进。若部分候选被拒绝，后续计算只能继承已确认前缀对应的状态。论文 §5.4.2 的做法是：

1. 缓存较小的 token 投影输入，避免为每个候选保存一份完整 recurrent state。
2. 在片上重放已接受 token 的递推，恢复正确状态。
3. 将重放、bonus token 和下一草稿窗口涉及的操作组织进融合 kernel。

因此，一轮推测的成本除了起草和目标评分，还包含状态管理。论文中的这种优化思路不等于任意 vLLM 后端都已采用同一实现。

做容量估计时，可以先按组成项记账：

$$
M_{\mathrm{rank}}
\approx M_{\mathrm{weights,rank}}
+\frac{L_{\mathrm{MLA}}Trb}{D}
+M_{\mathrm{KDA\ state}}
+M_{\mathrm{draft}}
+M_{\mathrm{other}}.
$$

这里 $T$ 是一条请求的历史长度，$r$ 是 latent 维度，$b$ 是每元素字节数，$D$ 是 DCP；对 K3，$L_{\mathrm{MLA}}=24$。这是均匀分片、忽略页填充的解释式，多请求需分别累加。$M_{\mathrm{other}}$ 包括量化元数据、临时张量和 Graph 等；若讨论 K3.1 的 DSA，还需计入 indexer。不能用 latent 项的 $1/D$ 缩放替代整个模型的显存预算。

## 5. 论文草稿与 K3.1 DSpark 的区别

| 比较项    | K3 论文 §4.1.4                   | K3.1 的 Inferact DSpark checkpoint     |
| ------ | ------------------------------ | ------------------------------------- |
| 来源     | 预训练的单层 MTP 微调                  | 发布方提供的专用草稿模块                          |
| 骨干     | 单 decoder 层，EAGLE-3 风格递归起草     | 五层 dense MLA decoder，约 2.7B BF16 额外参数 |
| 目标特征   | 第 1、第 4、最后一个 AttnRes block 输出  | 五个目标层的 hidden states                  |
| 候选依赖   | 草稿循环消费此前输出                     | 骨干批量产生表示，Markov head 顺序修正并采样          |
| 草稿长度信息 | 训练时展开 7 步                      | 模型卡报告每 forward 起草 7 token；本笔记分析 K=4   |
| 训练说明   | 目标冻结，优化 draft 和融合投影，使用 LK loss | 不能由 K3 论文推出该 checkpoint 的完整训练配方       |

K3 的目标 MLA 使用 NoPE；[K3.1 DSpark 模型卡](https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark-preview)另外声明了草稿的 causal MLA 与 YaRN。**目标模型和草稿模型的位置编码要分别读取配置。**

[[DSpark]] 中保留了具体结构、数学推导与源码位置，包括：

- 由目标 hidden states 构建草稿自己的 context KV，复用目标 embedding 和 LM head。
- Vanilla Markov head 的低秩 bias、条件分布与候选联合分布。
- 训练梯度、接受率与分布距离的关系，以及 confidence head 的作用边界。
- Block verification 的累计比值、最长接受前缀、残差纠正和 bonus token。

其中固定 revision 的模型配置细项仍有待复核项，见概念笔记的“配置快照与记号”。

## 6. 用 K=4 理解 vLLM 的一轮推测

以下只是所研究的 speculative 配置片段；目标模型、权重 dtype、硬件和并行布局需另外指定：

```json
{
  "method": "dspark",
  "model": "Inferact/Kimi-K3.1-Pro-DSpark-preview",
  "num_speculative_tokens": 4,
  "draft_sample_method": "probabilistic",
  "rejection_sample_method": "block"
}
```

令 anchor 是已采样、尚待进入本轮目标计算的 token。在所分析的 `sample_from_anchor=true` 布局中：

```text
目标已处理历史的 hidden states → 草稿自己的 context KV
anchor + 3 个 mask → 五层草稿 decoder → 4 组 base logits
anchor → Markov 采样 Y1 → Y2 → Y3 → Y4
目标一次评分 [anchor, Y1, Y2, Y3, Y4]
block verifier → 接受连续前缀 + 一个纠正或 bonus token
```

这里 K=4 指候选数；正常情况下目标验证涉及 5 个输入位置，本轮新提交数为 $J+1$，其中 $0\le J\le4$。Anchor 已在上一轮输出，不能再作为本轮新增 token 重复计数；EOS 和输出预算可提前截断。

`probabilistic` 要求验证使用真实草稿分布 $q$，包括 Markov 修正；不能用未修正的 base logits 代替。概率公式还要求采样处理一致，详见 [[DSpark#8. Markov 分布怎样接入 block verification？]]。

`block` 表示联合验证算法，不能只用逐 token 的 $\min(1,p/q)$ 描述。它可继续评估较长前缀，最终仍提交连续前缀。

Checkpoint 带有 confidence head 也不意味着启用了自适应验证。[本文引用的 vLLM 配置默认值](https://github.com/vllm-project/vllm/blob/ab72c466aaac03524e9282c66cd00daafce647dc/vllm/config/speculative.py#L676)是 `enable_adaptive_verification=false`；前述专用后端快照还明确拒绝启用该功能。

## 7. 怎样判断这些选项是否改善推理？

以下是由机制得到的工程判断，不是 K3 的实测结论：

| 观察指标                           | 回答什么问题                    | 比较时要固定什么                    |
| ------------------------------ | ------------------------- | --------------------------- |
| 每 rank 的权重、目标缓存、草稿缓存和 KDA 状态占用 | DCP 节省了哪一部分容量？            | 模型、dtype、上下文和并发             |
| TTFT、prefill tokens/s          | 长输入和 prefix cache 是否成为瓶颈？ | 输入长度、cache hit/miss、排队条件    |
| TPOT/ITL、输出 tokens/s           | 用户生成速度和整体吞吐如何变化？          | 请求负载、输出长度、采样参数              |
| 平均接受草稿数与每轮输出数                  | 验证多出的候选是否值得？              | K、verifier、是否计入 bonus token |
| Draft/verify、通信、状态重放耗时         | 接受率增加是否抵消额外成本？            | 后端、batch 和并行布局              |

例如 K=4 时，平均每轮输出数为 $1+\mathbb E[J]$，不等于整体加速倍数：一轮还要支付起草、目标批量验证、通信和状态管理的成本。模型卡中 K=7 的接受长度也不能直接当成本配置的结果。

论文 §5.4.3 还强调 prefix-cache affinity 与按请求类别分配预算：百万 token 请求的 cache miss 成本和短请求不同。比较 TP/DP/DCP 时，也要记录实际请求落在哪个 DP 组，以及 prefix cache 的命中条件。

## 8. 一手资料与版本范围

- [K3 论文 v2](https://arxiv.org/abs/2607.24653v2)：§2.1 混合 attention，Table 1 架构，§4.1.4 QAT 与草稿微调，§5.1 KDA/KCP，§5.4.2 推测状态处理，§5.4.3 调度。
- [vLLM 并行配置](https://github.com/vllm-project/vllm/blob/a9dcfae9ef0756ce076d0053429d6c9afef067a3/vllm/config/parallel.py#L640)、[Kimi MLA](https://github.com/vllm-project/vllm/blob/a9dcfae9ef0756ce076d0053429d6c9afef067a3/vllm/models/kimi_k3/nvidia/mla.py#L219)、[专用后端限制](https://github.com/vllm-project/vllm/blob/a9dcfae9ef0756ce076d0053429d6c9afef067a3/vllm/models/kimi_k3/nvidia/specialized_decode/runtime/__init__.py#L125)：固定于 `a9dcfae9ef0756ce076d0053429d6c9afef067a3`。
- [K3.1 DSpark 模型卡](https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark-preview)、[DSpark 模型实现](https://github.com/vllm-project/vllm/blob/ab72c466aaac03524e9282c66cd00daafce647dc/vllm/models/kimi_k3/nvidia/dspark_mla.py)、[采样实现](https://github.com/vllm-project/vllm/blob/ab72c466aaac03524e9282c66cd00daafce647dc/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py)：DSpark 源码固定于 `ab72c466aaac03524e9282c66cd00daafce647dc`，与前一组快照分别说明；未验证二者与某个发行镜像完全对应。
- [DSpark 原论文](https://arxiv.org/abs/2607.05147)、[Block Verification 原论文](https://arxiv.org/abs/2403.10444)：方法定义和理论依据，不是 K3 论文的同一套实验。

两份概念笔记分别记录推导与适用条件。本仓库尚未提供该配置的可运行部署复现或 GPU 性能结果。
