---
title: MLA、TP/DP 与 Decode Context Parallelism
aliases:
  - MLA
  - Multi-head Latent Attention
  - Decode Context Parallelism
tags:
  - llm
  - inference
  - vllm
  - mla
  - context-parallelism
url: https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/
updated: 2026-11-12
---

# MLA、TP/DP 与 Decode Context Parallelism

整理日期：2026-11-12。本文区分 MLA 的数学结构、vLLM 的通用并行规则和 Kimi-K3.1-Pro 的具体后端限制。

关联论文：[[Kimi K3 - Open Frontier Intelligence]]。草稿模型的缓存与采样见 [[DSpark]]。

直接结论：**TP4、DP2 不强制要求 DCP4。**对于不启用 PCP 的通常 MLA 布局，DCP 可以在 TP 组内取不同分片度；DCP4 能消除 TP4 组内共享 latent 历史缓存的重复。通用参数校验允许 DCP1、DCP2、DCP4，但具体模型的计算后端还可能有额外限制。

下面默认 PP 和 PCP 均为 1；容量公式只讨论目标模型 MLA latent 历史缓存，忽略分页填充、量化元数据、DSA indexer cache、KDA 状态、草稿模型、Graph 与临时工作区。

## 1. MLA 为每个历史 token 保存什么？

K3 官方论文 §2.1.2 说明：MLA 将一个 token 的 key/value 表示压缩成低维 latent，多头共享该 latent；K3 的 MLA 使用 NoPE。以下保留归一化，省略输出门控和最终输出投影，因为后两者不改变这里讨论的缓存依赖。

设历史位置为 j，隐藏状态维度为 d，latent 维度为 r。压缩后的向量为：

$$
x_j\in\mathbb R^d,\qquad
W_D^{KV}\in\mathbb R^{r\times d},\qquad
c_j=\operatorname{Norm}\!\left(W_D^{KV}x_j\right)\in\mathbb R^r.
$$

论文采用省略归一化的简写。这里将实际缓存的归一化后向量记为 c；实现中的量化还可能改变其存储格式。

对第 h 个 attention head，有各自的 key/value 上投影矩阵：

$$
W_{UK}^{(h)}\in\mathbb R^{d_k\times r},\qquad
W_{UV}^{(h)}\in\mathbb R^{d_v\times r}.
$$

该 head 对应的 key 和 value 为：

$$
k_j^{(h)}=W_{UK}^{(h)}c_j,\qquad
v_j^{(h)}=W_{UV}^{(h)}c_j.
$$

关键在于：**上投影矩阵带 head 下标，缓存向量不带 head 下标。**每个 head 都从同一个 latent 向量读取信息，只是读取方式不同。

因此，长度为 T 的单层 latent 历史缓存可以写成：

$$
C_T=
\begin{bmatrix}
c_1^\top\\
c_2^\top\\
\vdots\\
c_T^\top
\end{bmatrix}
\in\mathbb R^{T\times r}.
$$

这里没有可按 head 拆分的维度。不同 MLA 层、不同请求各自有自己的缓存；“共享”指同一层、同一请求中的不同 heads 共享 latent。[K3 论文 §2.1.2](https://arxiv.org/abs/2607.24653v2)

## 2. 解码时怎样直接利用 latent，避免保存展开后的 K/V？

设当前查询位置为 t，第 h 个 head 的 query 为：

$$
q_t^{(h)}\in\mathbb R^{d_k}.
$$

先考虑 NoPE 的普通因果 attention。将 key 上投影代入打分公式：

$$
s_{tj}^{(h)}
=\frac{\left(q_t^{(h)}\right)^\top k_j^{(h)}}{\sqrt{d_k}}
=\frac{\left(q_t^{(h)}\right)^\top W_{UK}^{(h)}c_j}{\sqrt{d_k}}
=\frac{\left(\left(W_{UK}^{(h)}\right)^\top q_t^{(h)}\right)^\top c_j}{\sqrt{d_k}}.
$$

定义投影到 latent 空间的 query：

$$
\widetilde q_t^{(h)}=\left(W_{UK}^{(h)}\right)^\top q_t^{(h)}\in\mathbb R^r.
$$

于是，可以直接用这个 query 与历史 latent 做内积。缩放因子仍保留原来的 key 维度：

$$
s_{tj}^{(h)}=\frac{\left(\widetilde q_t^{(h)}\right)^\top c_j}{\sqrt{d_k}},\qquad
a_{tj}^{(h)}=\frac{\exp\!\left(s_{tj}^{(h)}\right)}{\sum_{u=1}^{t}\exp\!\left(s_{tu}^{(h)}\right)}.
$$

value 上投影也可以移到求和之后：

$$
o_t^{(h)}
=\sum_{j=1}^{t}a_{tj}^{(h)}v_j^{(h)}
=W_{UV}^{(h)}\left(\sum_{j=1}^{t}a_{tj}^{(h)}c_j\right).
$$

这样，在 decode 路径中不必把每个历史 token 的全部多头 K/V 都展开并长期存储。K3.1 使用 DSA 时，可将求和范围改为 indexer 选出的历史位置集合；多头共享 latent 的关系保持不变。其他采用 RoPE 的 MLA 变体还可能缓存额外位置分量，这里专门解释 K3/K3.1 目标模型的 NoPE 路径。

这段重排在 [Kimi MLA 源码](https://github.com/vllm-project/vllm/blob/a9dcfae9ef0756ce076d0053429d6c9afef067a3/vllm/models/kimi_k3/nvidia/mla.py#L444)中对应吸收 key 上投影的 `W_UK_T`，以及最后的 `W_UV`。

## 3. TP 拆分 heads，为什么每张卡还需要完整缓存？

设 attention 的 query head 总数为 H，TP 大小为 P，且 H 可被 P 整除。普通 TP 将 heads 分成 P 组，每张 GPU 计算：

$$
H_{\mathrm{local}}=\frac{H}{P}.
$$

例如 K3 论文中有 96 个 attention heads，TP8 时每张卡负责 12 个。但这 12 个 heads 的每个打分都需要整个历史 latent 向量；它们不能只读取 latent 的八分之一维度就完成内积。

```text
GPU0：第  0—11 个 heads 的计算 + 完整 C_T
GPU1：第 12—23 个 heads 的计算 + 完整 C_T
...
GPU7：第 84—95 个 heads 的计算 + 完整 C_T
```

示意中 head 编号仅用于表达分工。**TP 已经切分了 head 相关的计算与权重，而普通布局中的共享缓存仍被复制。**Kimi MLA 源码明确说明，Q/KV latent 在 TP ranks 上复制，后续上投影才做 TP 切分；缓存规格也将 latent 作为单个 KV head 处理。[实现依据](https://github.com/vllm-project/vllm/blob/a9dcfae9ef0756ce076d0053429d6c9afef067a3/vllm/models/kimi_k3/nvidia/mla.py#L219)

从理论上也可以另行设计沿 latent 维度切分的 attention，但那需要跨卡合并内积等结果，改变计算和通信组织。这里的“TP 下复制缓存”描述的是这一普通实现布局，不能推导成任何实现都必须复制。

若 b 为每个 latent 元素的存储字节数，单层、单请求缓存大小约为：

$$
M_{\mathrm{latent}}=Trb.
$$

只增加 TP、保持这个缓存布局时，每张卡仍约保存这么多 latent；整个 TP 组的物理存储总量为：

$$
M_{\mathrm{TP\ group}}\approx PTrb.
$$

MLA 的压缩节省和跨卡复制可以同时发生：先将每份历史表示变小，再在多个 ranks 上保存这份较小表示。

## 4. DCP 改为沿历史 token 维度分片

设 DCP 大小为 D。它把历史位置拆成 D 个互不相交的子集：

$$
\{1,\ldots,T\}=\bigsqcup_{s=0}^{D-1}\mathcal I_s.
$$

各 rank 保存本分片位置对应的完整 latent 向量，而非所有位置的部分 latent 维度。交错分片的示意是：

```text
DCP rank 0：c_1, c_5, c_9, ...
DCP rank 1：c_2, c_6, c_10, ...
DCP rank 2：c_3, c_7, c_11, ...
DCP rank 3：c_4, c_8, c_12, ...
```

实际实现还有页表与可配置交错粒度。为使每个 head 得到完整输出，实现需要交换 query、局部输出或其他数据；下面给出局部结果如何合并的数学解释，不指定某一种通信原语。

对固定的一个 query/head，暂时省略 t、h 下标。每个分片计算局部归一化常数与局部 latent 输出：

$$
Z_s=\sum_{j\in\mathcal I_s}\exp(s_j),\qquad
z_s=\frac{\sum_{j\in\mathcal I_s}\exp(s_j)c_j}{Z_s}.
$$

全局输出则为：

$$
z=\frac{\sum_{s=0}^{D-1}Z_s z_s}{\sum_{s=0}^{D-1}Z_s},\qquad
o=W_{UV}z.
$$

因此，局部 softmax 输出需要按归一化常数加权合并，不能简单取平均。实际实现通常采用稳定的 log-sum-exp 信息；没有有效历史位置的分片贡献为零，不计算上述除以零的局部表达式。稀疏 attention 将位置集合进一步限制到选中的 top-k。[K3.1 DCP 实现例子](https://github.com/vllm-project/vllm/blob/a9dcfae9ef0756ce076d0053429d6c9afef067a3/vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/dsa/dcp_stage.py)

对充分长且均匀分片的历史，单 rank 存储和整个 TP 组存储分别约为：

$$
M_{\mathrm{rank}}\approx\frac{Trb}{D},\qquad
M_{\mathrm{TP\ group}}\approx\frac{P}{D}Trb.
$$

当 D 等于 P 时，每个请求的 latent 历史在整个 TP 组中合起来只保留一份；其他取值对应剩余的重复。这里仅统计历史 latent，本模型的全部显存使用并不遵循同一个缩放公式。

## 5. TP4、DP2 时，DCP 如何选？

不启用 PCP 时，DCP 复用 TP 组的 ranks；vLLM 通用配置校验要求：

$$
1\le D\le P,\qquad P\bmod D=0.
$$

因此对于 MLA 的通用布局：

$$
P=4\quad\Longrightarrow\quad D\in\{1,2,4\}.
$$

[vLLM 通用规则](https://github.com/vllm-project/vllm/blob/a9dcfae9ef0756ce076d0053429d6c9afef067a3/vllm/config/parallel.py#L640)：这只是通用必要条件，还需满足模型与后端支持。

假设一条请求的完整 latent 历史占 8 GiB，且只观察负责该请求的一个 TP4 组：

| DCP | 每张 GPU 保存的 latent 历史 | TP4 组合计 | 全组等效缓存副本数 |
|---:|---:|---:|---:|
| 1 | 8 GiB | 32 GiB | 4 |
| 2 | 4 GiB | 16 GiB | 2 |
| 4 | 2 GiB | 8 GiB | 1 |

这是容量关系的假设算例。DCP4 更节省此类缓存，但增加分布式 attention 通信，不能单凭它更省缓存就认定总是最快。

DP2 则建立两组 attention 副本，接收不同请求。在 PP、PCP 均为 1 时，总 GPU 数为：

$$
N_{\mathrm{GPU}}=\mathrm{TP}\times\mathrm{DP}=4\times2=8.
$$

用 DCP4 时，可以画成：

```text
DP rank 0 / TP 组 A：GPU0、GPU1、GPU2、GPU3
  请求 A 的 heads 由四卡分工；A 的缓存再由 DCP4 分成四片。

DP rank 1 / TP 组 B：GPU4、GPU5、GPU6、GPU7
  请求 B 的 heads 由四卡分工；B 的缓存再由 DCP4 分成四片。
```

DP2 不会自动让一个请求跨两个 TP 组共享缓存。因此，不能把总 GPU 数 8 直接当成此布局的 DCP 大小。若同一前缀被分别路由到两个 DP ranks，两边也可能各自建立前缀缓存；跨 DP 复用需要额外机制。

**Attention 分组不能直接套用到全部 MoE 权重。**在这里核对的 vLLM 实现中，即使不启用 EP，routed experts 也可以按 TP×DP 组成更大的张量并行组；启用 EP 后，则改为在这些 ranks 之间分配完整专家。因此，TP4、DP2 下，attention 按两个 TP4 组处理请求，而 routed experts 可以在八个 ranks 上做 TP 或 EP。仅凭 DP2 不能断言整套模型权重复制两份。[官方说明](https://github.com/vllm-project/vllm/blob/a9dcfae9ef0756ce076d0053429d6c9afef067a3/docs/serving/expert_parallel_deployment.md#L82)、[MoE 分组实现](https://github.com/vllm-project/vllm/blob/a9dcfae9ef0756ce076d0053429d6c9afef067a3/vllm/model_executor/layers/fused_moe/config.py#L1168)。MLA 缓存的 TP/DCP 分组仍按本节前面的规则理解。

## 6. 应用于 Kimi-K3.1-Pro，还要看专用路径的限制

前面的 DCP1、2、4 是通用规则。读取的 NVIDIA `latency_specialized` 路径还包含更严格的启动检查：

$$
D\in\{1,P\}.
$$

也就是说，该路径允许关闭 DCP，或者使 DCP 覆盖整个 TP 组，**不接受中间的 DCP2 / TP4 组合**。同一专用路径中，TP4 还要求启用专家并行；TP 大于 1 的专家并行需要 MegaMoE，并有其他附加条件。[固定版本的具体检查](https://github.com/vllm-project/vllm/blob/a9dcfae9ef0756ce076d0053429d6c9afef067a3/vllm/models/kimi_k3/nvidia/specialized_decode/runtime/__init__.py#L125)

这些检查是必要条件，不是完整可运行保证。部署预设固定的测试镜像与这里引用的源码快照也可能不同。因此，对 TP4、DP2、DCP4 的部署应分成两个判断：

1. **并行原理上**：TP4、DP2 采用 DCP4，可以使每个四卡组内的共享 latent 缓存不再重复；采用较小 DCP 也有原理上的意义。
2. **K3.1 的实际部署上**：需要先确定镜像、MoE 的 TP/EP 布局、后端与显存预算。TP4、DP2 与 TP8、DP1 的 attention 布局不同；而 routed experts 可能在八卡之间分片。既不能把两种配置直接视为等价，也不能直接按两个四卡完整模型副本估算显存。

## 7. 适用范围与关联论文

上述公式描述 NoPE MLA 的 latent 历史和理想均匀分片，不是整个模型的显存预算，也不是某套硬件上的性能测量。分页、DSA indexer、KDA recurrent state、草稿缓存、权重量化和工作区都需另计。当前笔记没有附可复现的数值实验或 GPU benchmark。

- [[Kimi K3 - Open Frontier Intelligence]]：§2.1.2 使用 NoPE Gated MLA；论文中的 KDA 状态与这里的历史 latent 不是同一种缓存。
- [[DSpark]]：K3.1 草稿模块有自己的 MLA 缓存；不能将它直接视为目标模型缓存的一部分。
- [K3 论文](https://arxiv.org/abs/2607.24653v2)、[vLLM 上下文并行说明](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)及上文固定版本源码：分别支持架构、通用机制与后端约束。本文后端结论固定在 `a9dcfae9ef0756ce076d0053429d6c9afef067a3`，不代表任意镜像或未来版本。
