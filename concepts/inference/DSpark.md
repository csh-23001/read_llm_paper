---
title: DSpark：网络结构、Markov head 与验证算法
aliases:
  - DSpark 推测解码
  - DSpark-Markov-Head
tags:
  - llm
  - inference
  - vllm
  - speculative-decoding
  - dspark
arxiv: https://arxiv.org/abs/2607.05147
url: https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark-preview
updated: 2026-11-12
---

# DSpark：网络结构、Markov head 与验证算法

整理日期：2026-11-12。对应配置为 `method: dspark`、`num_speculative_tokens: 4`、`draft_sample_method: probabilistic`、`rejection_sample_method: block`，草稿仓库为 `Inferact/Kimi-K3.1-Pro-DSpark-preview`。

> [!info] 阅读范围
> 本文以 K3.1 的 Inferact checkpoint 解释 DSpark，通用部分包括半自回归起草、vanilla Markov 公式与 block verification；层数、维度和具体后端行为限定于所列模型及源码。K3 论文本身的部署方案见 [[Kimi K3 - Open Frontier Intelligence]]。

## 0. 先回答：这个“草稿模型”是不是只是一个 head？

如果把 head 泛指“附加在目标模型上的草稿模块”，可以这样称呼；但这份 checkpoint 的网络结构包含 **额外的五层 dense decoder**，不能把它等同于一个线性输出层、一个 Markov head，或一个单层 MTP。

这份草稿模块依赖目标模型的中间特征，且复用目标模型的 token embedding 和 LM head。它与目标模型在同一推理流程中协作，不是重新加载一份完整的 2.8T 目标模型。

| 部件                 | 作用                              | 是否额外的草稿参数     |
| ------------------ | ------------------------------- | ------------- |
| Target backbone    | 原来的 Kimi-K3.1-Pro；产生目标分布与上下文特征  | 主模型本身         |
| Context projection | 融合选定目标层的 hidden states          | 是             |
| Draft backbone     | 五层 dense MLA decoder，生成各候选位置的表示 | 是             |
| Token embedding    | 将 anchor/mask token 转为向量        | 与目标模型对应组件共享   |
| LM head            | 将草稿表示投影到词表，得到 base logits       | 与目标模型对应组件共享   |
| Markov head        | 根据前一个 token 修正当前候选的 logits      | 是             |
| Confidence head    | 预测接受相关的置信度，供可选的自适应验证使用          | 是；是否用于推理由配置决定 |

[发布方模型卡](https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark-preview)给出的额外草稿规模约为 2.7B、BF16。[vLLM 类实现](https://github.com/vllm-project/vllm/blob/ab72c466aaac03524e9282c66cd00daafce647dc/vllm/models/kimi_k3/nvidia/dspark_mla.py#L147)实际创建 decoder 层列表和两个附加 heads，并声明不拥有独立 embedding/LM head；[加载逻辑](https://github.com/vllm-project/vllm/blob/ab72c466aaac03524e9282c66cd00daafce647dc/vllm/v1/worker/gpu/spec_decode/dspark/utils.py#L112)将目标组件共享给它。

不要混淆两种“共享”：**草稿与目标共享各自对应的 embedding 和 LM head**，不等于 **embedding 与 LM head 彼此绑成同一矩阵**。这里记录的配置为 `tie_word_embeddings: false`（待按固定 revision 复核）。

**K3** 官方论文 §4.1.4 描述的是“将单层 MTP 微调为 EAGLE-3 风格草稿”的方案；本笔记研究的是 **Inferact 的 K3.1 DSpark checkpoint**，其架构不同。vLLM 加载时使用 `dspark_head` 这个内部标签，也不能据此判断它只有一层。[K3 论文](https://arxiv.org/abs/2607.24653v2)

## 1. 配置快照与记号

以下按 checkpoint revision `6a2ef20a05efe88aabbf9ecaa12d020c3b0edf94` 记录配置尺寸，不将该 revision 视为当前 `main`。参考[固定 revision 文件树](https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark-preview/tree/6a2ef20a05efe88aabbf9ecaa12d020c3b0edf94)与[配置](https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark-preview/blob/6a2ef20a05efe88aabbf9ecaa12d020c3b0edf94/config.json)。

> [!note] 配置核对范围
> 发布方模型卡和关键 vLLM 源码已核对；固定 revision 的 `config.json` 本次未能取回。模型卡可确认五层、2.7B、MLA、7168/14336、五个目标层、Markov rank 512、共享 embedding/LM head 等主要结构。词表精确大小、目标层编号、`tie_word_embeddings`、vanilla/anchor/Markov-confidence 等配置细项仍需按该文件复核；下文相关推导以列出的配置为前提。

| 符号/配置 | 含义                  |                         本 checkpoint |
| ----- | ------------------- | -----------------------------------: |
| V     | 词表大小                |                               163840 |
| d     | 草稿 hidden dimension |                                 7168 |
| dₜ    | 目标 hidden dimension |                                 7168 |
| m     | 草稿 FFN 中间维度         |                                14336 |
| S     | 提取的目标层数量            |                                    5 |
| N     | 草稿 decoder 层数       |                                    5 |
| r     | Markov 低秩维度         |                                  512 |
| K     | 本轮实际起草长度            | 参数选择 4；checkpoint 的 `block_size` 为 7 |
| a     | 已知的 anchor token    |                  通常为上一轮最终采样得到的 token |
| Yᵢ    | 本轮第 i 个候选 token     |                                 随机变量 |
| τ     | 采样 temperature      |                         本文概率公式先假设大于零 |

待按固定 revision 复核的配置细项：`markov_head_type: vanilla`、`sample_from_anchor: true`、`enable_confidence_head: true`、`confidence_head_with_markov: true`；选定目标层的配置编号为 `[75,79,83,87,91]`。这里保留配置编号，不擅自转换成另一种层编号惯例。

数学公式先忽略量化、浮点舍入和额外词表截断；如使用 top-p、禁用 token 或其他处理，验证必须使用实际起草/目标采样所对应的分布。本文 vLLM 源码快照为 `ab72c466aaac03524e9282c66cd00daafce647dc`，不代表所有固定镜像都包含同一实现。

## 2. 一轮 DSpark 的完整数据流

```text
目标模型已经计算过的上下文 hidden states
               ↓ 选出五个层，拼接并投影
        草稿自己的 context KV
               ↓
anchor + 三个 mask → 五层 draft decoder → 四个 hidden states
                                                ↓ 共享 LM head
                                         四组 base logits
                                                ↓
anchor → Markov → Y1 → Markov → Y2 → Markov → Y3 → Markov → Y4
                                                ↓
                              目标模型批量评分 + block verification
```

### 2.1 目标特征不是“再跑一次完整目标模型”

设目标已经处理的某个上下文位置为 j，从 S 个层提取特征，先拼接并投影，再用 [[RMSNorm]] 对得到的 $d$ 维向量做归一化：

$$
f_j=
\operatorname{RMSNorm}\!\left(
W_c
\begin{bmatrix}
H_j^{(\ell_1)}\\
\vdots\\
H_j^{(\ell_S)}
\end{bmatrix}
\right),
\qquad
W_c\in\mathbb R^{d\times Sd_t},
\qquad
f_j\in\mathbb R^d.
\tag{1}
$$

然后用草稿各层自己的投影，将这些特征转换为该层需要的 context KV。概念上可以写为：

$$
K^{(l)}=[K_{\mathrm{ctx}}^{(l)};K_{\mathrm{query}}^{(l)}],
\qquad
V^{(l)}=[V_{\mathrm{ctx}}^{(l)};V_{\mathrm{query}}^{(l)}].
\tag{2}
$$

式 (2) 是展开视角的示意；实际 MLA 还采用 latent 存储。**这是由目标 hidden states 构建草稿自己的 KV，不是直接复制目标所有层的 KV。**实现可跨层融合这些投影，但不改变此依赖关系。[上下文投影与缓存实现](https://github.com/vllm-project/vllm/blob/ab72c466aaac03524e9282c66cd00daafce647dc/vllm/models/kimi_k3/nvidia/dspark_mla.py#L227)

### 2.2 Anchor 与候选位置的对齐

为避免差一位，设目标已处理历史为 x₁ 到 xₙ，而 anchor a 已采样、尚待进入下一轮目标计算。下一批草稿预测的是 anchor 后面的 token：

$$
a=x_{n+1},
\qquad
Y_i\ \text{对应位置}\ n+1+i,
\qquad i=1,\ldots,K.
\tag{3}
$$

因为 `sample_from_anchor=true`，K 个草稿 query 输入为：

$$
U^{(0)}
=
\begin{bmatrix}
E(a)^\top\\
E(\mathrm{mask})^\top\\
\vdots\\
E(\mathrm{mask})^\top
\end{bmatrix}
\in\mathbb R^{K\times d}.
\tag{4}
$$

对于 K 为 4，就是 `[anchor, mask, mask, mask]`。第一个 query 使用 anchor 的表示预测 Y₁；后面依次预测 Y₂、Y₃、Y₄。

位置和 attention mask 也参与计算；这里没有额外假设使用加法式位置 embedding。本 checkpoint 配置声明草稿 full attention 为 causal，不能将原论文其他实例的双向 block mask 直接照搬过来。即使 query attention 是 causal，它看到的也是 anchor 和占位 mask，而非本轮尚未采出的真实候选 token。

### 2.3 五层骨干一次并行计算所有候选表示

将上下文、anchor、位置等固定条件合记为 C。五层 decoder 产生：

$$
(g_1,\ldots,g_K)
=F_{\mathrm{draft}}\!\left(U^{(0)},\mathrm{KV}_{\mathrm{ctx}},\mathrm{positions},\mathrm{mask}\right),
\qquad g_i\in\mathbb R^d.
\tag{5}
$$

这里的“并行”指同一 forward 内同时计算 K 个 query 位置；五个 decoder 层之间仍按深度顺序执行。由于本轮真实候选尚未采样，gᵢ 不会随着随后采出的 Y₁、Y₂ 动态重算。

共享目标的 LM head。采用列向量记法，设其有效投影矩阵为：

$$
L\in\mathbb R^{V\times d},
\qquad
b_i=L g_i\in\mathbb R^V.
\tag{6}
$$

bᵢ 是第 i 个位置的 **base logits**。这里将可能的固定输出缩放并入有效投影；本配置不额外设置该缩放。

## 3. Markov head 的实际数学结构

### 3.1 两组权重与一次 lookup

为对应 PyTorch 中的实际权重形状，定义：

$$
A\in\mathbb R^{V\times r}
\quad(\texttt{markov\_w1.weight}),
\qquad
B\in\mathbb R^{V\times r}
\quad(\texttt{markov\_w2.weight}).
\tag{7}
$$

若前一个 token 是 u，先取其 r 维 Markov embedding，再投影到词表：

$$
m(u)=A_{u,:}^{\top}\in\mathbb R^r,
\qquad
\delta(u)=B\,m(u)\in\mathbb R^V.
\tag{8}
$$

具体到候选词 v：

$$
\delta(v\mid u)
=B_{v,:}A_{u,:}^{\top}
=\sum_{s=1}^{r}A_{u,s}B_{v,s}.
\tag{9}
$$

这是 vanilla head 的完整核心：**查表，再做线性投影**，没有额外 GRU 状态，也没有在这两步之间插入 Transformer。[DSparkMarkovHead 源码](https://github.com/vllm-project/vllm/blob/ab72c466aaac03524e9282c66cd00daafce647dc/vllm/model_executor/models/qwen3_dspark.py#L49)

注意 Markov embedding A 与共享的 token embedding E 是两套不同参数：前者维度为 512，后者为 7168；Markov 投影 B 也不是共享 LM head L。

### 3.2 加到 base logits，再做 softmax

定义 Y₀ 为 anchor：

$$
Y_0=a.
\tag{10}
$$

第 i 个候选的最终 logits 为：

$$
\lambda_i(v\mid C,Y_{i-1})
=b_i(v;C)+\delta(v\mid Y_{i-1}).
\tag{11}
$$

在本文仅考虑 temperature 的情形，实际起草分布为：

$$
q_i(v\mid C,Y_{i-1})
=
\frac{
\exp\!\left(\lambda_i(v\mid C,Y_{i-1})/\tau\right)
}{
\sum_{w\in\mathcal V}
\exp\!\left(\lambda_i(w\mid C,Y_{i-1})/\tau\right)
}.
\tag{12}
$$

随后从该分布采样 Yᵢ，并将其作为下一位置的 previous token。实现先算出所有 base logits，再循环执行 Markov lookup、bias、采样和更新 previous token。[顺序采样循环](https://github.com/vllm-project/vllm/blob/ab72c466aaac03524e9282c66cd00daafce647dc/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L272)

### 3.3 为什么称为 Markov？

固定 C 后，这个 vanilla head 对本轮已采样候选的直接依赖满足：

$$
q_i(Y_i\mid C,Y_1,\ldots,Y_{i-1})
=q_i(Y_i\mid C,Y_{i-1}).
\tag{13}
$$

它在候选链内部是一阶 Markov 结构。不过，**整个草稿并没有丢掉长上下文**：完整目标历史仍通过 context KV、anchor 和五层 draft backbone 影响 bᵢ。

更准确地说，这是给定上下文的、随候选位置变化的 Markov 链。转移 bias 在各位置复用，但 bᵢ 随位置和上下文变化，所以同一个前一 token 在不同语境下不会强制得到同一个下一 token 分布。

当前配置的 `vanilla` 没有显式记住候选链中更早的 token。其他 DSpark 变体可以使用 gated/RNN head，但不能将其状态更新公式当成这个 checkpoint 的公式。[DSpark 变体说明](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dspark.md)

## 4. 联合分布、因果性与“半自回归”

### 4.1 四 token 的具体分解

对于本配置：

$$
q(y_1,y_2,y_3,y_4\mid C)
=q_1(y_1\mid C,a)
q_2(y_2\mid C,y_1)
q_3(y_3\mid C,y_2)
q_4(y_4\mid C,y_3).
\tag{14}
$$

没有 Markov 修正、仅逐位置独立采样时，联合分布会变为：

$$
q_{\mathrm{ind}}(y_{1:K}\mid C)
=\prod_{i=1}^{K}q_i^{(0)}(y_i\mid C),
\qquad
q_i^{(0)}=\operatorname{softmax}(b_i/\tau).
\tag{15}
$$

DSpark 在保持重骨干并行的同时，让最终采样依赖前一个真实采出的 token。因此，“半自回归”并非把 token 完全独立地一次采完，也不是每个候选都重跑五层草稿网络。

### 4.2 归一化来自逐步条件分布

式 (14) 对全部候选序列求和，先对最后一个位置求和，其条件概率和为 1，再递归向前，因此：

$$
\sum_{y_{1:K}}q(y_{1:K}\mid C)=1.
\tag{16}
$$

一个容易误写的地方是把它当成全局归一化的链式能量模型。令：

$$
M=AB^\top\in\mathbb R^{V\times V},
\qquad
Z_i(u;C)=\sum_v\exp\!\left((b_i(v;C)+M_{u,v})/\tau\right).
\tag{17}
$$

真实联合分布为：

$$
q(y_{1:K}\mid C)
=
\frac{
\exp\!\left(
\sum_{i=1}^{K}[b_i(y_i;C)+M_{y_{i-1},y_i}]/\tau
\right)
}{
\prod_{i=1}^{K}Z_i(y_{i-1};C)
}.
\tag{18}
$$

分母通常依赖该候选路径；不能随意替换成一个与路径无关的全局常数。DSpark 以因果条件分布的乘积定义联合分布。[原论文 §3.1](https://arxiv.org/html/2607.05147v1#S3.SS1)

### 4.3 可读的伪代码

```python
# 为突出数学关系，省略 batch、TP、缓存、舍入与额外采样处理。
hidden = draft_backbone(anchor_and_masks, context_kv)  # [K, d]
base = shared_lm_head(hidden)                         # [K, V]
prev = anchor
for i in range(K):
    markov_embedding = A[prev]                        # [r]
    logits = base[i] + B @ markov_embedding           # [V]
    q = softmax(logits / temperature)
    y = sample_categorical(q)
    draft_tokens[i] = y
    cached_draft_logits[i] = logits
    prev = y
```

**验证时必须使用 Markov 修正后的实际草稿分布。**如果实际按式 (12) 采样，却把仅来自 bᵢ 的分布交给概率比/纠正算法，就无法沿用其分布保持保证。实现缓存修正后的 logits，再按采样规则计算概率。

## 5. 低秩转移偏置究竟学到了什么？

### 5.1 低秩的是 logits bias 矩阵

由式 (17)：

$$
\operatorname{rank}(M)\le r=512.
\tag{19}
$$

M 的一行描述“前一个 token 为 u 时，对所有下一 token 的偏好修正”。A 学习 token 作为前驱时的低维特征，B 学习 token 作为后继时如何响应这些特征。

M 可以正也可以负，行和不受约束，它本身不是概率转移矩阵。真正的概率转移由式 (12) 的 softmax 产生。指数和归一化是非线性运算，因此不能将式 (19) 直接套到概率转移矩阵上。

### 5.2 从赔率看它的作用

固定前驱 u，对两个后继候选 v、w：

$$
\frac{q_i(v\mid C,u)}{q_i(w\mid C,u)}
=
\frac{q_i^{(0)}(v\mid C)}{q_i^{(0)}(w\mid C)}
\exp\!\left(\frac{M_{u,v}-M_{u,w}}{\tau}\right).
\tag{20}
$$

因此，加性 logits 修正等价于对基础分布乘上一个与前驱有关的正因子，然后重新归一化：

$$
q_i(v\mid C,u)
=
\frac{
q_i^{(0)}(v\mid C)\exp(M_{u,v}/\tau)
}{
\sum_wq_i^{(0)}(w\mid C)\exp(M_{u,w}/\tau)
}.
\tag{21}
$$

它可以在不重跑草稿骨干的情况下，调高符合当前分支的后继，压低属于另一种续写分支的后继。

### 5.3 一个多模态续写的假设算例

将四个符号视为四个 token，合理续写有两条：“of course”和“no problem”。假设第一位置均匀选择 of/no，第二位置的基础分布均匀选择 course/problem。

独立采样会给四种组合各 1/4 的概率，因此跨分支组合占一半。若设置兼容转移的 bias 为 log 9、不兼容为 0，temperature 为 1，则：

$$
q_2(\mathrm{course}\mid\mathrm{of})
=q_2(\mathrm{problem}\mid\mathrm{no})
=\frac{9}{9+1}=0.9.
\tag{22}
$$

跨分支组合的概率降为 0.1。这是原理示意，不是 Kimi 的实测句子或接受率。真实 tokenizer 也未必将这些词各编码为一个 token。

### 5.4 一阶 Markov 的能力边界

固定同一上下文、同一候选位置，如果两段不同的候选前缀结束在同一个 token 上，vanilla head 会给下一位置同一个修正分布：

$$
y_{i-1}=y'_{i-1}
\ \Longrightarrow\
q_i(\cdot\mid C,y_{<i})
=q_i(\cdot\mid C,y'_{<i}).
\tag{23}
$$

因此它无法仅靠这条动态通路区分所有更早的候选差异；完整自回归草稿或 recurrent head 可以更有表达力，但增加串行成本。式 (23) 说的是同一轮、固定骨干输出之后的能力边界，不是说模型对整段历史只保留一个 token。

## 6. 如何训练这些参数？哪些结论适用于当前 checkpoint？

下面参考原始 DSpark / DeepSpec 的公开目标，解释学习机制；没有据此声称 Inferact 这份 r2 checkpoint 的所有训练超参数、数据与实现都已公开或已核对。

### 6.1 交叉熵与 Markov 参数梯度

固定一个位置、前驱 u 与监督标签 y*，使用无额外截断的 softmax：

$$
\mathcal L_i=-\log q_i(y^*\mid C,u).
\tag{24}
$$

设词表误差向量为：

$$
e_i=q_i-\operatorname{onehot}(y^*)\in\mathbb R^V.
\tag{25}
$$

对未除 temperature 的 logits：

$$
\frac{\partial\mathcal L_i}{\partial\lambda_i}
=\frac{e_i}{\tau}.
\tag{26}
$$

利用式 (8) 到 (11) 的链式法则：

$$
\frac{\partial\mathcal L_i}{\partial B}
=\frac{1}{\tau}e_i\,m(u)^\top,
\qquad
\frac{\partial\mathcal L_i}{\partial A_{u,:}^{\top}}
=\frac{1}{\tau}B^\top e_i.
\tag{27}
$$

其他没有被 lookup 的 A 行，在此单位置损失中梯度为零；多位置和 batch 中的梯度再求和。骨干则通过 base logits 得到：

$$
\frac{\partial\mathcal L_i}{\partial g_i}
=\frac{1}{\tau}L^\top e_i.
\tag{28}
$$

共享 LM head 即使被冻结，也仍能将梯度传回草稿 hidden states。这里对前驱 token 当作给定离散条件求导，不是在对离散采样操作本身求普通梯度。

### 6.2 分布匹配为什么有利于接受？

对固定前缀，标准逐 token 拒绝采样的平均接受率为：

$$
\bar\alpha_i
=\sum_v q_i(v)\min\!\left(1,\frac{p_i(v)}{q_i(v)}\right)
=\sum_v\min(p_i(v),q_i(v))
=1-\frac12\|p_i-q_i\|_1.
\tag{29}
$$

最后一步来自逐元素恒等式：

$$
\min(p,q)=\frac{p+q-|p-q|}{2}.
\tag{30}
$$

因此，可以联合优化交叉熵、目标/草稿分布的 L1 距离，以及 confidence 预测。DeepSpec 中相应项可概括为：

$$
\mathcal L
=\lambda_{\mathrm{CE}}\mathcal L_{\mathrm{CE}}
+\lambda_{\mathrm{L1}}\mathcal L_{\mathrm{L1}}
+\lambda_{\mathrm{conf}}\mathcal L_{\mathrm{conf}},
\tag{31}
$$

$$
\mathcal L_{\mathrm{CE}}
=-\sum_iw_i\log q_i(y_i^*),
\qquad
\mathcal L_{\mathrm{L1}}
=\sum_iw_i\|p_i-q_i\|_1.
\tag{32}
$$

这里的正权重 wᵢ 可强调较早位置；具体设置属于训练配方。[DeepSpec 损失实现](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/modeling/dspark/loss.py)

式 (29) 直接解释标准逐 token 接受率；不要把它未经推导地当成下面 block verification 的每个联合子块阈值。

## 7. Confidence head 与 Markov head 的区别

Markov head 输出 V 个修正 logits，改变草稿的 token 分布。Confidence head 输出一个标量，预测验证相关的接受置信度。对于当前配置的 hidden/Markov 拼接：

$$
\widehat c_i
=\sigma\!\left(
w_h^\top g_i+w_m^\top m(Y_{i-1})+b_c
\right).
\tag{33}
$$

公开训练目标使用式 (29) 的 soft label：

$$
c_i^*=\operatorname{stopgrad}\!\left(1-\frac12\|p_i-q_i\|_1\right),
\tag{34}
$$

$$
\mathcal L_{\mathrm{conf}}
=-\sum_iw_i\left[
c_i^*\log\widehat c_i+(1-c_i^*)\log(1-\widehat c_i)
\right].
\tag{35}
$$

它不直接替代目标验证。原始 DSpark 还包含利用 confidence 选择验证长度的调度机制，但本笔记的配置没有启用 `enable_adaptive_verification`；引用的 vLLM 快照默认该选项为 false，只有启用后才使用对应 confidence 信号。[运行条件](https://github.com/vllm-project/vllm/blob/ab72c466aaac03524e9282c66cd00daafce647dc/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L177)、[配置默认值](https://github.com/vllm-project/vllm/blob/ab72c466aaac03524e9282c66cd00daafce647dc/vllm/config/speculative.py)

即使 checkpoint 内有 confidence 参数，也不能因此认定这条命令已经启用了论文里的完整自适应调度。尤其联合 block verification 的前缀接受概率，与标准逐 token confidence 的简单连乘需要分别分析。

## 8. Markov 分布怎样接入 block verification？

### 8.1 主模型评分与两种条件分布

将已确定历史与 anchor 合记为 c，目标模型分布为：

$$
p_i(v)=p_{\mathrm{target}}(v\mid c,y_{<i}).
\tag{36}
$$

草稿的实际分布为式 (12)，条件依赖可以简写为：

$$
q_i(v)=q_i(v\mid C,y_{i-1}).
\tag{37}
$$

两者的表达能力可以不同；分布保持的验证不要求 q 等于 p，要求用对实际的 q。

主模型批量处理 `[anchor,y1,y2,y3,y4]`，得到候选路径上的 p₁ 到 p₅。因果 mask 保证 pᵢ 的计算只依赖相应前缀；更靠后的候选作为输入出现在同一 forward 中，不会泄漏到较早位置的条件分布。

### 8.2 联合 block 验证的累计比值

下面以固定 K、无提前 EOS、非贪心采样为前提，重写 vLLM 对应的 block 算法。设：

$$
\rho_0=1,\qquad
\rho_i=
\min\!\left(
1,\rho_{i-1}\frac{p_i(y_i)}{q_i(y_i)}
\right),\qquad i=1,\ldots,K.
\tag{38}
$$

生成出的候选有正的草稿概率，因此该路径上的分母可定义。注意每一步都截断到 1；这一般不同于只对完整概率比乘积在末尾截断一次。

实际实现采用 log 空间以避免长乘积下溢：

$$
\log\rho_i
=\min\!\left(
0,\log\rho_{i-1}+\log p_i(y_i)-\log q_i(y_i)
\right).
\tag{39}
$$

对 i 从 0 到 K−1，定义下一位置的残差质量：

$$
Z_i=
\sum_v\left[\rho_i p_{i+1}(v)-q_{i+1}(v)\right]_+,
\qquad [x]_+=\max(x,0).
\tag{40}
$$

对每个可能的非空候选子块，联合验证阈值为：

$$
\beta_i=
\begin{cases}
\dfrac{Z_i}{Z_i+1-\rho_i},&1\le i<K,\\[6pt]
\rho_K,&i=K.
\end{cases}
\tag{41}
$$

如果第一个分支的分母为零，按实现取阈值 1。独立抽取均匀变量后，选择满足阈值检验的最长子块：

$$
U_i\sim\operatorname{Uniform}(0,1),
\qquad
J=\max\!\left(\{0\}\cup\{i:U_i\le\beta_i\}\right).
\tag{42}
$$

这里即使某个较短子块没有通过检验，也继续评估更长子块；最终仍提交连续前缀，而不是跳着保留 token。

### 8.3 纠正与额外 token

如果最终接受长度小于 K，追加一个纠正 token，其分布为：

$$
r_J(v)=
\frac{
\left[\rho_Jp_{J+1}(v)-q_{J+1}(v)\right]_+
}{
Z_J
}.
\tag{43}
$$

如果 J 等于 K，追加 token 则从目标分布 pₖ₊₁ 采样。零残差质量的分支在精确算法中不应以正概率进入；实际实现还有数值与边界处理。正常情况下本轮提交：

$$
(y_1,\ldots,y_J,z),\qquad L_{\mathrm{out}}=J+1.
\tag{44}
$$

式 (38) 到 (43) 对应 [vLLM 拒绝采样 kernel](https://github.com/vllm-project/vllm/blob/ab72c466aaac03524e9282c66cd00daafce647dc/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py)和 [Block Verification 原论文 Algorithm 2](https://arxiv.org/html/2403.10444v3#S3)。这部分引用算法的分布保持结果；本笔记没有重写其完整定理证明。

一个重要联系是：Markov 不仅影响候选 yᵢ，也影响用于累计比值和残差计算的 **整个 qᵢ 向量**。这就是 `probabilistic` 起草要保留真实草稿分布信息的原因。

## 9. 参数量、共享与串行开销

### 9.1 Markov head 低秩，但不是“零成本的小层”

两组矩阵的参数量为：

$$
N_{\mathrm{Markov}}=2Vr
=2\times163840\times512
=167772160.
\tag{45}
$$

若全部按 BF16 存储，仅这些权重就是：

$$
M_{\mathrm{Markov}}
=167772160\times2\ \mathrm{bytes}
=320\ \mathrm{MiB}.
\tag{46}
$$

如果直接保存完整的 token-to-token bias 矩阵：

$$
V^2=26843545600\ \text{个参数},
\qquad
M_{\mathrm{full,BF16}}=50\ \mathrm{GiB}.
\tag{47}
$$

低秩存储缩小的比例为：

$$
\frac{V^2}{2Vr}=\frac{V}{2r}=160.
\tag{48}
$$

这些是由配置尺寸推得的参数量，不包含框架工作区。当前通用 head 将这两组权重复制在 TP ranks 上，以减少每个顺序位置的跨卡同步；融合 kernel 可以优化执行与访存，但不会消除候选之间的逻辑依赖。[权重布局说明](https://github.com/vllm-project/vllm/blob/ab72c466aaac03524e9282c66cd00daafce647dc/vllm/model_executor/models/qwen3_dspark.py#L49)

### 9.2 为什么叫“轻量”？

对一个请求、一个位置，Markov 主要投影的乘加次数约为 Vr；共享 LM head 的基础投影约为 Vd：

$$
\frac{Vr}{Vd}=\frac{r}{d}=\frac{512}{7168}=\frac1{14}.
\tag{49}
$$

而且前者没有再跑 attention 和 FFN。这里是算术规模比较，不是实际耗时必然为十四分之一；串行路径的访存与启动开销仍可能显著。

整个起草阶段的时间可以概括为：

$$
T_{\mathrm{draft}}
\approx
T_{\mathrm{backbone}}(K)
+T_{\mathrm{LMhead}}(K)
+\sum_{i=1}^{K}T_{\mathrm{Markov},i}.
\tag{50}
$$

前两项可以对 K 个位置批量执行，最后一项需要沿候选链顺序推进。融合采样 kernel 主要降低最后一项的执行开销。

### 9.3 为什么五层草稿绝不只是一个输出 head？

仅五个标准 SwiGLU FFN 的三个矩阵，参数量就约为：

$$
5\times3dm
=5\times3\times7168\times14336
=1541406720.
\tag{51}
$$

这还没有计算五层 MLA、context projection 和 Markov 参数，已经超过 1.5B。约 2.7B 的附属草稿规模与此相符。

共享 embedding 或 LM head 的一个完整矩阵，各有：

$$
Vd=163840\times7168=1174405120
\tag{52}
$$

个参数。共享它们可以避免另存一套对应矩阵，但不使五层 decoder 自动消失，也不意味着其额外参数包含全部目标模型权重。

## 10. 适用范围与研究边界

公式是浮点误差、额外采样处理和 EOS 边界之外的数学说明；参数量由配置尺寸推得，不等于运行时显存占用。当前笔记没有附可复现的数值实验、实际 GPU kernel 测试或性能 benchmark。Block verification 的一般分布保持结论引用其原论文，不由本笔记重新证明。

研究时尤其应分别记录：

1. 当前 checkpoint 的 vanilla Markov 结构，和其他 gated/RNN 变体。
2. 真正的 decoder/LM/Markov 参数，和“draft head”这种宽泛命名。
3. 草稿前向的并行性，和最后 token 采样链的串行性。
4. 实际采样分布 q，和未经 Markov 修正的 base 分布。
5. 训练用的接受率代理，和实际 block verifier 的联合决策。
6. 模型具备 confidence head，和运行时确实启用了自适应调度。

## 11. 关联笔记与主要一手资料

- [[Kimi K3 - Open Frontier Intelligence]]：区分 K3 论文中的单层 EAGLE-3 风格草稿与 K3.1 的 DSpark 实现。
- [[MLA-TP-DP-DCP]]：目标与草稿的 latent 历史缓存、TP 复制和 DCP 分片。

- [K3.1 DSpark 发布方模型卡](https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark-preview)及[配置](https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark-preview/blob/6a2ef20a05efe88aabbf9ecaa12d020c3b0edf94/config.json)：核对具体 checkpoint。
- [DSpark 原论文](https://arxiv.org/abs/2607.05147)：半自回归结构、置信度和训练背景；其他模型实例的配置不能自动套用于 Kimi。
- [DeepSpec 官方代码](https://github.com/deepseek-ai/DeepSpec)：通用训练与 Markov 实现。
- 上文固定版本的 vLLM 源码：核对该配置的实际接入、采样和验证路径。
- [Block Verification 原论文](https://arxiv.org/abs/2403.10444)：联合验证的算法与理论结果。
