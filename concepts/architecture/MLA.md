---
title: Multi-head Latent Attention
aliases:
  - Multi-head Latent Attention
  - 多头潜在注意力
tags:
  - llm
  - architecture
  - attention
  - mla
arxiv: https://arxiv.org/abs/2405.04434
---

# MLA：Multi-head Latent Attention

MLA 为每个历史 token 保存一份各个 heads 共享的低维 latent，再用各 head 的投影读取其中的 key/value 信息，从而减少 KV cache。

下面先推导不带显式位置编码的 [[NoPE]] 版本，再补充 DeepSeek-V2 风格的解耦 RoPE。公式只讨论单层、单请求，向量均为列向量，省略 bias、量化和并行通信。矩阵重排在实数运算下等价，不保证不同浮点 kernel 的结果逐位相同。

## 1. 符号与普通多头注意力

| 符号 | 含义与维度 |
|---|---|
| $x_j\in\mathbb R^d$ | 历史位置 $j$ 的输入特征 |
| $t$、$H$ | 当前 query 位置、attention head 数；$h=1,\ldots,H$ |
| $d_k$、$d_v$ | 每个 head 的内容 key/query 维度、value 维度 |
| $r$、$r_q$ | KV latent 维度、query latent 维度；两者不必相同 |
| $q_t^{(h)},k_j^{(h)}\in\mathbb R^{d_k}$ | 第 $h$ 个 head 的 query 和 key |
| $v_j^{(h)},o_t^{(h)}\in\mathbb R^{d_v}$ | 第 $h$ 个 head 的 value 和聚合结果 |

普通 MHA 为每个 head 单独计算：

$$
q_t^{(h)}=W_Q^{(h)}x_t,\qquad
k_j^{(h)}=W_K^{(h)}x_j,\qquad
v_j^{(h)}=W_V^{(h)}x_j,
\tag{1}
$$

其中 $W_Q^{(h)},W_K^{(h)}\in\mathbb R^{d_k\times d}$，$W_V^{(h)}\in\mathbb R^{d_v\times d}$。长度为 $T$ 的历史需要缓存约 $TH(d_k+d_v)$ 个 K/V 元素；query 随当前计算产生，不属于通常的历史 KV cache。

## 2. MLA 怎样压缩 K/V？

### 2.1 一次下投影，得到共享 latent

$$
c_j=\mathcal N_{KV}\!\left(W_D^{KV}x_j\right)\in\mathbb R^r,
\qquad W_D^{KV}\in\mathbb R^{r\times d}.
\tag{2}
$$

$D$ 表示 down-projection，通常 $r\ll H(d_k+d_v)$。$\mathcal N_{KV}$ 表示 latent 上的归一化：Kimi 的实现使用 [[RMSNorm]]；省略该操作的数学简写可令它为恒等映射。这里将归一化后的结果统一记为 $c_j$。

### 2.2 各 head 从同一 latent 读取自己的 K/V

$$
k_j^{(h)}=W_{UK}^{(h)}c_j,\qquad
v_j^{(h)}=W_{UV}^{(h)}c_j,
\tag{3}
$$

$$
W_{UK}^{(h)}\in\mathbb R^{d_k\times r},\qquad
W_{UV}^{(h)}\in\mathbb R^{d_v\times r}.
$$

$U$ 表示 up-projection。**$c_j$ 没有 head 下标，而上投影矩阵有**：共享 latent 不代表所有 heads 使用相同的 K/V。不同层、不同请求仍各有自己的 latent。

把全部 heads 的 K/V 堆起来，更容易看到“联合压缩”的含义：

$$
\begin{bmatrix}
k_j^{(1)}\\ \vdots\\ k_j^{(H)}\\
v_j^{(1)}\\ \vdots\\ v_j^{(H)}
\end{bmatrix}
=
\begin{bmatrix}
W_{UK}^{(1)}\\ \vdots\\ W_{UK}^{(H)}\\
W_{UV}^{(1)}\\ \vdots\\ W_{UV}^{(H)}
\end{bmatrix}c_j
\in\mathbb R^{H(d_k+d_v)}.
\tag{4}
$$

这些展开后的 K/V 由同一个 $r$ 维向量生成。MLA 是训练时采用的参数化结构；不能直接把任意已有 MHA 的权重无损换成任意小的 $r$。

### 2.3 Query 也可以先压缩

$$
c_t^Q=\mathcal N_Q\!\left(W_D^Qx_t\right)\in\mathbb R^{r_q},
\qquad
q_t^{(h)}=W_{UQ}^{(h)}c_t^Q,
\tag{5}
$$

其中 $W_D^Q\in\mathbb R^{r_q\times d}$，$W_{UQ}^{(h)}\in\mathbb R^{d_k\times r_q}$，$\mathcal N_Q$ 与 KV 分支的归一化各有参数。Query 低秩投影是配套设计，节省历史 KV cache 的关键仍是式 (2) 的共享 $c_j$。

原始联合压缩和 query 压缩见 [DeepSeek-V2 §2.1.2](https://arxiv.org/html/2405.04434v5#S2.SS1.SSS2)；上述 latent 归一化对应 [Kimi MLA 实现](https://github.com/vllm-project/vllm/blob/a9dcfae9ef0756ce076d0053429d6c9afef067a3/vllm/models/kimi_k3/nvidia/mla.py#L255)。

## 3. 先按展开后的 K/V 理解完整计算

因果 attention 只读取 $j\le t$ 的位置。每个 head 先计算打分，再沿历史位置做 softmax：

$$
s_{tj}^{(h)}=
\frac{(q_t^{(h)})^\top k_j^{(h)}}{\sqrt{d_k}},
\qquad
a_{tj}^{(h)}=
\frac{\exp(s_{tj}^{(h)})}{\sum_{u=1}^{t}\exp(s_{tu}^{(h)})}.
\tag{6}
$$

然后用权重聚合 value，将各 heads 的输出拼接并投影回模型维度：

$$
o_t^{(h)}=\sum_{j=1}^{t}a_{tj}^{(h)}v_j^{(h)},
\qquad
y_t=W_O
\begin{bmatrix}o_t^{(1)}\\ \vdots\\ o_t^{(H)}\end{bmatrix},
\quad W_O\in\mathbb R^{d\times Hd_v}.
\tag{7}
$$

此时 MLA 仍有各 head 独立的打分和 softmax；压缩的是历史表示，并没有把所有 heads 的 attention 权重变成同一份。

## 4. 解码时怎样直接在 latent 上计算？

### 4.1 把 key 上投影移到 query 一侧

将式 (3) 代入式 (6)：

$$
\begin{aligned}
(q_t^{(h)})^\top k_j^{(h)}
&=(q_t^{(h)})^\top W_{UK}^{(h)}c_j\\
&=\left((W_{UK}^{(h)})^\top q_t^{(h)}\right)^\top c_j.
\end{aligned}
\tag{8}
$$

定义 latent 空间的 query：

$$
\widetilde q_t^{(h)}=(W_{UK}^{(h)})^\top q_t^{(h)}\in\mathbb R^r.
\tag{9}
$$

于是历史 key 不必逐个展开，打分直接写为：

$$
s_{tj}^{(h)}=
\frac{(\widetilde q_t^{(h)})^\top c_j}{\sqrt{d_k}}.
\tag{10}
$$

**分母仍是 $\sqrt{d_k}$，不能因为内积改在 $r$ 维空间计算就改成 $\sqrt r$。**这是对同一标量的代数重排，原来的缩放不变。

若 query 采用式 (5)，还可以预先组合两份线性上投影：

$$
\widetilde q_t^{(h)}=
\underbrace{(W_{UK}^{(h)})^\top W_{UQ}^{(h)}}_{\in\mathbb R^{r\times r_q}}
c_t^Q.
\tag{11}
$$

归一化仍先作用于下投影结果，不能把跨越 RMSNorm 的全部操作当成一个固定线性矩阵。式 (11) 是可用的代数合并；具体 kernel 不一定把它物化成一份新权重。

### 4.2 把 value 上投影移到加权求和之后

$$
\begin{aligned}
o_t^{(h)}
&=\sum_{j=1}^{t}a_{tj}^{(h)}W_{UV}^{(h)}c_j\\
&=W_{UV}^{(h)}\left(\sum_{j=1}^{t}a_{tj}^{(h)}c_j\right).
\end{aligned}
\tag{12}
$$

令 $z_t^{(h)}=\sum_j a_{tj}^{(h)}c_j\in\mathbb R^r$，就只需先聚合 latent，再计算 $o_t^{(h)}=W_{UV}^{(h)}z_t^{(h)}$。历史 value 也不必逐个展开。

进一步记 $W_O=[W_O^{(1)}\ \cdots\ W_O^{(H)}]$，每块 $W_O^{(h)}\in\mathbb R^{d\times d_v}$。**没有额外输出门控时**，式 (7) 可写成：

$$
y_t=\sum_{h=1}^{H}
\underbrace{W_O^{(h)}W_{UV}^{(h)}}_{\in\mathbb R^{d\times r}}
z_t^{(h)}.
\tag{13}
$$

这就是上投影可“吸收”到 query/output 投影中的含义；不是把模型的 head 数改成了 1。[固定版本的 Kimi decode 路径](https://github.com/vllm-project/vllm/blob/a9dcfae9ef0756ce076d0053429d6c9afef067a3/vllm/models/kimi_k3/nvidia/mla.py#L444)预拆出 `W_UK_T`、`W_UV`，分别执行 query 向 latent 的投影和聚合后的 value 上投影。

### 4.3 K3 的 Gated MLA 多了什么？

记 $u_t=[o_t^{(1)};\ldots;o_t^{(H)}]\in\mathbb R^{Hd_v}$。K3 在输出投影前增加逐通道门控：

$$
g_t=\operatorname{sigmoid}(W_gx_t),\qquad
y_t=W_O(g_t\odot u_t),
\quad W_g\in\mathbb R^{Hd_v\times d}.
\tag{14}
$$

门控依赖当前 token。式 (12) 的 value 聚合重排仍成立，但式 (13) 中 $W_O^{(h)}W_{UV}^{(h)}$ 的固定合并不再直接适用：两者之间插入了随 $t$ 变化的门控。该设计见 [K3 §2.1.2](https://arxiv.org/abs/2607.24653v2)。

## 5. 带 RoPE 的 MLA 为什么还缓存一个位置分量？

上述推导是 [[NoPE]] 路径。以 DeepSeek-V2 的解耦 RoPE 为例，额外设位置分量维度为偶数 $d_R$，其旋转矩阵为 $R_p\in\mathbb R^{d_R\times d_R}$。内容分量仍按前文计算，位置分量单独计算：

$$
q_t^{R,(h)}=R_tW_{QR}^{(h)}c_t^Q,\qquad
k_j^R=R_jW_{KR}x_j,
\tag{15}
$$

其中 $W_{QR}^{(h)}\in\mathbb R^{d_R\times r_q}$，$W_{KR}\in\mathbb R^{d_R\times d}$。位置 query 按 head 区分，位置 key 则在各 heads 间共享。

拼接后的 query/key 为 $[q_t^{(h)};q_t^{R,(h)}]$、$[k_j^{(h)};k_j^R]$，内积等于两个分支内积之和：

$$
s_{tj}^{(h)}=
\frac{
(\widetilde q_t^{(h)})^\top c_j
+(q_t^{R,(h)})^\top k_j^R
}{\sqrt{d_k+d_R}}.
\tag{16}
$$

Softmax 与 value 聚合沿用式 (6)、(12)，但使用式 (16) 的新打分。历史需缓存 **$(c_j,k_j^R)$**：内容分支保留矩阵吸收，位置分支另算。

为什么不直接对展开的内容 key 做 RoPE？若内容空间中的旋转为 $P_p\in\mathbb R^{d_k\times d_k}$，则：

$$
(P_tq_t^{(h)})^\top P_jW_{UK}^{(h)}c_j
=\left((W_{UK}^{(h)})^\top P_j^\top P_tq_t^{(h)}\right)^\top c_j.
\tag{17}
$$

右侧的“latent query”随历史位置 $j$ 改变，通常无法再用式 (9) 那样的一份 query 给全部历史打分。这正是解耦位置分支解决的问题。公式对应 [DeepSeek-V2 §2.1.3 与 Appendix C](https://arxiv.org/html/2405.04434v5#S2.SS1.SSS3)。

**MLA 不要求必须使用 NoPE。**K3 目标 MLA 采用 NoPE；K3.1 的 DSpark 草稿模型卡则声明 causal MLA 与 YaRN，不能直接套用目标模型的位置编码设置。具体变体见 [[DSpark]]。

## 6. 缓存到底省在哪里？

以下只计单层、长度 $T$ 的历史缓存元素数，不计 TP 复制、页填充、量化元数据和额外状态：

| 路径 | 每 token 保存什么 | 历史缓存元素数 |
|---|---|---|
| 普通 MHA | 各 head 的 K/V | $TH(d_k+d_v)$ |
| NoPE MLA | 共享 latent $c_j$ | $Tr$ |
| 上述解耦 RoPE MLA | 共享 latent 和位置 key | $T(r+d_R)$ |

相同存储精度下，NoPE MLA 相对该 MHA 基线的容量比例为 $r/[H(d_k+d_v)]$；元素数乘每元素字节数才得到存储字节数。压缩后的历史仍随 $T$ 增长，MLA 不是把全部历史合成一份固定大小状态。

Prefill 可以为高吞吐临时展开 K/V，decode 可以直接读取 latent；**长期缓存格式与一次 kernel 的临时张量要分别看**。TP 是否复制这份缓存、DCP 怎样分片，见 [[MLA-TP-DP-DCP]]。

## 7. 在哪里用到

- [[Kimi K3 - Open Frontier Intelligence]]：NoPE Gated MLA，与 KDA 交错使用。
- [[DSpark#2.1 目标特征不是“再跑一次完整目标模型”]]：$f_j$ 是草稿 context 分支的输入；各草稿层再用自己的投影生成 latent，不能把 $f_j$ 本身当成全部层共享的 MLA cache。
- [[RMSNorm]]：说明式 (2)、(5) 中的归一化；它与 DSpark 特征融合后的归一化处于不同位置。

原始方法：[DeepSeek-V2，2024](https://arxiv.org/abs/2405.04434)。Kimi 实现事实限定于上文固定的 vLLM commit；这些代数公式不构成某套硬件上的性能测量。
