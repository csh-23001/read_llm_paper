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

下面先推导不带显式位置编码的 [[NoPE]] 版本，再补充 DeepSeek-V2 风格的解耦 RoPE。单 token 公式只讨论单层、单请求，向量均为列向量；§2.4–2.6 另外用批量张量展示完整形状。省略 bias、量化和并行通信。矩阵重排在实数运算下等价，不保证不同浮点 kernel 的结果逐位相同。

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

§2.1–2.3 给出单 token 公式；包含 batch 维度的完整形状链路见 §2.4–2.6。

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

### 2.4 完整张量形状：从输入到 attention 输出

先看 **NoPE、展开 K/V 的计算视角**。这里采用最后一维为特征的行布局，因此单 token 的 $Wx$ 在批量形式下写成 $XW^\top$；它与前文列向量公式等价。

- $B$：batch 中的请求数。
- $L_q$：本轮 query 位置数；$L_k$：可以读取的全部历史与当前 K/V 位置数。
- $d$：模型特征维度；$H$：head 数；$d_k,d_v$：每 head 的 key/query、value 维度。
- $r,r_q$：KV、query 的压缩维度。**不要求 $d=Hd_k=Hd_v$，也不要求 $L_q=L_k$。**

输入记作 $X_Q\in\mathbb R^{B\times L_q\times d}$ 和 $X_{KV}\in\mathbb R^{B\times L_k\times d}$。$X_{KV}$ 是历史特征堆叠的数学视图；decode 实际复用已缓存的 latent，只为新位置追加投影结果，不重新投影全部历史。

**① 压缩前的参照：普通 MHA 的 K/V 有多大？**

| 步骤           | 运算                                                  | 输出形状               |
| ------------ | --------------------------------------------------- | ------------------ |
| 原始特征         | $X_{KV}$                                            | `[B, L_k, d]`      |
| 普通 K 投影      | $X_{KV}(W_K^{\mathrm{MHA}})^\top$，权重形状 `[H*d_k, d]` | `[B, L_k, H*d_k]`  |
| K 拆出 heads   | reshape                                             | `[B, L_k, H, d_k]` |
| K 将 head 轴前移 | transpose，交换位置轴与 head 轴                             | `[B, H, L_k, d_k]` |
| 普通 V 投影      | $X_{KV}(W_V^{\mathrm{MHA}})^\top$，权重形状 `[H*d_v, d]` | `[B, L_k, H*d_v]`  |
| V 拆出 heads   | reshape                                             | `[B, L_k, H, d_v]` |
| V 将 head 轴前移 | transpose，交换位置轴与 head 轴                             | `[B, H, L_k, d_v]` |

这里的普通 MHA 只是容量参照。**MLA 实际从 $X_{KV}$ 直接下投影得到 latent，不需要先生成上述完整 K/V 再压缩。**

**② MLA 的 KV 下投影、归一化与各 head 上投影**。令 $W_{UK}\in\mathbb R^{Hd_k\times r}$、$W_{UV}\in\mathbb R^{Hd_v\times r}$ 为各 head 上投影矩阵沿行堆叠后的权重。

| 步骤           | 运算                                                     | 输出形状               |
| ------------ | ------------------------------------------------------ | ------------------ |
| KV 下投影       | $C_{\mathrm{raw}}=X_{KV}(W_D^{KV})^\top$，权重形状 `[r, d]` | `[B, L_k, r]`      |
| Latent 归一化   | $C=\mathcal N_{KV}(C_{\mathrm{raw}})$，沿最后的 $r$ 维计算     | `[B, L_k, r]`      |
| 内容 K 上投影     | $K_{\mathrm{flat}}=CW_{UK}^\top$，权重形状 `[H*d_k, r]`     | `[B, L_k, H*d_k]`  |
| K 拆出 heads   | reshape                                                | `[B, L_k, H, d_k]` |
| K 将 head 轴前移 | transpose                                              | `[B, H, L_k, d_k]` |
| V 上投影        | $V_{\mathrm{flat}}=CW_{UV}^\top$，权重形状 `[H*d_v, r]`     | `[B, L_k, H*d_v]`  |
| V 拆出 heads   | reshape                                                | `[B, L_k, H, d_v]` |
| V 将 head 轴前移 | transpose                                              | `[B, H, L_k, d_v]` |

**长期缓存的是 $C$，形状为 `[B, L_k, r]`，没有 head 轴**。表中的完整 K/V 是便于理解的展开结果，是否临时物化由计算路径决定。

**③ Query 的下投影与上投影**。令 $W_{UQ}\in\mathbb R^{Hd_k\times r_q}$ 为所有 query heads 的上投影权重。

| 步骤               | 运算                                                     | 输出形状               |
| ---------------- | ------------------------------------------------------ | ------------------ |
| 本轮输入             | $X_Q$                                                  | `[B, L_q, d]`      |
| Query 下投影        | $C_{\mathrm{raw}}^Q=X_Q(W_D^Q)^\top$，权重形状 `[r_q, d]`   | `[B, L_q, r_q]`    |
| Query latent 归一化 | $C^Q=\mathcal N_Q(C_{\mathrm{raw}}^Q)$，沿最后的 $r_q$ 维计算  | `[B, L_q, r_q]`    |
| Query 上投影        | $Q_{\mathrm{flat}}=C^QW_{UQ}^\top$，权重形状 `[H*d_k, r_q]` | `[B, L_q, H*d_k]`  |
| Q 拆出 heads       | reshape                                                | `[B, L_q, H, d_k]` |
| Q 将 head 轴前移     | transpose                                              | `[B, H, L_q, d_k]` |

**④ Attention 打分、mask、softmax 和 value 聚合**。下式中的转置只交换最后两个轴，保留 batch 和 head 轴：

$$
S=\frac{QK^\top}{\sqrt{d_k}}+M,
\qquad
A=\operatorname{softmax}_{\mathrm{last}}(S),
\qquad O=AV.
$$

| 步骤           | 运算及形状变化                                     | 输出形状               |
| ------------ | ------------------------------------------- | ------------------ |
| K 的最后两轴转置    | `[B, H, L_k, d_k]` → `[B, H, d_k, L_k]`     | `[B, H, d_k, L_k]` |
| Query-key 内积 | `[B, H, L_q, d_k]` @ `[B, H, d_k, L_k]`     | `[B, H, L_q, L_k]` |
| 缩放           | 除以 $\sqrt{d_k}$                             | `[B, H, L_q, L_k]` |
| 加因果 mask     | 加上 `[1, 1, L_q, L_k]` 的 $M$，广播到各 batch/head | `[B, H, L_q, L_k]` |
| Softmax      | 沿最后的 $L_k$ 轴归一化，每个 query/head 得到一组历史位置权重    | `[B, H, L_q, L_k]` |
| Value 加权求和   | `[B, H, L_q, L_k]` @ `[B, H, L_k, d_v]`     | `[B, H, L_q, d_v]` |

若 query 对应 K/V 的末尾 $L_q$ 个连续位置，采用从 1 开始的局部下标，因果 mask 为：

$$
M_{1,1,i,j}=
\begin{cases}
0,&j\le L_k-L_q+i,\\
-\infty,&j>L_k-L_q+i,
\end{cases}
\quad i=1,\ldots,L_q,\quad j=1,\ldots,L_k.
$$

例如 decode 时 $L_q=1$，唯一 query 可以看到全部 $L_k$ 个已存在的位置，包括当前位置；不能直接对一个 $1\times L_k$ 的矩阵套用从左上角起算的三角 mask，否则会错误地只保留第一个 key。

**⑤ 合并 heads，投影回模型维度**。这里“最后输出”指 attention 模块的输出，不包含后续残差相加或 FFN。

| 步骤           | 运算                                                        | 输出形状               |
| ------------ | --------------------------------------------------------- | ------------------ |
| 各 head 的聚合结果 | $O$                                                       | `[B, H, L_q, d_v]` |
| 将位置轴移回前面     | transpose                                                 | `[B, L_q, H, d_v]` |
| 拼接所有 heads   | reshape，得到 $U$                                            | `[B, L_q, H*d_v]`  |
| 若采用 K3 输出门控  | $G=\operatorname{sigmoid}(X_QW_g^\top)$，权重形状 `[H*d_v, d]` | `[B, L_q, H*d_v]`  |
| 应用门控         | $U_g=G\odot U$；没有门控时直接使用 $U$                              | `[B, L_q, H*d_v]`  |
| 输出投影         | $Y=U_gW_O^\top$，权重形状 `[d, H*d_v]`                         | `[B, L_q, d]`      |

这条路径的 attention 形状主线是：

```text
Q [B,H,L_q,d_k] × K^T [B,H,d_k,L_k]
→ scores / mask / softmax [B,H,L_q,L_k]
× V [B,H,L_k,d_v]
→ head outputs [B,H,L_q,d_v]
→ transpose + concatenate [B,L_q,H*d_v]
→ optional gate [B,L_q,H*d_v]
→ output projection [B,L_q,d]
```

### 2.5 不展开历史 K/V 时，形状怎样变化？

§4 推导了矩阵吸收；这里直接列出形状。令 $\mathcal W_K$ 为各 head 的 $W_{UK}^{(h)}$ 堆叠，形状 `[H, d_k, r]`；$\mathcal W_V$ 为各 head 的 $(W_{UV}^{(h)})^\top$ 堆叠，形状 `[H, r, d_v]`。

| 步骤                 | 运算及形状变化                                                    | 输出形状            |
| ------------------ | ---------------------------------------------------------- | --------------- |
| 将 Q 投影到 latent 空间  | $\widetilde Q=Q\mathcal W_K$：`[B,H,L_q,d_k]` @ `[H,d_k,r]` | `[B,H,L_q,r]`   |
| 给缓存增加可广播的 head 轴   | $C_H=C[:,\mathrm{None},:,:]$，由 `[B,L_k,r]` 增加长度为 1 的轴      | `[B,1,L_k,r]`   |
| 缓存最后两轴转置           | $C_H^\top$，仅交换最后两个轴                                        | `[B,1,r,L_k]`   |
| 直接对 latent 打分      | $\widetilde Q C_H^\top$：`[B,H,L_q,r]` @ `[B,1,r,L_k]`      | `[B,H,L_q,L_k]` |
| 缩放、mask、softmax    | 仍除以 $\sqrt{d_k}$，使用与 §2.4 相同的 mask 和 softmax 轴，得到 $A$      | `[B,H,L_q,L_k]` |
| 聚合 latent          | $Z=AC_H$：`[B,H,L_q,L_k]` @ `[B,1,L_k,r]`                   | `[B,H,L_q,r]`   |
| 将聚合结果上投影为 value 输出 | $O=Z\mathcal W_V$：`[B,H,L_q,r]` @ `[H,r,d_v]`              | `[B,H,L_q,d_v]` |
| 合并 heads、门控、输出投影   | 与 §2.4 的步骤 ⑤ 相同                                            | `[B,L_q,d]`     |

`[B,1,L_k,r]` 的 head 轴按计算需要广播到 $H$；**广播不意味着必须存下 $H$ 份 latent**。这条路径保留相同形状的 attention 权重，但用共享 latent 代替展开后的历史 K/V。

### 2.6 Prefill、decode 与 RoPE 的形状对照

下面取一组**假设尺寸**：$B=2,d=16,H=2,d_k=4,d_v=6,r=3,r_q=5$。先 prefill 3 个 token，再 decode 1 个新 token：

| 张量                          | Prefill：$L_q=L_k=3$ | 下一轮 decode：$L_q=1,L_k=4$ |
| --------------------------- | ------------------- | ------------------------ |
| 本轮输入 $X_Q$                  | `[2,3,16]`          | `[2,1,16]`               |
| 本轮 query latent $C^Q$       | `[2,3,5]`           | `[2,1,5]`                |
| 全部 KV latent cache $C$      | `[2,3,3]`           | `[2,4,3]`，只追加 `[2,1,3]`  |
| Query $Q$                   | `[2,2,3,4]`         | `[2,2,1,4]`              |
| 展开后的 K                      | `[2,2,3,4]`         | `[2,2,4,4]`              |
| 展开后的 V                      | `[2,2,3,6]`         | `[2,2,4,6]`              |
| Scores / attention 权重 $A$   | `[2,2,3,3]`         | `[2,2,1,4]`              |
| Latent query $\widetilde Q$ | `[2,2,3,3]`         | `[2,2,1,3]`              |
| Latent 聚合结果 $Z$             | `[2,2,3,3]`         | `[2,2,1,3]`              |
| 每 head 输出 $O$               | `[2,2,3,6]`         | `[2,2,1,6]`              |
| 拼接 / 门控后的输出 $U$ 或 $U_g$     | `[2,3,12]`          | `[2,1,12]`               |
| 最终输出 $Y$                    | `[2,3,16]`          | `[2,1,16]`               |

这里 prefill 的 $L_k$ 与 $r$ 恰好都等于 3，所以 $A$、$\widetilde Q$ 和 $Z$ 的数组尺寸相同，**轴的含义仍不同**：$A$ 最后一轴是历史位置，后两者最后一轴是 latent 特征。

若使用 §5 的解耦 RoPE，令 $d_R$ 为额外的位置特征维度，内容分支沿用上面的形状；位置分支及其合并方式为：

| 步骤                          | 形状变化                                                                   |
| --------------------------- | ---------------------------------------------------------------------- |
| 位置 query 投影                 | $C^QW_{QR}^\top$：`[B,L_q,r_q]` @ `[r_q,H*d_R]` → `[B,L_q,H*d_R]`       |
| 位置 query 拆 heads 并前移 head 轴 | `[B,L_q,H*d_R]` → `[B,L_q,H,d_R]` → `[B,H,L_q,d_R]`                    |
| 对位置 query 应用 RoPE           | `[B,H,L_q,d_R]` → `[B,H,L_q,d_R]`                                      |
| 共享位置 key 投影并旋转              | $X_{KV}W_{KR}^\top$：`[B,L_k,d]` @ `[d,d_R]` → `[B,L_k,d_R]`；RoPE 后形状不变 |
| 为位置 key 增加 head 轴并广播        | `[B,L_k,d_R]` → `[B,1,L_k,d_R]`，逻辑上可广播到 `[B,H,L_k,d_R]`                |
| 拼接内容与位置 query               | `[B,H,L_q,d_k]` + `[B,H,L_q,d_R]` → `[B,H,L_q,d_k+d_R]`                |
| 拼接内容与位置 key                 | `[B,H,L_k,d_k]` + 广播的位置 key → `[B,H,L_k,d_k+d_R]`                      |
| 拼接后的 QK 打分                  | `[B,H,L_q,d_k+d_R]` @ `[B,H,d_k+d_R,L_k]` → `[B,H,L_q,L_k]`            |
| 缩放、mask、softmax             | 改用 $\sqrt{d_k+d_R}$；权重形状仍为 `[B,H,L_q,L_k]`                             |
| Value 聚合到最终输出               | V 维度仍是 $d_v$；后续形状与 §2.4、§2.5 相同                                        |

表里的 `+` 表示沿最后一轴拼接，不是逐元素相加。Latent 路径则分别计算内容打分与位置打分，二者都是 `[B,H,L_q,L_k]`，相加后再缩放、加 mask、softmax；不必实际拼接完整历史 key。其逻辑缓存由 `[B,L_k,r]` 的 $C$ 和 `[B,L_k,d_R]` 的位置 key 组成。

以上是等长请求的逻辑张量布局；vLLM 可采用打包 token、分页缓存和专用 kernel 布局，不能将这些形状直接视为其物理显存排列。

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

**分母仍是 $\sqrt{d_k}$，不能因为内积改在 $r$ 维空间计算就改成 $\sqrt r$**。这是对同一标量的代数重排，原来的缩放不变。

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

**MLA 不要求必须使用 NoPE**。K3 目标 MLA 采用 NoPE；K3.1 的 DSpark 草稿模型卡则声明 causal MLA 与 YaRN，不能直接套用目标模型的位置编码设置。具体变体见 [[DSpark]]。

## 6. 缓存到底省在哪里？

以下只计单层、长度 $T$ 的历史缓存元素数，不计 TP 复制、页填充、量化元数据和额外状态：

| 路径            | 每 token 保存什么      | 历史缓存元素数       |
| ------------- | ----------------- | ------------- |
| 普通 MHA        | 各 head 的 K/V      | $TH(d_k+d_v)$ |
| NoPE MLA      | 共享 latent $c_j$   | $Tr$          |
| 上述解耦 RoPE MLA | 共享 latent 和位置 key | $T(r+d_R)$    |

相同存储精度下，NoPE MLA 相对该 MHA 基线的容量比例为 $r/[H(d_k+d_v)]$；元素数乘每元素字节数才得到存储字节数。压缩后的历史仍随 $T$ 增长，MLA 不是把全部历史合成一份固定大小状态。

Prefill 可以为高吞吐临时展开 K/V，decode 可以直接读取 latent；**长期缓存格式与一次 kernel 的临时张量要分别看**。TP 是否复制这份缓存、DCP 怎样分片，见 [[MLA-TP-DP-DCP]]。

## 7. 在哪里用到

- [[Kimi K3 - Open Frontier Intelligence]]：NoPE Gated MLA，与 KDA 交错使用。
- [[DSpark#2.1 目标特征不是“再跑一次完整目标模型”]]：$f_j$ 是草稿 context 分支的输入；各草稿层再用自己的投影生成 latent，不能把 $f_j$ 本身当成全部层共享的 MLA cache。
- [[RMSNorm]]：说明式 (2)、(5) 中的归一化；它与 DSpark 特征融合后的归一化处于不同位置。

原始方法：[DeepSeek-V2，2024](https://arxiv.org/abs/2405.04434)。Kimi 实现事实限定于上文固定的 vLLM commit；这些代数公式不构成某套硬件上的性能测量。
