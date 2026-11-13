---
title: NoPE
aliases:
  - No Position Encoding
  - No Positional Encoding
tags:
  - llm
  - architecture
  - positional-encoding
arxiv: https://arxiv.org/abs/2607.24653
---

# NoPE：No Position Encoding

NoPE 表示不使用显式位置编码，例如不加入位置 embedding，也不对 query/key 应用 RoPE。它描述的是位置编码方式，与 [[MLA]] 怎样压缩 K/V 是两件事。

## 公式上少了什么？

以普通点积 attention 为例，对输入特征 $x_t,x_j\in\mathbb R^d$：

$$
q_t=W_Qx_t,\qquad k_j=W_Kx_j,\qquad
s_{tj}=\frac{q_t^\top k_j}{\sqrt{d_k}}+m_{tj},
$$

其中 $W_Q,W_K\in\mathbb R^{d_k\times d}$。这里没有显式的位置向量或旋转；因果 mask 仍然保留：

$$
m_{tj}=\begin{cases}0,&j\le t,\\-\infty,&j>t.\end{cases}
$$

Mask 限制哪些 token 可见；它与给特征添加位置编码的作用不同。

## 没有显式编码，是否就不知道顺序？

不能这样推断。$x_j$ 可以已经包含前面网络层提取的顺序信息，因果 mask 也区分可见前缀。

例如 [[Kimi K3 - Open Frontier Intelligence]] 的 MLA 不在 query/key 上加显式位置编码，但交错的 KDA 通过递归门控与衰减处理序列顺序。这是 K3 的架构配合，不意味着任意模型删除 RoPE 后都能保持原有能力。依据：[K3 §2.1.2、§3.4](https://arxiv.org/abs/2607.24653v2)。

与带解耦 RoPE 的 MLA 对照，见 [[MLA#5. 带 RoPE 的 MLA 为什么还缓存一个位置分量？]]。
