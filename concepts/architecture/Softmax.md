---
title: Softmax
aliases:
  - softmax
tags:
  - llm
  - architecture
  - probability
url: https://docs.pytorch.org/docs/2.15/generated/torch.nn.Softmax.html
---

# Softmax

Softmax 把一组实数打分转成非负、和为 1 的权重或概率，打分越大，对应的权重越大。

## 公式

对 $n$ 个打分 $z_1,\ldots,z_n$：

$$
p_i=\frac{e^{z_i}}{\sum_{j=1}^{n}e^{z_j}},
\qquad \sum_{i=1}^{n}p_i=1.
$$

例如 $(0,0,0)$ 会变成 $(1/3,1/3,1/3)$。实际计算常先减去最大打分，再取指数，以降低溢出风险；数学结果不变。

**Softmax 不改变张量形状，但必须指定归一化轴**。例如 attention 输入、输出都是 `[B,H,L_q,L_k]`，沿最后的 $L_k$ 轴计算，每个 batch/head/query 各得到一组和为 1 的历史位置权重。

与 [[Sigmoid]] 的区别：Softmax 对一组值共同归一化，各分量相互关联；Sigmoid 独立处理每个值，不保证它们的和为 1。

## 在哪里用到

- [[MLA#3. 先按展开后的 K/V 理解完整计算]]：将 query-key 打分转成 attention 权重。
- [[DSpark#3.2 加到 base logits，再做 softmax]]：沿词表维度，将 logits 转成候选 token 的采样概率。

参考：[PyTorch Softmax 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Softmax.html)。
