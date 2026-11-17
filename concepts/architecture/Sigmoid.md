---
title: Sigmoid
aliases:
  - sigmoid
tags:
  - llm
  - architecture
  - activation
url: https://docs.pytorch.org/docs/2.15/generated/torch.nn.Sigmoid.html
---

# Sigmoid

Sigmoid 将每个实数独立映射到 0 和 1 之间，可以用作连续的“开关”或二分类概率输出。

## 公式

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

- $z=0$ 时，输出为 $0.5$。
- $z$ 越大，输出越接近 1；$z$ 越小，输出越接近 0。
- 对张量逐元素计算，输入、输出形状相同。

例如 $(0,0,0)$ 会变成 $(0.5,0.5,0.5)$，它们的和为 $1.5$。与 [[Softmax]] 不同，Sigmoid 不在多个元素之间分配总量为 1 的权重。

## 在哪里用到

- [[MLA#1.4.3 K3 的输出门控]]：把门控打分变成逐通道系数，再与特征相乘；靠近 0 表示抑制，靠近 1 表示放行。输入、输出形状均为 `[B,L_q,H*d_v]`。
- [[DSpark#7. Confidence head 与 Markov head 的区别]]：将 confidence head 的标量打分映射到 0～1。

参考：[PyTorch Sigmoid 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Sigmoid.html)。
