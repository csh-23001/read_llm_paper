---
title: RMSNorm
aliases:
  - Root Mean Square Layer Normalization
tags:
  - llm
  - architecture
  - normalization
arxiv: https://arxiv.org/abs/1910.07467
---

# RMSNorm

RMSNorm 用均方根缩放特征，再乘可学习的逐维权重，控制特征的整体幅度。

## 公式

对一个 token 的特征向量 $x\in\mathbb R^d$，常见的带稳定项写法为：

$$
s=\sqrt{\frac{1}{d}\sum_{k=1}^{d}x_k^2+\epsilon},
\qquad
y_i=\gamma_i\frac{x_i}{s},\quad i=1,\ldots,d.
$$

- $d$：特征维度；对每个 token 单独计算，不跨 token 求平均。
- $\gamma_i$：每一维可学习的缩放权重，通常初始化为 1。
- $\epsilon$：很小的正数，防止分母为零或过小。

例如忽略 $\epsilon$、令 $\gamma_i=1$，输入 $(3,3)$ 会变成 $(1,1)$。RMSNorm 不减均值，所以输出不必是零均值。上述公式见 [PyTorch RMSNorm 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.RMSNorm.html)。

## 在哪里用到

[[DSpark#2.1 目标特征不是“再跑一次完整目标模型”]]：目标层特征先拼接，再投影成 $d$ 维向量，最后通过 RMSNorm 调整幅度；输入、输出都是 $d$ 维。

方法来源：[Zhang & Sennrich，Root Mean Square Layer Normalization，2019](https://arxiv.org/abs/1910.07467)。
