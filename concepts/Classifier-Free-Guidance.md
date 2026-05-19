---
title: Classifier-Free Guidance
aliases:
  - CFG
  - classifier-free guidance
tags:
  - diffusion
  - guidance
  - generative-model
---

# Classifier-Free Guidance (CFG)

## 一句话定义

在扩散 / 流匹配模型推理时，用**条件预测和无条件预测的线性外推**来加强条件信号，在质量与多样性之间做 trade-off。不需要额外训练 classifier。

## 关键公式

设条件 $c$（可以是 class label、文本 prompt 或自我条件），无条件用 $\emptyset$（通常是 0 向量）表示。

在 score / $\epsilon$-prediction 框架：

$$
\tilde \epsilon(z_t \mid c) = \omega\, \epsilon_\theta(z_t \mid c) + (1 - \omega)\, \epsilon_\theta(z_t \mid \emptyset)
$$

在 [[Flow-Matching]] 框架（[[ELF]] 用的就是这种）：

$$
v_{\text{cfg}}(z_t \mid c) = \omega\, v_\theta(z_t \mid c) + (1 - \omega)\, v_\theta(z_t \mid \emptyset)
$$

- $\omega = 1$ 时退化为纯条件预测；$\omega > 1$ 是常见取值，越大越"听话"，多样性越低；$\omega = 0$ 退化为无条件。
- 训练时随机以一定概率（常用 10–20%）把条件丢成 $\emptyset$，从而同一个网络既能做条件预测也能做无条件预测。

## 推理成本与"training-time CFG"

朴素 CFG 每步需要**两次前向**（条件 + 无条件），推理成本翻倍。"training-time CFG"（Chen et al. 2025; Geng et al. 2025; Geng et al. NeurIPS 2025）让网络**直接学** $v_{\text{cfg}}$，推理时只需一次前向。[[ELF]] 采用了这个 trick。

## 在离散 vs 连续 DLM 上的差异

- 图像扩散与连续 [[Diffusion-Language-Model]] / [[Flow-Matching]] 中 CFG 是默认强 baseline。
- 在**离散 DLM**（masked diffusion / D3PM 系）里效果弱得多，是离散 DLM 一个公认的痛点（见 LangFlow、Discrete Flow Maps 等讨论）。这是 [[ELF]] 选择留在连续空间的一个直接动机。

## 被哪些论文使用 / 变体

- 原始：Ho & Salimans, *Classifier-Free Diffusion Guidance*, NeurIPS Workshops 2021
- Training-time 变体：Chen et al. 2025（视觉无 guidance 生成）、Geng et al. 2025（Mean Flows）、Geng et al. 2025（Improved Mean Flows）
- 语言扩散里的应用：[[ELF]] 用 self-conditioning 当 $c$，做 text-to-text
