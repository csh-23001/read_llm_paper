---
title: Flow Matching
aliases:
  - FM
  - Rectified Flow
tags:
  - generative-model
  - diffusion
  - flow
---

# Flow Matching

## 一句话定义

Flow Matching (FM) 是一类**连续时间**生成模型框架：通过学习一个速度场 $v_\theta(z_t, t)$ 把噪声分布上的样本沿 ODE/SDE 流动到数据分布。可以视为把 score-based diffusion 推广到连续时间 + 任意 interpolant，并允许用 ODE 直接采样。

## 关键公式

给定数据 $x \sim p_{\text{data}}$、噪声 $\epsilon \sim p_{\text{noise}}$，定义一条插值路径（"rectified flow" 是其中最常用的线性形式）：

$$
z_t = t\, x + (1 - t)\, \epsilon,\quad t \in [0, 1]
$$

对应的真实速度场（target velocity）：

$$
v(z_t, t) = \frac{d z_t}{dt} = x - \epsilon
$$

训练目标（v-prediction）：

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x, \epsilon}\, \| v_\theta(z_t, t) - (x - \epsilon) \|^2
$$

也可以参数化为预测干净样本 $x$（**x-prediction**）或噪声 $\epsilon$，三者通过 $v(z_t, t) = (x - z_t)/(1-t)$ 等关系互推。

## 采样

- **ODE**：$dz_t/dt = v_\theta(z_t, t)$，用 Euler / Heun / RK45 等数值求解；步数少且 deterministic。
- **SDE**：可派生出对应的 stochastic 形式，每步注入小噪声、回拨 $t$，在少步数下通常更稳。

## 与 score / DDPM 的关系

- DDPM 的 $\epsilon$-prediction、score matching 的 $\nabla \log p_t$，都能通过 $v(z_t, t) = (x - z_t)/(1-t)$ 类似关系互相 reparameterize。
- FM 的好处是**插值路径自由**（不限于 VP/VE SDE 决定的曲线），且天然兼容连续时间。

## 被哪些论文使用 / 变体

- [[ELF]]（embedded language flows，x-prediction、共享权重 denoiser-decoder）
- DFM (Potaptchik 2026)、CFM (Davis 2026) —— 离散/分类版本的 flow map
- FLM / FMLM (Lee 2026) —— flow map language model
- LangFlow (Chen 2026)
- SiT (Ma 2024) —— 把 FM 用到图像扩散 Transformer
- SD3 / FLUX (Esser 2024) —— 大规模图像生成里的 rectified flow

## 主要参考

- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023
- Liu et al., *Flow Straight and Fast: Rectified Flow*, ICLR 2023
- Albergo & Vanden-Eijnden, *Building Normalizing Flows with Stochastic Interpolants*, ICLR 2023
- Albergo, Boffi, Vanden-Eijnden, *Stochastic Interpolants: A Unifying Framework*, JMLR 2025
