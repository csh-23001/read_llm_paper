---
title: "ELF: Embedded Language Flows"
authors:
  - Keya Hu
  - Linlu Qiu
  - Yiyang Lu
  - Hanhong Zhao
  - Tianhong Li
  - Yoon Kim
  - Jacob Andreas
  - Kaiming He
year: 2026
venue: arXiv preprint (May 2026)
arxiv: https://arxiv.org/abs/2605.10938
code: https://github.com/lillian039/ELF
tags:
  - llm
  - diffusion-language-model
  - flow-matching
  - continuous-dlm
aliases:
  - ELF
  - Embedded Language Flows
---

# ELF: Embedded Language Flows

## TL;DR

ELF 是一类**连续嵌入空间**里的扩散/流匹配语言模型。它把语言生成完全放在 T5 编码器的连续 embedding 空间中做 [[Flow-Matching]] 去噪，**只在最后一个时间步 $t=1$ 通过一个共享权重网络做 token 离散化**，不再像以往的连续 DLM 那样每一步都加 token 级 cross-entropy 监督，也不像 latent diffusion 那样再训一个独立 decoder。配合 [[Classifier-Free-Guidance]] 与一个 SDE-style 采样器，105M 参数的 ELF-B 用 **10× 更少训练 token、更少采样步数**就在 OWT / WMT14 / XSum 上打过 MDLM、Duo、FLM、LangFlow 等强基线。

## 1. 动机

主流 [[Diffusion-Language-Model]] 分两类：

- **离散 DLM**（MDLM、Duo、E2D2…）：直接在 token 空间做扩散，目前在 DLM 里整体性能更强，但与图像域的扩散/流匹配技术栈断裂（例如 CFG 在离散 DLM 上效果有限）。
- **连续 DLM**（Diffusion-LM、CDCD、DiffuSeq、SSD-LM、TESS、LD4LG…）：在 embedding / simplex / latent 空间扩散，但通常用**每步 token 级 cross-entropy / rounding loss / simplex 约束**把轨迹强行拉回离散空间，限制了 flow 的自由度。
- **Latent diffusion 路线**（LD4LG 等）：压缩到低维潜空间，再额外训练一个 decoder 还原 token。

作者的主张：当前连续 DLM 的差距不一定来自"语言本质上是离散的"，而是来自**算法设计选择不够干净**。ELF 想做一个最小化设计：**整条轨迹都在连续 embedding 空间里跑，只把"离散化"放到最后一步**，并和共享权重的去噪网络绑定。

## 2. 方法

### 2.1 总体框架（见论文 Fig. 3）

输入 token 序列 $s = [s_1, ..., s_L]$ → 经 embedding 模块（默认是**冻结的 T5 small encoder**）映射为干净 embedding $x \in \mathbb{R}^{L \times d}$ → 在 $x$ 的空间上做连续时间 [[Flow-Matching]]。

线性插值（rectified flow）：

$$
z_t = t\, x + (1 - t)\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I),\ t \in [0, 1]
$$

目标速度场 $v = dz/dt = x - \epsilon$。

### 2.2 关键设计：x-prediction（"back-to-basics"）

ELF 不预测速度 $v$，而是**预测干净 embedding $x$ 本身**（参考 [Li & He, 2025, *Back to basics: Let denoising generative models denoise*]，arXiv:2511.13720）。损失：

$$
\mathcal{L}_{\text{MSE}} = \mathbb{E}_{t, x, \epsilon}\, \frac{1}{(1 - t)^2}\, \| x_\theta(z_t, t) - x \|^2
$$

为什么选 x-prediction：

1. 高维（768d）embedding 上预测 $x$ 比预测 $v$ 在实证上更稳；
2. **跟"最后一步要解码到干净 token embedding"这个目标天然对齐**——共享权重的网络在 $t<1$ 输出 $\hat x$ 做去噪，在 $t=1$ 直接输出 $\hat x$ 然后过一个 unembedding 矩阵 $W$ 拿 logits。论文实测：用 v-prediction 时再做权重共享，MSE 与 CE 的耦合就崩了。

### 2.3 关键设计：共享权重做"去噪 + 解码"，不要单独 decoder

最后一步 $t \to 1$ 时，$z_t \to x$，没有可学的"破坏-恢复"信号，于是论文人为造一个 **token-level 的扰动 $\tilde z$**（细节在 Appendix B.1），让网络 $\text{net}_\theta$ 在 decode 模式下学：

$$
\mathcal{L}_{\text{CE}} = \mathbb{E}_{\tilde z}\, \text{CrossEnt}\!\left(W\, x_\theta(\tilde z),\ s \right)
$$

- 同一个 $\text{net}_\theta$ 由一个**二值 mode token（denoise / decode）**条件区分两种模式。
- 训练时 80% batch 走 denoise 分支（MSE），20% 走 decode 分支（CE）。
- 因此 ELF **没有独立 decoder**，也不需要 latent diffusion 那种两阶段训练。

### 2.4 采样：ODE + SDE-inspired

- ODE：标准 Euler 解 $dz_t/dt = v_\theta(z_t, t)$，最后一步切到 decode 模式，过 $W$ + argmax。
- SDE-inspired：每步注入小噪声，同时把 $t$ 往噪声方向轻微回拨（Appendix Alg. 6）。少步采样下显著优于 ODE。

### 2.5 条件生成：Self-conditioning + CFG

- **Self-conditioning**：标准 Flow Matching 一次前向得到 $\hat x'$，再把 $\hat x'$ 作为条件拼回输入做第二次前向，输出 $\hat x = \text{net}_\theta(z_t \mid \hat x', t)$；训练时 50% 概率用 $\hat x'$，否则用 0 条件。
- **CFG**：$v_{\text{cfg}}(z_t \mid c) = \omega\, v(z_t \mid c) + (1 - \omega)\, v(z_t \mid \emptyset)$，其中 $c$ 就来自 self-conditioning。
- 为了避免推理时双前向，作者直接用 **training-time CFG**（一次前向就建模 $v_{\text{cfg}}$），相当于把 [Visual generation without guidance, ICML 2025] 那一类技术搬过来。
- 条件生成（翻译 / 摘要）做法：把 condition 的干净 embedding **不加噪**地拼到模型输入前面，让 self-attention 直接看到。

### 2.6 训练算法（Alg. 1 摘要）

```
x = encode(s)                       # 冻结 T5
if uniform(0,1) < threshold:
    # denoise branch
    t = sample_t()
    e = randn_like(x)
    z = t*x + (1-t)*e
    x_pred = net(z, t, mode="denoise")
    v_pred = (x_pred - z) / (1 - t)
    loss = mse_loss(v_pred, x - e)
else:
    # decode branch, t=1
    z = corrupt(x)
    x_pred = net(z, t=1, mode="decode")
    s_pred = unembed(x_pred)
    loss = ce_loss(s_pred, s)
```

## 3. 实验

### 3.1 设置

- **模型**：冻结 T5-small encoder（35M），bottleneck 把 512 维投到 128 维再投回 hidden。三个规模：ELF-B 105M / ELF-M 342M / ELF-L 652M。
- **数据**：OWT（无条件生成，5 epochs，~95K steps，约 45B token）、WMT14 De→En（100 epochs）、XSum（100 epochs）。
- **优化器**：Muon，lr 2e-3，batch 512。
- **评测**：Gen-PPL（用 GPT-2 Large 当评分器）、unigram 熵；翻译用 BLEU；摘要用 ROUGE-1/2/L。

### 3.2 主结果

无条件生成（Fig. 7）：

- ELF-B (105M) 用 **32 步** SDE 采样达 Gen-PPL ≈ 24，显著优于需要 1024 步或额外 distillation 的 MDLM / Duo / FLM / LangFlow。
- **训练 token 上**：先前方法通常 500B+，ELF 只用 ~45B。

条件生成（Tab. 1）：

| 任务 | 指标 | 最佳基线 | ELF-B (105M+35M) |
|---|---|---|---|
| WMT14 De→En | BLEU | AR 25.2 | **26.4** |
| XSum | ROUGE-1 | MDLM 33.4 | **36.0** |
| XSum | ROUGE-2 | MDLM 11.6 | **12.2** |
| XSum | ROUGE-L | MDLM 25.8 | **27.8** |

### 3.3 关键 ablation（论文 Fig. 5）

- **Embedding 选择**：pretrained contextual（T5）> 从头训的 contextual > pretrained token-level > learnable embedding。说明**预训练的上下文化 embedding 是 ELF 的友好表示**。
- **解码策略**：共享权重（denoiser ≡ decoder） vs 两阶段独立 decoder——trade-off 曲线相近，但共享权重在低 Gen-PPL 区间能延伸更远，而且省一个训练阶段。
- **采样器**：少步数下 SDE 显著优于 ODE。
- **预测目标**（Appendix C.1）：在权重共享前提下，x-prediction 明显胜过 v-prediction。
- **Scaling**（Fig. 6）：B → M → L 一致地推动 PPL-熵 frontier 向左下，扩参数仍有红利。

## 4. 我的总结 / 启示

- ELF 的"创新"几乎全在**接口设计**上：把"什么是噪声空间、什么是 token 空间、谁负责解码"这几个边界一刀切到只有 $t=1$ 一个位置，剩下交给图像域已经成熟的 Flow Matching 技术栈。这种 minimalist 思路和 He / Li 在视觉那条线一贯的"back to basics"风格一致。
- 把 [[Self-Conditioning]] 当作 CFG 的条件来源，是连续 DLM 用 CFG 的关键 trick，否则没有显式 class label 就没法做 unconditional 对照。
- 这篇是相当强的证据：**"连续 vs 离散"在 DLM 上的差距，更多来自工程细节而非范式本身**。后续工作应关注：更大数据、和 AR 的混合训练、与 [Mean Flow / FMLM] 蒸馏路线的协同。
- 待确认问题：
  - 推理时是否真的能保持只用一次前向（论文用 training-time CFG 解决了，但 self-conditioning 的展开成本细节要看 Appendix）。
  - "decode-mode 的人为扰动 $\tilde z$" 究竟怎么构造，对最终 token 准确率影响多大（Appendix B.1）。
  - T5 encoder 冻结的依赖性——换一个更现代的 encoder（例如 LLaMA embedding）能不能继续 scale。

## 5. 相关工作脉络

- 连续 DLM：Diffusion-LM、CDCD、DiffuSeq、SSD-LM、TESS、CDM-LM、LD4LG → ELF
- 离散 DLM：D3PM、MDLM、Duo、E2D2、Seed Diffusion、LLaDA
- 同期连续 flow / flow-map 路线：DFM、CFM、FLM/FMLM、LangFlow、CoDAR
- 关键依赖：Flow Matching (Lipman 2023, Liu 2023, Albergo & Vanden-Eijnden 2023)、[[Classifier-Free-Guidance]] (Ho & Salimans 2021)、Self-conditioning (Chen, Zhang & Hinton 2023)、"Back to basics" x-prediction (Li & He 2025)
