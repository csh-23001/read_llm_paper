# read_llm_paper

大模型相关论文的阅读笔记。本仓库**同时是一个 Obsidian Vault**，用 Obsidian 打开根目录即可获得双链、图谱等体验。

## 阅读入口

| 论文 | 重点 | 关联概念 |
|---|---|---|
| [[ELF - Embedded Language Flows]] | 连续嵌入空间中的语言生成 | [[Diffusion-Language-Model]]、[[Flow-Matching]]、[[Classifier-Free-Guidance]] |
| [[Kimi K3 - Open Frontier Intelligence]] | vLLM 推理：混合 attention、并行缓存、推测解码；另列 K3.1 实现对照 | [[MLA-TP-DP-DCP]]、[[DSpark]] |

Kimi K3 笔记按问题提供入口，区分论文架构、固定版本的 vLLM 行为和工程推断。公式与实现说明独立保存在概念笔记中，引用指向论文、官方文档及源码。

## 仓库约定

### 1. 文件名全局唯一

整个 vault 内所有 Markdown 笔记的 basename 必须唯一（不论位于哪个子目录）。Obsidian 的 `[[wikilink]]` 按 basename 解析，重名会导致链接歧义。新建笔记前请先确认：

```bash
find . -name "<name>.md"
```

如主题相近需要复用关键词，请加区分后缀，例如 `Attention.md` vs `Attention-FlashAttention.md`。

### 2. 论文原文放在 `_files/`，不入库

- 论文 PDF / 原始材料统一放在根目录的 `_files/` 下，该目录已在 `.gitignore` 中。
- **笔记中不要引用 `_files/` 的本地路径**（本地路径对仓库的其他读者和未来的自己都无效）。
- 论文一律使用**稳定的外部链接**，优先级：arXiv abs 页面 > 会议/期刊官方链接 > DOI > 作者主页。每篇笔记的 frontmatter 中至少要有一个外链字段（如 `arxiv:` 或 `url:`）。

### 3. wikilink 中避免使用 `|`（alias 分隔符）

Obsidian 的 wikilink 用 `|` 表示别名：`[[Note|显示文本]]`；但 `|` 同时是 Markdown 表格的列分隔符，**当带 alias 的 wikilink 出现在表格单元格内时，`|` 会被解析为列边界，链接被截断**。所以：

- 默认不写 `[[A|B]]`。需要不同显示文本时，优先在目标笔记的 frontmatter 里写 `aliases: [...]`，再用 `[[别名]]` 直接链过去。
- 表格里的 wikilink 一律写无 alias 的 `[[Note]]`；表格单元格里出现的字面量 `|` 必须转义为 `\|`。

### 4. 单篇论文一个笔记，方法外链化

- 每篇论文对应**一个独立的 Markdown 笔记**，文件名建议形如 `<Short-Name>.md` 或 `<Year>-<Short-Name>.md`，保持全局唯一。
- 论文中可被多篇论文共享/比较的概念（方法、技巧、数据集、benchmark、架构组件等），应抽出为**独立的概念笔记**，在论文笔记中通过 `[[概念笔记]]` 链接过去；不要在每篇论文里重复展开同一方法。
- 概念笔记记录该方法的核心定义、关键公式 / 伪代码，以及"被哪些论文使用 / 如何变体"的回链列表，方便横向比较。
- 推理相关概念统一放在 `concepts/inference/`，包括 KV cache、并行策略、推测解码、推理量化与服务调度；后续同类问题继续整理到这个目录，文件名仍保持全仓库唯一。
- 模型架构相关概念放在 `concepts/architecture/`，例如 [[RMSNorm]] 这样的归一化组件；基础概念尽量用简短定义、公式和符号解释说明。

### 笔记 frontmatter 推荐字段

```yaml
---
title: <论文完整标题>
authors: [<作者1>, <作者2>]
year: 2024
venue: <会议/期刊/arXiv>
arxiv: https://arxiv.org/abs/xxxx.xxxxx
tags: [llm, <子领域>]
---
```

## 目录结构（建议）

```
.
├── README.md
├── AGENTS.md          # 给 Codex 的执行准则
├── CLAUDE.md          # 给 Claude 的执行准则
├── _files/            # 论文 PDF（被 .gitignore）
├── papers/            # 单篇论文笔记
└── concepts/          # 跨论文复用的方法/概念笔记
    ├── architecture/  # 模型架构组件：归一化、位置编码、激活函数等
    └── inference/     # 推理相关概念：缓存、并行、推测解码等
```

子目录不是强制约束 —— 因为文件名全局唯一，Obsidian 的双链不依赖目录组织。
