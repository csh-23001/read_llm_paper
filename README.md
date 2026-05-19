# read_llm_paper

大模型相关论文的阅读笔记。本仓库**同时是一个 Obsidian Vault**，用 Obsidian 打开根目录即可获得双链、图谱等体验。

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
├── CLAUDE.md          # 给 Claude 的执行准则（与本文件一致）
├── _files/            # 论文 PDF（被 .gitignore）
├── papers/            # 单篇论文笔记
└── concepts/          # 跨论文复用的方法/概念笔记
```

子目录不是强制约束 —— 因为文件名全局唯一，Obsidian 的双链不依赖目录组织。
