# OpenApply RAG Demo

这是一个面向 OpenApply 试用任务的最小可用 RAG demo。系统接收用户的自然语言升学问题，从给定的院校/政策资料中检索相关片段，并调用云端 LLM 生成基于资料的、可追溯的回答。

本项目重点关注：

- 检索准确性
- 回答忠实度
- 依据可追溯
- 工程结构清晰、可运行、可评估

## 功能概览

当前 demo 支持：

- Markdown 院校资料加载
- 文档切分为结构化 chunk
- 本地向量化与 Chroma 入库
- BM25 关键词检索
- 向量语义检索
- BM25 + Vector 的 RRF 融合检索
- metadata 条件过滤
- 轻量 rerank 排序 和 rerank 精排
- 云端 LLM 生成回答
- 回答质量 judge
- Query、检索结果、排序分数、回答结果日志记录
- 命令行评测 intent 和 RAG 效果
- Django Admin 查看日志和检索命中记录

## 技术栈

- Python
- Django
- SQLite
- ChromaDB
- sentence-transformers embedding
- BM25 自实现
- LangChain
- OpenAI-compatible 云端 LLM API

当前云端模型通过 `OPENAI_BASE_URL` 兼容 OpenAI SDK 调用。


## 安装依赖

建议使用 Python 3.11 或 Python 3.12。

在项目根目录执行：

```bash
python -m pip install -r requirements.txt
```
## Embedding 模型

项目默认使用：

```text
BAAI/bge-base-zh-v1.5
```

## 运行外部测试集

将外部测试文件放到：

```text
data/eval/
```

例如：

```text
data/eval/external_testset.json
```

测试脚本支持以下格式：

- JSON
- JSONL
- CSV

每条测试数据的问题字段可以使用：

- `question`
- `query`
- `input`
- `text`

推荐使用 JSON 格式：

```json
[
  {
    "id": "case_001",
    "question": "英国哪些学校的硕士毕业后有利于回国落户？"
  },
  {
    "id": "case_002",
    "question": "雅思6.5，想申请英国计算机，有哪些学校适合？"
  },
  {
    "id": "case_003",
    "question": "OpenApply资料里有没有火星大学的申请要求？"
  }
]
```

运行测试：

```bash
python manage.py run_external_testset --input data/eval/external_testset.json
```

测试结果默认保存到：

```text
output/eval/external_run_时间戳/
├── summary.json
└── details.jsonl
```

其中：

- `summary.json`：本次测试的汇总和完整结果。
- `details.jsonl`：每个测试问题的独立运行结果。

也可以指定输出目录：

```bash
python manage.py run_external_testset --input data/eval/external_testset.json --output-dir output/eval/external_result
```

CSV 文件示例：

```csv
id,question
case_001,英国哪些学校的硕士毕业后有利于回国落户？
case_002,雅思6.5，想申请英国计算机，有哪些学校适合？
```

运行 CSV 测试：

```bash
python manage.py run_external_testset --input data/eval/external_testset.csv
```



## 项目结构

```text
OPENAPPLY_RAGDEMO/
├── data/
│   ├── raw/                         # 原始资料
│   ├── processed/                   # 切分后的 JSONL chunks
│   ├── vector_store/                # Chroma 向量库
│   └── eval/                        # 评测问题与结果
├── rag/
│   ├── management/commands/         # 命令行工具
│   ├── services/
│   │   ├── splitter.py              # 文档切分
│   │   ├── vector_store.py          # 向量库构建与检索
│   │   ├── bm25_store.py            # BM25 检索
│   │   ├── hybrid_retriever.py      # 混合检索与 RRF 融合
│   │   ├── result_filters.py        # metadata 过滤
│   │   ├── rerankers/
│   │       ├── base.py
│   │       ├── weighted.py
│   │       ├── qwen.py
│   │       └── __init__.py
│   │   ├── intent_classifier.py     # 规则 + LLM 兜底意图识别
│   │   ├── query_filters.py         # 查询条件抽取
│   │   ├── generator.py             # LLM 调用、回答生成、judge
│   │   ├── prompts.py               # Prompt 模板
│   │   ├── rag_chain.py             # 主 RAG pipeline
│   │   └── logging_service.py       # 日志写入
│   └── models.py                    # 文档、chunk、query log 等模型
├── openapply_backend/               # Django 项目配置
├── manage.py
└── README.md
```



## 构建知识库

### 1. 切分原始资料

```bash
python manage.py build_chunks data/raw/OpenApply_样本院校知识库.md --output data/processed/openapply_school_chunks.jsonl
```

该命令会将原始 Markdown 院校资料切分为结构化 chunk，并保留：

- `source_ref`
- `school metadata`
- `country`
- `programme`
- `IELTS`
- `cost`
- `value tags`
- `chunk_id`

### 2. 构建向量库

```bash
python manage.py build_vector_store --input data/processed/openapply_school_chunks.jsonl
```

该命令会：

- 读取 JSONL chunks
- 使用 embedding 模型生成向量
- 写入本地 ChromaDB
- 保留 metadata，便于后续过滤和追溯

## 关键技术选择与原因

### 为什么按院校/条目切分资料

本任务的数据主要是院校和政策类结构化资料，而不是长篇连续文章。用户问题通常会围绕某一所学校、某一个国家、某类申请条件或某个政策标签展开。

因此本项目没有采用固定长度的滑窗切分，而是优先按院校条目和资料结构进行切分。这样做有几个好处：

- 一个 chunk 尽量对应一所学校或一个完整信息单元，避免把同一所学校的信息切散。
- metadata 可以和 chunk 对齐，例如国家、专业方向、雅思要求、费用、价值标签等。
- 检索命中后，LLM 拿到的是相对完整的学校信息，回答时不容易漏字段。
- source_ref 可以直接指向具体院校条目，方便追溯。

对于这个 demo 来说，资料规模较小，结构化程度较高，所以按语义条目切分比固定 token 长度切分更适合。

### 为什么使用 ChromaDB

本任务目标是交付一个可以独立运行的 RAG demo，不需要接入生产系统，也不需要处理高并发。因此选择本地持久化的 ChromaDB。

选择 ChromaDB 的原因：

- 本地即可运行，部署和演示成本低。
- 支持持久化保存向量库，适合 demo 和小规模知识库。
- 支持 metadata 存储，便于后续按国家、专业、雅思、预算等条件过滤。
- Python 接入简单，能快速完成资料入库、检索和复现。

如果后续进入生产环境，可以再评估 pgvector、Milvus、Qdrant 或云向量数据库。

### 为什么使用 BM25 + 向量混合检索

院校问答里同时存在两类检索需求：

第一类是精确匹配，例如学校名、国家、雅思、预算、专业、政策标签。这类信息适合 BM25。

第二类是语义匹配，例如“回国发展友好”“人才政策方便”“门槛低一些”“适合计算机方向”。这类表达不一定和资料中的原文完全一致，更适合向量检索。

因此本项目同时使用 BM25 和向量检索，再用 RRF 融合两路结果。这样可以降低单一路径的风险：

- 只用向量检索，可能漏掉学校名、国家、费用等精确字段。
- 只用 BM25，可能无法处理用户的自然语言改写。
- 混合检索可以同时兼顾精确实体和语义表达。

### 为什么使用 RRF 融合

BM25 分数和向量相似度不是同一尺度，直接加权会比较不稳定。RRF 主要基于排名融合，不强依赖不同检索器的原始分数是否可比。

因此本项目用 RRF 作为第一阶段融合方式，让 BM25 和向量检索都能贡献候选结果。

### 为什么使用 metadata filter

用户问题中经常包含硬条件，例如：

- 国家
- 专业方向
- 雅思分数
- 预算
- 落户/人才政策标签

这些条件不应该完全交给 LLM 判断，否则容易出现回答中推荐了不满足条件的学校。因此系统会先抽取 query filters，再在检索候选上做 metadata filter，把明显不符合硬条件的结果过滤掉。

### 为什么没有训练 BERT 分类器

当前样本数据较少，不适合训练稳定的意图分类模型。因此本项目采用简单规则 + LLM 兜底：

- 规则处理高确定性问题，例如“介绍 UCL”“UCL 和曼大怎么选”“哪些学校不需要面试”。
- 规则未命中时，再调用 LLM 判断意图。
- 这样比训练小样本分类器更稳定，也更容易解释和维护。


### 精排消融实验

为了评估云端语义精排对当前知识库的实际价值，本项目在同一组
10 个测试问题上比较了两种精排方案：

1. Weighted Rerank：结合 RRF、向量相似度、BM25 和查询词重合度
   进行本地轻量加权精排。
2. Qwen3-Rerank：使用云端 Qwen3-Rerank 对候选文档进行语义精排。

两组实验使用相同的文档切分、召回参数、metadata filter、回答模型、
Prompt 和上下文数量。

| 精排方式 | 通过数 | Hit@5（6 个已标注案例） | MRR | Judge 通过率 | 平均耗时 | Fallback |
|---|---:|---:|---:|---:|---:|---:|
| Weighted Rerank | 10/10 | 100.00% | 1.0000 | 100.00% | 20.08 秒 | 0 |
| Qwen3-Rerank | 9/10 | 83.33% | 0.7738 | 100.00% | 22.17 秒 | 0 |

其中，Hit@5 和 MRR 仅基于具有人工预期来源标注的 6 个案例计算。
其他案例主要用于检查回答忠实度和资料外问题的拒答能力。


### 最终选择

本项目最终默认采用 Weighted Rerank：

```env
RERANK_BACKEND=weighted
```

## 运行查询

命令行查询：

```bash
python manage.py rag_query "英国哪些学校的硕士毕业后有利于回国落户？"
```

示例问题：

```bash
python manage.py rag_query "雅思6.5，想申请英国计算机，有哪些学校适合？"
```

```bash
python manage.py rag_query "介绍一下UCL"
```

```bash
python manage.py rag_query "OpenApply资料里有没有火星大学的申请要求？"
```

系统会输出：

- `query_id`
- `rewritten_query`
- `judge` 结果
- 最终回答

同时会将本轮查询、检索结果、排序分数和回答写入数据库日志。


## Django Admin

创建本地管理员账号：

```bash
python manage.py createsuperuser
```

启动服务：

```bash
python manage.py runserver
```

访问：

```text
http://127.0.0.1:8000/admin/
```

Admin 可用于查看查询日志、检索结果、排序分数、引用来源和会话消息。