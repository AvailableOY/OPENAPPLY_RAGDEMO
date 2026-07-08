## Django Admin

本项目默认提供 Django Admin，用于查看文档、分块、查询日志、检索命中记录和对话消息。

### 创建管理员账号

python manage.py createsuperuser

Username: root
Email address: 1813787887@qq.com
Password: OpenApply@2026

# 做json文件
python manage.py build_chunks data/raw/OpenApply_样本院校知识库.md --output data/processed/openapply_school_chunks.jsonl
# 创建向量库
python manage.py build_vector_store --input data/processed/openapply_school_chunks.jsonl

# 查看SQLite数据库
http://127.0.0.1:8000/admin/


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
- 轻量 rerank 排序
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
│   │   ├── reranker.py              # 轻量 rerank
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
## 环境准备

建议使用 Python 3.11 或 3.12。

安装依赖：

```bash
pip install django chromadb sentence-transformers langchain langchain-openai python-dotenv
```

如果项目已有虚拟环境，请先激活虚拟环境后再安装依赖。

## 环境变量

在项目根目录创建 `.env`：

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=your_model_name
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1

EMBEDDING_MODEL_PATH=D:\PythonModel\huggingface\bge-base-zh-v1.5

ENABLE_RERANK=false
```

说明：

- `OPENAI_API_KEY`：云端 LLM API Key
- `OPENAI_MODEL`：用于 query rewrite、回答生成、judge 的模型
- `OPENAI_BASE_URL`：OpenAI-compatible 服务地址
- `EMBEDDING_MODEL_PATH`：本地 embedding 模型路径
- `ENABLE_RERANK`：是否启用轻量 rerank 排序

## 初始化数据库

```bash
python manage.py migrate
```

可选：创建管理员账号，用于查看 Django Admin：

```bash
python manage.py createsuperuser
```

启动服务：

```bash
python manage.py runserver
```

Django Admin 地址：

```text
http://127.0.0.1:8000/admin/
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