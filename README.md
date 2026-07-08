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
环境准备
建议使用 Python 3.11 或 3.12。
安装依赖：
pip install django chromadb sentence-transformers langchain langchain-openai python-dotenv
如果项目已有虚拟环境，请先激活虚拟环境后再安装依赖。
环境变量
在项目根目录创建 .env：
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=your_model_name
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1

EMBEDDING_MODEL_PATH=D:\PythonModel\huggingface\bge-base-zh-v1.5

ENABLE_RERANK=false
说明：
OPENAI_API_KEY：云端 LLM API Key
OPENAI_MODEL：用于 query rewrite、回答生成、judge 的模型
OPENAI_BASE_URL：OpenAI-compatible 服务地址
EMBEDDING_MODEL_PATH：本地 embedding 模型路径
ENABLE_RERANK：是否启用轻量 rerank 排序
初始化数据库
python manage.py migrate
可选：创建管理员账号，用于查看 Django Admin：
python manage.py createsuperuser
启动服务：
python manage.py runserver
Django Admin 地址：
http://127.0.0.1:8000/admin/
构建知识库
1. 切分原始资料
python manage.py build_chunks data/raw/OpenApply_样本院校知识库.md --output data/processed/openapply_school_chunks.jsonl
该命令会将原始 Markdown 院校资料切分为结构化 chunk，并保留：
source_ref
school metadata
country
programme
IELTS
cost
value tags
chunk_id
2. 构建向量库
python manage.py build_vector_store --input data/processed/openapply_school_chunks.jsonl
该命令会：
读取 JSONL chunks
使用 embedding 模型生成向量
写入本地 ChromaDB
保留 metadata，便于后续过滤和追溯
运行查询
命令行查询：
python manage.py rag_query "英国哪些学校的硕士毕业后有利于回国落户？"
示例问题：
python manage.py rag_query "雅思6.5，想申请英国计算机，有哪些学校适合？"
python manage.py rag_query "介绍一下UCL"
python manage.py rag_query "OpenApply资料里有没有火星大学的申请要求？"
系统会输出：
query_id
rewritten_query
judge 结果
最终回答
同时会将本轮查询、检索结果、排序分数和回答写入数据库日志。
检索流程
当前检索流程如下：
用户问题
-> query rewrite
-> BM25 检索
-> 向量检索
-> RRF 融合
-> metadata filter
-> rerank
-> 选取 top contexts
-> LLM 回答生成
-> judge 检查
-> 日志保存
为什么使用 BM25 + 向量混合检索
院校问答中同时存在两类信息：
精确关键词
例如学校名、国家、雅思、预算、专业、政策标签。

语义表达
例如“回国发展友好”“人才政策方便”“门槛低一些”。

单独使用向量检索容易漏掉精确实体；单独使用 BM25 又不擅长语义改写。因此本项目使用：
BM25 + Vector + RRF
来提升召回稳定性。
为什么使用 RRF
RRF 可以融合不同检索器的排序结果，不依赖不同分数体系之间的绝对数值可比性。BM25 分数和向量相似度尺度不同，直接加权容易不稳定，因此使用 RRF 做第一阶段融合。
metadata filter
系统会从 query 中抽取条件，例如：
country
major
ielts_max
budget_max_rmb_wan
value_tag
然后对候选结果做 metadata 过滤，避免把不符合硬条件的学校交给 LLM。
rerank
当前实现了轻量 rerank，可通过 .env 控制：
ENABLE_RERANK=true
启用后，系统会综合：
RRF 分数
向量相似度
BM25 分数
query 与 chunk 的词面重合度
生成 rerank_score 并重新排序。
该 rerank 是轻量、可解释版本，没有引入额外 cross-encoder 模型。当前任务样本较小，因此优先保持系统简单、可运行、可解释。
意图识别
当前意图识别采用：
简单规则匹配 + LLM 兜底
支持的 intent：
list_schools：学校列表、推荐学校、有哪些学校
continue_list：继续查看上一轮列表
school_detail：具体学校详情
compare_schools：学校对比或怎么选
filter_requirement：带雅思、预算、门槛、面试等筛选条件
general_qa：其他普通问题
没有使用 BERT 分类器，原因是当前样本量较少，不适合训练可靠分类模型。规则负责高确定性场景，规则未命中时由 LLM 兜底判断。
回答生成与防幻觉
Prompt 中强制要求模型：
只基于检索到的资料回答
不使用资料外常识补充
关键结论必须带引用编号
资料不足时必须说明“当前资料未找到相关依据”
最后列出依据来源
对于资料中不存在的问题，例如：
OpenApply资料里有没有火星大学的申请要求？
系统应回答当前资料未找到相关依据，而不是编造。
可追溯设计
每个检索片段会被格式化为：
[1] 来源：source_ref
内容：...
回答中通过 [1]、[2] 等编号引用依据。
数据库中还会记录：
原始问题
改写后的 query
intent_result
filters
retrieval strategy
BM25 score
vector score
vector distance
RRF score
rerank score
是否进入最终回答上下文
judge 结果
最终回答
这使得每次回答都可以回溯到具体资料片段，也方便排查错误来自召回、排序、过滤还是生成阶段。
评测
意图识别评测
python manage.py eval_intent --output data/eval/intent_eval_results_latest.json
当前意图识别目标是：
intent + filters 尽量 10/10
RAG 整体评测
python manage.py eval_rag --output data/eval/rag_eval_results_latest.json
RAG 评测会检查：
intent 是否正确
filters 是否正确
是否命中期望 source
judge 是否通过
输出结果会保存到 data/eval/ 目录。
日志查看
可以通过 Django Admin 查看：
QueryLog
RetrievedChunkLog
ConversationSession
Message
Document
DocumentChunk
也可以直接查看 SQLite 数据库中的日志记录。
重点关注：
retrieval_params
metadata.retrieval_debug
retrieved_chunks
bm25_score
vector_score
rrf_score
rerank_score
used_in_answer
当前完成情况
已完成：
原始资料切分
向量化入库
BM25 检索
向量检索
RRF 融合
metadata filter
轻量 rerank
规则 + LLM 兜底意图识别
LLM 回答生成
回答 judge
可追溯引用
检索与回答日志
intent eval
rag eval
已知限制
当前 demo 仍有一些限制：
数据集较小
当前主要针对试用任务样本资料，不覆盖真实生产规模数据。

rerank 是轻量版本
当前没有接入 cross-encoder reranker。若后续数据规模和问题复杂度提升，可以接入 BGE reranker 或云端 rerank API。

规则识别只覆盖高频场景
对模糊表达依赖 LLM fallback。

评测集较小
当前评测主要用于验证核心链路是否稳定。后续应从真实 query log 中持续补充测试问题。

没有前端界面
根据任务要求，本 demo 聚焦命令行和简单服务能力，不实现前端 UI。

后续可改进方向
如果继续推进，可以考虑：
增加更多真实测试问题
为失败 case 建立回归测试
接入 cross-encoder reranker
增加无答案检测阈值
优化 chunk 切分粒度
将检索日志做成可视化 dashboard
支持多文档、多数据源增量入库
设计取舍说明
本项目没有追求复杂架构，而是优先完成任务书要求的最小可用闭环：
资料入库
-> 准确检索
-> 基于资料回答
-> 给出引用依据
-> 可运行、可评估、可排查
选择 Django 是为了快速获得命令行、数据库模型、Admin 日志查看能力。
选择 Chroma 是因为本地持久化简单，适合 demo 阶段快速验证。
选择 BM25 + 向量混合检索，是因为院校资料同时包含精确实体和语义表达，混合检索比单一路径更稳。
选择规则 + LLM 兜底做意图识别，是因为当前样本量不足以训练稳定分类模型，而规则能够覆盖高确定性场景，LLM 可以处理规则未覆盖的模糊表达。
选择轻量 rerank，是为了在不增加重依赖的情况下提升排序可解释性，并将排序特征写入日志，方便后续分析。
```