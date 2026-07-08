from rag.models import (
    ConversationSession,
    Message,
    QueryLog,
    RetrievedChunkLog,
)


def get_or_create_session(session_id=None, title=""):
    """按 session_id 获取或创建会话；没有 session_id 时按单轮问题处理。"""
    if not session_id:
        return None

    session, _ = ConversationSession.objects.get_or_create(
        session_id=session_id,
        defaults={"title": title},
    )
    return session


def save_message(session, role, content, metadata=None):
    """保存用户或助手消息，便于后续查看连续对话记录。"""
    if session is None:
        return None

    return Message.objects.create(
        session=session,
        role=role,
        content=content,
        metadata=metadata or {},
    )


def create_query_log(
    question,
    session=None,
    rewritten_query="",
    intent="",
    model_name="",
    retrieval_params=None,
    metadata=None,
):
    """创建一次问题请求日志，记录原问题、改写 query、模型名和检索参数。"""
    return QueryLog.objects.create(
        session=session,
        question=question,
        rewritten_query=rewritten_query,
        intent=intent,
        model_name=model_name,
        retrieval_params=retrieval_params or {},
        metadata=metadata or {},
    )


def update_query_answer(query_log, answer):
    """在生成完成后回填最终回答。"""
    query_log.answer = answer
    query_log.save(update_fields=["answer"])
    return query_log


def save_retrieved_chunk(
    query_log,
    chunk=None,
    rank=1,
    bm25_score=None,
    vector_score=None,
    rrf_score=None,
    rerank_score=None,
    used_in_answer=False,
    snapshot_content="",
    snapshot_source_ref="",
):
    """保存某个 query 命中的证据片段快照及各阶段排序分数。"""
    return RetrievedChunkLog.objects.create(
        query=query_log,
        chunk=chunk,
        rank=rank,
        bm25_score=bm25_score,
        vector_score=vector_score,
        rrf_score=rrf_score,
        rerank_score=rerank_score,
        used_in_answer=used_in_answer,
        snapshot_content=snapshot_content,
        snapshot_source_ref=snapshot_source_ref,
    )
