from django.conf import settings

from rag.services.generator import (
    extract_task_memory,
    generate_answer,
    judge_answer,
    rewrite_query,
)
from rag.services.hybrid_retriever import hybrid_retrieve
from rag.services.intent_classifier import classify_intent
from rag.services.logging_service import (
    create_query_log,
    get_or_create_session,
    save_message,
    save_retrieved_chunk,
    update_query_answer,
)
from rag.services.memory import get_task_memory, merge_task_memory, save_task_memory
from rag.services.prompts import format_contexts
from rag.services.query_filters import extract_query_filters
from rag.services.rerankers import rerank
from rag.services.result_filters import apply_metadata_filters
from rag.services.retrieval_strategy import build_retrieval_strategy

MAX_ATTEMPTS = 2


def get_result_key(item):
    metadata = item.get("metadata") or {}
    return (
        item.get("key")
        or metadata.get("chunk_id")
        or metadata.get("source_ref")
        or item.get("source_ref")
        or item.get("content", "")[:120]
    )


def build_candidate_log_item(item):
    return {
        "key": get_result_key(item),
        "source_ref": item.get("source_ref", ""),
        "pre_rerank_rank": item.get("pre_rerank_rank", item.get("rank")),
        "rerank_rank": item.get("rerank_rank"),
        "bm25_score": item.get("bm25_score"),
        "vector_score": item.get("vector_score"),
        "vector_distance": item.get("vector_distance"),
        "rrf_score": item.get("rrf_score"),
        "rerank_score": item.get("rerank_score"),
        "rank_sources": item.get("rank_sources", {}),
        "rerank_features": item.get("rerank_features", {}),
        "rerank_backend": item.get("rerank_backend"),
    }


def save_pipeline_logs(state):
    session = get_or_create_session(
        session_id=state["session_id"],
        title="Demo Session",
    )

    save_message(session, "user", state["question"])

    query_log = create_query_log(
        question=state["question"],
        session=session,
        rewritten_query=state.get("rewritten_query", ""),
        intent=state.get("intent", "school_policy_query"),
        model_name=getattr(settings, "OPENAI_MODEL", ""),
        retrieval_params={
            "bm25_top_k": getattr(settings, "BM25_TOP_K", 20),
            "vector_top_k": getattr(settings, "VECTOR_TOP_K", 20),
            "rrf_k": getattr(settings, "RRF_K", 60),
            "final_context_top_k": getattr(settings, "FINAL_CONTEXT_TOP_K", 5),
            "rerank_backend": settings.RERANK_BACKEND,
            "rerank_model": settings.RERANK_MODEL,
            "rerank_candidate_top_k": settings.RERANK_CANDIDATE_TOP_K,
            "attempts": state.get("attempt", 1),
            "retrieval_candidate_top_k": getattr(settings, "RRF_CANDIDATE_TOP_K", 30),
            "answer_display_top_k": getattr(settings, "ANSWER_DISPLAY_TOP_K", 5),
            "strategy": state.get("strategy", {}),
            "filters": state.get("filters", {}),
            "intent_result": state.get("intent_result", {}),
        },
        metadata={
            "mode": "rag_pipeline",
            "judge": state.get("judge", {}),
            "task_memory": state.get("task_memory", {}),
            "filters": state.get("filters", {}),
            "strategy": state.get("strategy", {}),
            "retrieval_debug": {
                "candidate_count": len(state.get("candidates", [])),
                "context_count": len(state.get("contexts", [])),
                "candidates": [
                    build_candidate_log_item(item)
                    for item in state.get("candidates", [])
                ],
            },
        },
    )

    used_context_keys = {
        get_result_key(item)
        for item in state.get("contexts", [])
    }

    for index, item in enumerate(state.get("candidates", []), start=1):
        save_retrieved_chunk(
            query_log=query_log,
            rank=item.get("rerank_rank") or index,
            bm25_score=item.get("bm25_score"),
            vector_score=item.get("vector_score"),
            rrf_score=item.get("rrf_score"),
            rerank_score=item.get("rerank_score"),
            used_in_answer=get_result_key(item) in used_context_keys,
            snapshot_content=item.get("content", ""),
            snapshot_source_ref=item.get("source_ref", ""),
        )

    update_query_answer(query_log, state.get("answer", ""))
    save_message(
        session,
        "assistant",
        state.get("answer", ""),
        metadata={
            "judge": state.get("judge", {}),
            "attempt": state.get("attempt"),
        },
    )

    state["query_id"] = query_log.id
    return state


def run_rag_query(question, session_id="demo-session"):
    session = get_or_create_session(
        session_id=session_id,
        title="Demo Session",
    )

    old_task_memory = get_task_memory(session)
    new_task_memory = extract_task_memory(question)
    task_memory = merge_task_memory(
        old_memory=old_task_memory,
        new_memory=new_task_memory,
        question=question,
    )
    save_task_memory(session, task_memory)

    intent_result = classify_intent(question)
    filters = extract_query_filters(question)
    strategy = build_retrieval_strategy(intent_result["intent"], filters)

    state = {
        "question": question,
        "session_id": session_id,
        "attempt": 0,
        "rewritten_query": question,
        "intent": intent_result["intent"],
        "intent_result": intent_result,
        "filters": filters,
        "strategy": strategy,
        "task_memory": task_memory,
    }

    while state["attempt"] < MAX_ATTEMPTS:
        state["attempt"] += 1

        if state["attempt"] == 1:
            state["rewritten_query"] = rewrite_query(state["question"])
        else:
            state["rewritten_query"] = rewrite_query(
                f"{state['question']}\n"
                f"Previous answer failed judge, reason: "
                f"{state.get('judge', {}).get('reason', '')}"
            )

        candidates = hybrid_retrieve(
            state["rewritten_query"],
            top_k=strategy["retrieval_top_k"],
        )
        candidates = apply_metadata_filters(candidates, filters)

        rerank_query = (
            f"用户原始问题：{state['question']}\n"
            f"检索改写：{state['rewritten_query']}"
        )

        reranked_candidates = rerank(
            query=rerank_query,
            candidates=candidates,
            top_k=len(candidates),
        )
        contexts = reranked_candidates[:strategy["context_top_k"]]

        contexts_text = format_contexts(contexts)
        answer = generate_answer(
            question=state["question"],
            contexts_text=contexts_text,
            intent=state["intent"],
            filters=state.get("filters", {}),
            display_top_k=strategy["display_top_k"],
            remaining_count=max(
                len(reranked_candidates) - strategy["display_top_k"],
                0,
            ),
        )
        judge = judge_answer(state["question"], contexts_text, answer)

        state["candidates"] = reranked_candidates
        state["contexts"] = contexts
        state["contexts_text"] = contexts_text
        state["answer"] = answer
        state["judge"] = judge

        if judge.get("pass") is True:
            break

    state = save_pipeline_logs(state)

    return {
        "query_id": state["query_id"],
        "question": state["question"],
        "rewritten_query": state["rewritten_query"],
        "answer": state["answer"],
        "contexts": state["contexts"],
        "judge": state["judge"],
        "attempt": state["attempt"],
        "task_memory": state.get("task_memory", {}),
        "intent_result": state.get("intent_result", {}),
        "filters": state.get("filters", {}),
        "strategy": state.get("strategy", {}),
    }
