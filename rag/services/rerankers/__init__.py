from django.conf import settings

from rag.services.rerankers.weighted import WeightedReranker


def get_reranker():
    backend = settings.RERANK_BACKEND.lower()

    if backend == "qwen":
        from rag.services.rerankers.qwen import QwenReranker

        return QwenReranker()

    if backend == "weighted":
        return WeightedReranker()

    raise ValueError(
        f"Unsupported RERANK_BACKEND: {backend}"
    )


def rerank(query, candidates, top_k):
    reranker = get_reranker()

    return reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=top_k,
    )