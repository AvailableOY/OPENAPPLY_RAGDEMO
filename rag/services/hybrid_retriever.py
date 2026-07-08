from django.conf import settings

from rag.services.bm25_store import search_bm25


def get_result_key(item):
    metadata = item.get("metadata") or {}
    return (
        item.get("key")
        or metadata.get("chunk_id")
        or metadata.get("source_ref")
        or item.get("source_ref")
        or item.get("content", "")[:120]
    )


def add_rrf_scores(merged, results, source_name, rrf_k):
    for rank, item in enumerate(results, start=1):
        key = get_result_key(item)
        existing = merged.setdefault(
            key,
            {
                "key": key,
                "content": item.get("content", ""),
                "source_ref": item.get("source_ref")
                or (item.get("metadata") or {}).get("source_ref", ""),
                "metadata": item.get("metadata") or {},
                "chunk": item.get("chunk"),
                "bm25_score": None,
                "vector_score": None,
                "vector_distance": None,
                "rrf_score": 0.0,
                "rank_sources": {},
            },
        )

        if item.get("content") and not existing.get("content"):
            existing["content"] = item["content"]
        if item.get("source_ref") and not existing.get("source_ref"):
            existing["source_ref"] = item["source_ref"]
        if item.get("metadata"):
            existing["metadata"] = {**existing["metadata"], **item["metadata"]}
        if item.get("chunk") is not None:
            existing["chunk"] = item["chunk"]

        if source_name == "bm25":
            existing["bm25_score"] = item.get("bm25_score")
        elif source_name == "vector":
            existing["vector_score"] = item.get("vector_score")
            existing["vector_distance"] = item.get("distance")

        existing["rrf_score"] += 1 / (rrf_k + rank)
        existing["rank_sources"][source_name] = rank


def hybrid_retrieve(query, top_k=None):
    from rag.services.vector_store import search_vector_store

    bm25_top_k = getattr(settings, "BM25_TOP_K", 20)
    vector_top_k = getattr(settings, "VECTOR_TOP_K", 20)
    rrf_k = getattr(settings, "RRF_K", 60)
    final_top_k = top_k or getattr(settings, "RRF_CANDIDATE_TOP_K", 12)

    bm25_results = search_bm25(query=query, top_k=bm25_top_k)
    vector_results = search_vector_store(query=query, top_k=vector_top_k)

    merged = {}
    add_rrf_scores(merged, bm25_results, "bm25", rrf_k)
    add_rrf_scores(merged, vector_results, "vector", rrf_k)

    candidates = list(merged.values())
    candidates.sort(
        key=lambda item: (
            item["rrf_score"],
            item.get("bm25_score") or 0,
            item.get("vector_score") or 0,
        ),
        reverse=True,
    )

    for index, item in enumerate(candidates, start=1):
        item["rank"] = index

    return candidates[:final_top_k]
