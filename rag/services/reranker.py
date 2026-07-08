import re

from django.conf import settings


def tokenize(text):
    return set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", str(text).lower()))


def normalize_scores(candidates, field):
    values = [
        item.get(field)
        for item in candidates
        if isinstance(item.get(field), (int, float))
    ]
    if not values:
        return {}

    low = min(values)
    high = max(values)
    if high == low:
        return {id(item): 1.0 for item in candidates if item.get(field) is not None}

    return {
        id(item): (item.get(field) - low) / (high - low)
        for item in candidates
        if isinstance(item.get(field), (int, float))
    }


def lexical_overlap_score(query, item):
    metadata = item.get("metadata") or {}
    searchable_text = " ".join(
        [
            item.get("content", ""),
            str(metadata.get("school_name", "")),
            str(metadata.get("programme", "")),
            str(metadata.get("country", "")),
            str(metadata.get("clusters", "")),
            str(metadata.get("clusters_enriched", "")),
            str(metadata.get("value_tags", "")),
            str(metadata.get("value_tags_enriched", "")),
        ]
    )
    query_terms = tokenize(query)
    text_terms = tokenize(searchable_text)
    if not query_terms or not text_terms:
        return 0.0

    return len(query_terms & text_terms) / len(query_terms)


def rerank(query, candidates, top_k=5):
    """Lightweight explainable rerank before a model-based reranker is added."""
    if not candidates or top_k <= 0:
        return []

    ranked = []
    bm25_norm = normalize_scores(candidates, "bm25_score")
    vector_norm = normalize_scores(candidates, "vector_score")
    rrf_norm = normalize_scores(candidates, "rrf_score")
    enable_rerank = getattr(settings, "ENABLE_RERANK", False)

    for index, item in enumerate(candidates, start=1):
        enriched = dict(item)
        enriched["pre_rerank_rank"] = item.get("rank", index)

        if enable_rerank:
            lexical_score = lexical_overlap_score(query, item)
            rerank_score = (
                0.40 * rrf_norm.get(id(item), 0.0)
                + 0.30 * vector_norm.get(id(item), 0.0)
                + 0.15 * bm25_norm.get(id(item), 0.0)
                + 0.15 * lexical_score
            )
            enriched["rerank_score"] = rerank_score
            enriched["rerank_features"] = {
                "rrf_norm": rrf_norm.get(id(item), 0.0),
                "vector_norm": vector_norm.get(id(item), 0.0),
                "bm25_norm": bm25_norm.get(id(item), 0.0),
                "lexical_overlap": lexical_score,
            }
        else:
            enriched["rerank_score"] = item.get("rrf_score")
            enriched["rerank_features"] = {
                "mode": "disabled_preserve_rrf_order",
            }

        ranked.append(enriched)

    if enable_rerank:
        ranked.sort(
            key=lambda item: (
                item.get("rerank_score") or 0,
                item.get("rrf_score") or 0,
                item.get("vector_score") or 0,
            ),
            reverse=True,
        )

    for index, item in enumerate(ranked, start=1):
        item["rerank_rank"] = index

    return ranked[:top_k]
