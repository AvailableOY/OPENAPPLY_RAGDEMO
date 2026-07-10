import re

from rag.services.rerankers.base import BaseReranker, add_final_ranks


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text):
    return set(TOKEN_PATTERN.findall(str(text).lower()))


def normalize_scores(candidates, field):
    scored_items = [
        item
        for item in candidates
        if isinstance(item.get(field), (int, float))
    ]

    if not scored_items:
        return {}

    values = [item[field] for item in scored_items]
    low = min(values)
    high = max(values)

    if high == low:
        return {
            id(item): 1.0
            for item in scored_items
        }

    return {
        id(item): (item[field] - low) / (high - low)
        for item in scored_items
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


class WeightedReranker(BaseReranker):
    def rerank(self, query, candidates, top_k):
        if not candidates or top_k <= 0:
            return []

        bm25_norm = normalize_scores(candidates, "bm25_score")
        vector_norm = normalize_scores(candidates, "vector_score")
        rrf_norm = normalize_scores(candidates, "rrf_score")

        ranked = []

        for index, item in enumerate(candidates, start=1):
            lexical_score = lexical_overlap_score(query, item)

            enriched = dict(item)
            enriched["pre_rerank_rank"] = item.get("rank", index)
            enriched["rerank_score"] = (
                0.40 * rrf_norm.get(id(item), 0.0)
                + 0.30 * vector_norm.get(id(item), 0.0)
                + 0.15 * bm25_norm.get(id(item), 0.0)
                + 0.15 * lexical_score
            )
            enriched["rerank_backend"] = "weighted"
            enriched["rerank_features"] = {
                "mode": "weighted",
                "rrf_norm": rrf_norm.get(id(item), 0.0),
                "vector_norm": vector_norm.get(id(item), 0.0),
                "bm25_norm": bm25_norm.get(id(item), 0.0),
                "lexical_overlap": lexical_score,
            }

            ranked.append(enriched)

        ranked.sort(
            key=lambda item: (
                item.get("rerank_score") or 0,
                item.get("rrf_score") or 0,
                item.get("vector_score") or 0,
            ),
            reverse=True,
        )

        return add_final_ranks(ranked, top_k)