import logging

import requests
from django.conf import settings

from rag.services.rerankers.base import BaseReranker, add_final_ranks
from rag.services.rerankers.weighted import WeightedReranker


logger = logging.getLogger(__name__)


class QwenReranker(BaseReranker):
    def __init__(self):
        self.api_url = settings.RERANK_API_URL
        self.api_key = settings.RERANK_API_KEY
        self.model = settings.RERANK_MODEL
        self.timeout = settings.RERANK_TIMEOUT_SECONDS
        self.candidate_top_k = settings.RERANK_CANDIDATE_TOP_K
        self.fallback = WeightedReranker()

    def rerank(self, query, candidates, top_k):
        if not candidates or top_k <= 0:
            return []

        # 只将前 N 个 RRF 候选发送给云端精排。
        head = candidates[: self.candidate_top_k]
        tail = candidates[self.candidate_top_k :]

        try:
            ranked_head = self._call_qwen(query, head)
        except Exception as exc:
            logger.exception(
                "Qwen rerank failed; falling back to weighted reranker"
            )

            fallback_result = self.fallback.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )

            for item in fallback_result:
                features = item.get("rerank_features") or {}
                item["rerank_features"] = {
                    **features,
                    "mode": "weighted_fallback",
                    "fallback_reason": type(exc).__name__,
                }

            return fallback_result

        ranked = ranked_head + self._preserve_tail(tail)
        return add_final_ranks(ranked, top_k)

    def _call_qwen(self, query, candidates):
        documents = [
            item.get("content", "")
            for item in candidates
        ]

        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
                "instruct": (
                    "Given an international education question, "
                    "rank passages by whether they directly provide "
                    "evidence needed to answer the question."
                ),
            },
            timeout=self.timeout,
        )

        response.raise_for_status()
        payload = response.json()

        results = payload.get("results")

        if not isinstance(results, list):
            raise ValueError("Qwen rerank response is missing results")

        ranked = []
        used_indexes = set()

        for rank, result in enumerate(results, start=1):
            original_index = result.get("index")

            if not isinstance(original_index, int):
                raise ValueError("Invalid document index in rerank response")

            if original_index < 0 or original_index >= len(candidates):
                raise ValueError("Rerank document index is out of range")

            if original_index in used_indexes:
                raise ValueError("Duplicate document index in rerank response")

            used_indexes.add(original_index)

            item = dict(candidates[original_index])
            item["pre_rerank_rank"] = item.get(
                "rank",
                original_index + 1,
            )
            item["rerank_rank"] = rank
            item["rerank_score"] = result.get("relevance_score")
            item["rerank_backend"] = self.model
            item["rerank_features"] = {
                "mode": "qwen",
                "original_index": original_index,
            }

            ranked.append(item)

        # 如果 API 少返回了文档，剩余文档按原 RRF 顺序放到后面。
        for index, item in enumerate(candidates):
            if index in used_indexes:
                continue

            enriched = dict(item)
            enriched["pre_rerank_rank"] = item.get("rank", index + 1)
            enriched["rerank_score"] = item.get("rrf_score")
            enriched["rerank_backend"] = self.model
            enriched["rerank_features"] = {
                "mode": "qwen_unreturned_preserve_order",
            }

            ranked.append(enriched)

        return ranked

    def _preserve_tail(self, candidates):
        result = []

        for item in candidates:
            enriched = dict(item)
            enriched["pre_rerank_rank"] = item.get("rank")
            enriched["rerank_score"] = item.get("rrf_score")
            enriched["rerank_backend"] = self.model
            enriched["rerank_features"] = {
                "mode": "qwen_tail_preserve_order",
            }
            result.append(enriched)

        return result