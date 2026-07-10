from abc import ABC, abstractmethod


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query, candidates, top_k):
        """返回重新排序后的候选列表。"""
        raise NotImplementedError


def add_final_ranks(candidates, top_k):
    result = []

    for index, item in enumerate(candidates, start=1):
        enriched = dict(item)
        enriched["rerank_rank"] = index
        result.append(enriched)

    return result[:top_k]