import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from django.conf import settings

from rag.models import DocumentChunk


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


def tokenize(text):
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text or "")]
    chinese_chars = [token for token in tokens if len(token) == 1 and "\u4e00" <= token <= "\u9fff"]
    chinese_bigrams = [
        chinese_chars[index] + chinese_chars[index + 1]
        for index in range(len(chinese_chars) - 1)
    ]
    return tokens + chinese_bigrams


def load_bm25_documents():
    jsonl_path = Path(settings.RAG_DATA_DIR) / "processed" / "openapply_school_chunks.jsonl"
    if jsonl_path.exists():
        documents = []
        with jsonl_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                metadata = row.get("metadata", {})
                source_ref = row.get("source_ref", metadata.get("source_ref", ""))
                documents.append(
                    {
                        "key": row.get("id") or metadata.get("chunk_id") or source_ref,
                        "content": row.get("content", ""),
                        "source_ref": source_ref,
                        "metadata": {
                            **metadata,
                            "source_ref": source_ref,
                        },
                        "chunk": None,
                    }
                )

        return documents

    chunks = list(
        DocumentChunk.objects.select_related("document")
        .order_by("document_id", "chunk_index")
    )

    return [
        {
            "key": chunk.vector_id
            or chunk.metadata.get("chunk_id")
            or chunk.source_ref
            or f"db:{chunk.id}",
            "content": chunk.content,
            "source_ref": chunk.source_ref,
            "metadata": {
                **(chunk.metadata or {}),
                "chunk_db_id": chunk.id,
                "document_id": chunk.document_id,
                "source_ref": chunk.source_ref,
            },
            "chunk": chunk,
        }
        for chunk in chunks
    ]


@lru_cache(maxsize=1)
def get_bm25_index():
    documents = load_bm25_documents()
    tokenized_documents = []
    document_frequency = Counter()
    total_length = 0

    for document in documents:
        tokens = tokenize(document["content"])
        token_counts = Counter(tokens)
        tokenized_documents.append(
            {
                **document,
                "tokens": tokens,
                "token_counts": token_counts,
                "length": len(tokens),
            }
        )
        total_length += len(tokens)
        document_frequency.update(token_counts.keys())

    document_count = len(tokenized_documents)
    average_length = total_length / document_count if document_count else 0

    return {
        "documents": tokenized_documents,
        "document_frequency": document_frequency,
        "document_count": document_count,
        "average_length": average_length,
    }


def reset_bm25_index_cache():
    get_bm25_index.cache_clear()


def bm25_score(query_tokens, document, document_frequency, document_count, average_length):
    if not query_tokens or not document["length"] or not document_count:
        return 0.0

    k1 = 1.5
    b = 0.75
    score = 0.0

    for token in set(query_tokens):
        term_frequency = document["token_counts"].get(token, 0)
        if term_frequency == 0:
            continue

        df = document_frequency.get(token, 0)
        idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
        denominator = term_frequency + k1 * (
            1 - b + b * document["length"] / (average_length or 1)
        )
        score += idf * (term_frequency * (k1 + 1)) / denominator

    return score


def search_bm25(query, top_k=20):
    index = get_bm25_index()
    query_tokens = tokenize(query)

    scored_items = []
    for document in index["documents"]:
        score = bm25_score(
            query_tokens=query_tokens,
            document=document,
            document_frequency=index["document_frequency"],
            document_count=index["document_count"],
            average_length=index["average_length"],
        )
        if score <= 0:
            continue

        scored_items.append(
            {
                "key": document["key"],
                "content": document["content"],
                "source_ref": document["source_ref"],
                "metadata": document["metadata"],
                "chunk": document["chunk"],
                "bm25_score": score,
            }
        )

    scored_items.sort(key=lambda item: item["bm25_score"], reverse=True)
    return scored_items[:top_k]
