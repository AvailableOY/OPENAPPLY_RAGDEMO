import json
from pathlib import Path

import chromadb
from django.conf import settings

from rag.services.embeddings import embed_query, embed_texts


COLLECTION_NAME = "openapply_school_knowledge"


def load_jsonl_chunks(jsonl_path):
    path = Path(jsonl_path)
    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    return chunks


def get_chroma_client():
    persist_directory = Path(settings.RAG_VECTOR_DIR) / "chroma"
    persist_directory.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(path=str(persist_directory))


def get_collection():
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_vector_store_from_jsonl(jsonl_path):
    chunks = load_jsonl_chunks(jsonl_path)
    collection = get_collection()

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["id"])
        documents.append(chunk["content"])

        metadata = {
            **chunk.get("metadata", {}),
            "source_ref": chunk.get("source_ref", ""),
        }

        metadatas.append(sanitize_metadata(metadata))

    embeddings = embed_texts(documents)

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    return {
        "chunk_count": len(ids),
        "collection_name": COLLECTION_NAME,
        "persist_directory": str(Path(settings.RAG_VECTOR_DIR) / "chroma"),
    }


def search_vector_store(query, top_k=5):
    collection = get_collection()
    query_embedding = embed_query(query)

    candidate_k = max(top_k * 4, 20)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    items = []

    for index, document in enumerate(result["documents"][0]):
        items.append(
            {
                "key": result["ids"][0][index],
                "content": document,
                "metadata": result["metadatas"][0][index],
                "source_ref": result["metadatas"][0][index].get("source_ref", ""),
                "distance": result["distances"][0][index],
                "vector_score": 1 - result["distances"][0][index],
            }
        )

    query_ielts = parse_ielts_from_query(query)

    if query_ielts is not None:
        matched = []
        unmatched = []

        for item in items:
            school_ielts = parse_ielts_from_metadata(item["metadata"])
            item["school_ielts"] = school_ielts

            if school_ielts is not None and school_ielts <= query_ielts:
                item["filter_match"] = True
                matched.append(item)
            else:
                item["filter_match"] = False
                unmatched.append(item)

        matched.sort(
            key=lambda item: (
                abs(item["school_ielts"] - query_ielts),
                item["distance"],
            )
        )

        items = matched + unmatched

    return items[:top_k]

def sanitize_metadata(metadata):
    clean = {}

    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = json.dumps(value, ensure_ascii=False)

    return clean

import re


def parse_ielts_from_query(query):
    match = re.search(r"(?:雅思|IELTS)\s*([0-9](?:\.[0-9])?)", query, re.IGNORECASE)
    if not match:
        return None

    return float(match.group(1))

def parse_ielts_from_metadata(metadata):
    value = metadata.get("ielts", "")
    match = re.search(r"[0-9](?:\.[0-9])?", str(value))
    if not match:
        return None

    return float(match.group(0))
