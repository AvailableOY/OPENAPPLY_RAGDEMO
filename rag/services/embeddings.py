from functools import lru_cache

from django.conf import settings
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(settings.EMBEDDING_MODEL_PATH)


def embed_texts(texts):
    model = get_embedding_model()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return vectors.tolist()


def embed_query(query):
    model = get_embedding_model()

    vector = model.encode(
        query,
        normalize_embeddings=True,
    )

    return vector.tolist()