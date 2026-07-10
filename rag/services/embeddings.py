import logging
from pathlib import Path
from threading import Lock

from django.conf import settings
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

_embedding_model = None
_embedding_model_lock = Lock()


def ensure_embedding_model():
    model_path = Path(settings.EMBEDDING_MODEL_PATH)
    config_path = model_path / "config.json"

    if config_path.exists():
        return model_path

    logger.info(
        "Embedding model not found locally. "
        "Downloading %s to %s",
        settings.EMBEDDING_MODEL_ID,
        model_path,
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_download(
        repo_id=settings.EMBEDDING_MODEL_ID,
        local_dir=str(model_path),
    )

    if not config_path.exists():
        raise RuntimeError(
            f"Embedding model download incomplete: {model_path}"
        )

    logger.info(
        "Embedding model downloaded successfully: %s",
        model_path,
    )

    return model_path


def get_embedding_model():
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    # 防止多个评测线程在模型尚未下载或加载时重复初始化。
    with _embedding_model_lock:
        if _embedding_model is not None:
            return _embedding_model

        model_path = ensure_embedding_model()

        _embedding_model = SentenceTransformer(
            str(model_path)
        )

    return _embedding_model


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