from pathlib import Path
import hashlib


def read_text_file(file_path):
    path = Path(file_path)

    encodings = ["utf-8-sig", "utf-8", "gb18030"]
    last_error = None

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def calculate_file_hash(file_path):
    path = Path(file_path)
    file_bytes = path.read_bytes()
    return hashlib.sha256(file_bytes).hexdigest()


def load_markdown_file(file_path):
    path = Path(file_path)
    text = read_text_file(path)

    return {
        "title": path.stem,
        "file_name": path.name,
        "file_type": path.suffix.lstrip("."),
        "source_path": str(path),
        "file_hash": calculate_file_hash(path),
        "text": text,
    }