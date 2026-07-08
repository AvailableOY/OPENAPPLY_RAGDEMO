import json
from pathlib import Path

from django.core.management.base import BaseCommand

from rag.services.loaders import load_markdown_file
from rag.services.splitter import split_school_markdown


class Command(BaseCommand):
    help = "Build processed JSONL chunks from OpenApply markdown."

    def add_arguments(self, parser):
        parser.add_argument("input_path", type=str)
        parser.add_argument(
            "--output",
            type=str,
            default="data/processed/openapply_school_chunks.jsonl",
        )

    def handle(self, *args, **options):
        input_path = options["input_path"]
        output_path = Path(options["output"])

        output_path.parent.mkdir(parents=True, exist_ok=True)

        loaded = load_markdown_file(input_path)

        chunks = split_school_markdown(
            text=loaded["text"],
            source_file=loaded["file_name"],
        )

        with output_path.open("w", encoding="utf-8") as file:
            for index, chunk in enumerate(chunks, start=1):
                chunk_id = f"openapply_school_{index:04d}"

                row = {
                    "id": chunk_id,
                    "content": chunk["content"],
                    "raw_content": chunk["raw_content"],
                    "source_ref": chunk["source_ref"],
                    "metadata": {
                        **chunk["metadata"],
                        "chunk_id": chunk_id,
                        "chunk_index": index,
                        "source_file": loaded["file_name"],
                        "source_path": loaded["source_path"],
                        "file_hash": loaded["file_hash"],
                    },
                }

                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        self.stdout.write(
            self.style.SUCCESS(
                f"切分完成：共生成 {len(chunks)} 个 chunk，输出文件：{output_path}"
            )
        )