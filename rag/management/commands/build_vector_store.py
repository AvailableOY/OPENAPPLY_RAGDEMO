from django.core.management.base import BaseCommand

from rag.services.vector_store import build_vector_store_from_jsonl


class Command(BaseCommand):
    help = "Build local Chroma vector store from processed JSONL chunks."

    def add_arguments(self, parser):
        parser.add_argument(
            "--input",
            type=str,
            default="data/processed/openapply_school_chunks.jsonl",
        )

    def handle(self, *args, **options):
        result = build_vector_store_from_jsonl(options["input"])

        self.stdout.write(self.style.SUCCESS("向量库入库完成"))
        self.stdout.write(f"入库 chunk 数量：{result['chunk_count']}")
        self.stdout.write(f"集合名称：{result['collection_name']}")
        self.stdout.write(f"向量库目录：{result['persist_directory']}")