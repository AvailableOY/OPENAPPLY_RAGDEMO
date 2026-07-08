from django.core.management.base import BaseCommand

from rag.services.vector_store import search_vector_store


class Command(BaseCommand):
    help = "Search local Chroma vector store."

    def add_arguments(self, parser):
        parser.add_argument("query", type=str)
        parser.add_argument("--top-k", type=int, default=5)

    def handle(self, *args, **options):
        results = search_vector_store(
            query=options["query"],
            top_k=options["top_k"],
        )

        self.stdout.write(self.style.SUCCESS(f"检索完成，共返回 {len(results)} 条结果"))

        for index, item in enumerate(results, start=1):
            metadata = item.get("metadata", {})

            self.stdout.write("")
            self.stdout.write(f"===== Top {index} =====")
            self.stdout.write(f"vector_score：{item.get('vector_score')}")
            self.stdout.write(f"distance：{item.get('distance')}")
            self.stdout.write(f"学校：{metadata.get('school_name_zh')} / {metadata.get('school_name_en')}")
            self.stdout.write(f"国家：{metadata.get('country')}")
            self.stdout.write(f"来源：{metadata.get('source_ref')}")
            self.stdout.write(item.get("content", "")[:500])
