from django.core.management.base import BaseCommand

from rag.services.hybrid_retriever import hybrid_retrieve


class Command(BaseCommand):
    help = "搜索块与BM25 +矢量混合检索"

    def add_arguments(self, parser):
        parser.add_argument("query", type=str)
        parser.add_argument("--top-k", type=int, default=5)

    def handle(self, *args, **options):
        results = hybrid_retrieve(
            query=options["query"],
            top_k=options["top_k"],
        )

        self.stdout.write(self.style.SUCCESS(f"Hybrid results: {len(results)}"))

        for index, item in enumerate(results, start=1):
            metadata = item.get("metadata", {})
            self.stdout.write("")
            self.stdout.write(f"===== Top {index} =====")
            self.stdout.write(f"rrf_score: {item.get('rrf_score')}")
            self.stdout.write(f"bm25_score: {item.get('bm25_score')}")
            self.stdout.write(f"vector_score: {item.get('vector_score')}")
            self.stdout.write(f"rank_sources: {item.get('rank_sources')}")
            self.stdout.write(
                f"school: {metadata.get('school_name_zh')} / {metadata.get('school_name_en')}"
            )
            self.stdout.write(f"country: {metadata.get('country')}")
            self.stdout.write(f"source: {item.get('source_ref')}")
            self.stdout.write(item.get("content", "")[:500])
