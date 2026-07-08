from django.core.management.base import BaseCommand

from rag.services.bm25_store import reset_bm25_index_cache, search_bm25


class Command(BaseCommand):
    help = "Search chunks with local BM25."

    def add_arguments(self, parser):
        parser.add_argument("query", type=str)
        parser.add_argument("--top-k", type=int, default=5)
        parser.add_argument("--refresh", action="store_true")

    def handle(self, *args, **options):
        if options["refresh"]:
            reset_bm25_index_cache()

        results = search_bm25(
            query=options["query"],
            top_k=options["top_k"],
        )

        self.stdout.write(self.style.SUCCESS(f"BM25 results: {len(results)}"))

        for index, item in enumerate(results, start=1):
            metadata = item.get("metadata", {})
            self.stdout.write("")
            self.stdout.write(f"===== Top {index} =====")
            self.stdout.write(f"bm25_score: {item.get('bm25_score')}")
            self.stdout.write(
                f"school: {metadata.get('school_name_zh')} / {metadata.get('school_name_en')}"
            )
            self.stdout.write(f"country: {metadata.get('country')}")
            self.stdout.write(f"source: {item.get('source_ref')}")
            self.stdout.write(item.get("content", "")[:500])
