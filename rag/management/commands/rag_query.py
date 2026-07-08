from django.core.management.base import BaseCommand

from rag.services.rag_chain import run_rag_query


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument("question", type=str)
        parser.add_argument("--session-id", type=str, default="demo-session")

    def handle(self, *args, **options):
        # 命令行调试入口，与 Postman API 走同一条 pipeline。
        result = run_rag_query(
            question=options["question"],
            session_id=options["session_id"],
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"LangChain RAG query saved. query_id={result['query_id']}, attempt={result['attempt']}"
            )
        )
        self.stdout.write(f"rewritten_query: {result['rewritten_query']}")
        self.stdout.write(f"judge: {result['judge']}")
        self.stdout.write(result["answer"])
