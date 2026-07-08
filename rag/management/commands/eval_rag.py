import json
from pathlib import Path

from django.core.management.base import BaseCommand
from django.utils import timezone

from rag.services.rag_chain import run_rag_query


class Command(BaseCommand):
    help = "Run RAG evaluation cases from a JSON file."

    def add_arguments(self, parser):
        parser.add_argument(
            "--input",
            type=str,
            default="data/eval/rag_test_questions.json",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="data/eval/rag_eval_results.json",
        )
        parser.add_argument(
            "--session-prefix",
            type=str,
            default="eval",
        )

    def handle(self, *args, **options):
        input_path = Path(options["input"])
        output_path = Path(options["output"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cases = json.loads(input_path.read_text(encoding="utf-8"))
        results = []

        for index, case in enumerate(cases, start=1):
            case_id = case.get("id") or f"case_{index:03d}"
            question = case["question"]
            session_id = f"{options['session_prefix']}-{case_id}"

            self.stdout.write("")
            self.stdout.write(f"===== {case_id} =====")
            self.stdout.write(question)

            result = run_rag_query(
                question=question,
                session_id=session_id,
            )

            contexts = result.get("contexts", [])
            source_refs = [
                item.get("source_ref", "")
                for item in contexts
            ]

            actual_intent = result.get("intent_result", {}).get("intent")
            actual_filters = result.get("filters", {})

            checks = {
                "intent_match": check_intent(case, actual_intent),
                "filters_match": check_filters(case, actual_filters),
                "expected_source_hit": check_expected_sources(case, source_refs),
                "judge_pass": result.get("judge", {}).get("pass") is True,
            }

            passed = all(checks.values())

            row = {
                "id": case_id,
                "question": question,
                "expected_intent": case.get("expected_intent"),
                "actual_intent": actual_intent,
                "expected_filters": case.get("expected_filters", {}),
                "actual_filters": actual_filters,
                "expected_sources_any": case.get("expected_sources_any", []),
                "actual_sources": source_refs,
                "checks": checks,
                "passed": passed,
                "rewritten_query": result.get("rewritten_query"),
                "answer": result.get("answer"),
                "judge": result.get("judge"),
                "intent_result": result.get("intent_result", {}),
                "strategy": result.get("strategy", {}),
            }

            results.append(row)

            status = "PASS" if passed else "FAIL"
            self.stdout.write(self.style.SUCCESS(status) if passed else self.style.ERROR(status))
            self.stdout.write(f"intent: {actual_intent}")
            self.stdout.write(f"filters: {actual_filters}")
            self.stdout.write(f"sources: {source_refs[:5]}")

        summary = {
            "run_at": timezone.now().isoformat(),
            "total": len(results),
            "passed": sum(1 for item in results if item["passed"]),
            "failed": sum(1 for item in results if not item["passed"]),
            "results": results,
        }

        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                f"Eval done. passed={summary['passed']}/{summary['total']}, output={output_path}"
            )
        )


def check_intent(case, actual_intent):
    expected_intent = case.get("expected_intent")
    if not expected_intent:
        return True

    return actual_intent == expected_intent


def check_filters(case, actual_filters):
    expected_filters = case.get("expected_filters") or {}

    for key, expected_value in expected_filters.items():
        actual_value = actual_filters.get(key)

        if actual_value != expected_value:
            return False

    return True


def check_expected_sources(case, actual_sources):
    expected_sources = case.get("expected_sources_any") or []

    if not expected_sources:
        return True

    return any(source in actual_sources for source in expected_sources)