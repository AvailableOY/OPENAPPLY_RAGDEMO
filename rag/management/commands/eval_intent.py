import json
from pathlib import Path

from django.core.management.base import BaseCommand
from django.utils import timezone

from rag.services.intent_classifier import classify_intent
from rag.services.query_filters import extract_query_filters
from rag.services.retrieval_strategy import build_retrieval_strategy


class Command(BaseCommand):
    help = "Evaluate intent classification and query filters without running RAG generation."

    def add_arguments(self, parser):
        parser.add_argument(
            "--input",
            type=str,
            default="data/eval/rag_test_questions.json",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="data/eval/intent_eval_results.json",
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

            intent_result = classify_intent(question)
            filters = extract_query_filters(question)
            strategy = build_retrieval_strategy(intent_result["intent"], filters)

            checks = {
                "intent_match": check_intent(case, intent_result["intent"]),
                "filters_match": check_filters(case, filters),
            }

            passed = all(checks.values())

            row = {
                "id": case_id,
                "question": question,
                "expected_intent": case.get("expected_intent"),
                "actual_intent": intent_result["intent"],
                "expected_filters": case.get("expected_filters", {}),
                "actual_filters": filters,
                "intent_result": intent_result,
                "strategy": strategy,
                "checks": checks,
                "passed": passed,
            }

            results.append(row)

            status = "PASS" if passed else "FAIL"

            self.stdout.write("")
            self.stdout.write(f"===== {case_id} =====")
            self.stdout.write(question)
            self.stdout.write(self.style.SUCCESS(status) if passed else self.style.ERROR(status))
            self.stdout.write(f"expected_intent: {case.get('expected_intent')}")
            self.stdout.write(f"actual_intent: {intent_result['intent']}")
            self.stdout.write(f"method: {intent_result.get('method')}")

            if intent_result.get("method") != "rule":
                self.stdout.write(f"confidence: {intent_result.get('confidence')}")

            if intent_result.get("reason"):
                self.stdout.write(f"reason: {intent_result.get('reason')}")

            self.stdout.write(f"expected_filters: {case.get('expected_filters', {})}")
            self.stdout.write(f"actual_filters: {filters}")

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
                f"Intent eval done. passed={summary['passed']}/{summary['total']}, output={output_path}"
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