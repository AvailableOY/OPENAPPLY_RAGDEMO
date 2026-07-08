import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        parser.add_argument(
            "--workers",
            type=int,
            default=1,
            help="多线程测评",
        )

    def handle(self, *args, **options):
        input_path = Path(options["input"])
        output_path = Path(options["output"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cases = json.loads(input_path.read_text(encoding="utf-8"))
        results = []

        workers = max(1, options["workers"])

        if workers == 1:
            for index, case in enumerate(cases, start=1):
                row = run_case(
                    index=index,
                    case=case,
                    session_prefix=options["session_prefix"],
                )
                results.append(row)
                print_case_result(self, row)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_index = {
                    executor.submit(
                        run_case,
                        index,
                        case,
                        options["session_prefix"],
                    ): index
                    for index, case in enumerate(cases, start=1)
                }

                for future in as_completed(future_to_index):
                    row = future.result()
                    results.append(row)
                    print_case_result(self, row)

        results.sort(key=lambda item: item["id"])

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

def run_case(index, case, session_prefix):
    case_id = case.get("id") or f"case_{index:03d}"
    question = case["question"]
    session_id = f"{session_prefix}-{case_id}"

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

    return {
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

def print_case_result(command, row):
    status = "PASS" if row["passed"] else "FAIL"

    command.stdout.write("")
    command.stdout.write(f"===== {row['id']} =====")
    command.stdout.write(row["question"])
    command.stdout.write(
        command.style.SUCCESS(status)
        if row["passed"]
        else command.style.ERROR(status)
    )
    command.stdout.write(f"intent: {row['actual_intent']}")
    command.stdout.write(f"filters: {row['actual_filters']}")
    command.stdout.write(f"sources: {row['actual_sources'][:5]}")


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


