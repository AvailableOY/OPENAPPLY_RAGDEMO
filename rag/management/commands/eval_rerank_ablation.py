import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

from django.core.management.base import BaseCommand
from django.db import close_old_connections
from django.test.utils import override_settings
from django.utils import timezone

from rag.services.rag_chain import run_rag_query


BACKENDS = ["weighted", "qwen"]


class Command(BaseCommand):
    help = "Compare weighted rerank with Qwen3-Rerank."

    def add_arguments(self, parser):
        parser.add_argument(
            "--input",
            default="data/eval/rag_test_questions.json",
        )
        parser.add_argument(
            "--output",
            default="data/eval/rerank_ablation_results.json",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=3,
            help="Number of concurrent evaluation workers per backend.",
        )
        parser.add_argument(
            "--backends",
            nargs="+",
            choices=BACKENDS,
            default=BACKENDS,
            help="Only run selected rerank backends.",
        )

        parser.add_argument(
            "--case-ids",
            nargs="+",
            default=None,
            help="Only run selected case IDs.",
        )

    def handle(self, *args, **options):
        input_path = Path(options["input"])
        output_path = Path(options["output"])
        workers = max(1, options["workers"])
        backends = options["backends"]
        case_ids = options["case_ids"]

        cases = json.loads(
            input_path.read_text(encoding="utf-8")
        )

        if case_ids:
            requested_ids = set(case_ids)

            cases = [
                case
                for case in cases
                if case.get("id") in requested_ids
            ]

            found_ids = {
                case.get("id")
                for case in cases
            }

            missing_ids = requested_ids - found_ids

            if missing_ids:
                raise ValueError(
                    f"Unknown case IDs: {sorted(missing_ids)}"
                )

        if not cases:
            raise ValueError(
                "No evaluation cases selected."
            )

        experiments = []

        for backend in backends:
            self.stdout.write("")
            self.stdout.write(
                self.style.WARNING(
                    f"===== backend={backend}, workers={workers} ====="
                )
            )
            experiments.append(
                self.run_backend(
                    backend=backend,
                    cases=cases,
                    workers=workers,
                )
            )

        report = {
            "run_at": timezone.now().isoformat(),
            "input": str(input_path),
            "workers": workers,
            "backends": backends,
            "case_ids": case_ids,
            "experiments": experiments,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.print_summary(experiments)
        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(f"Results saved to {output_path}")
        )

    def run_backend(self, backend, cases, workers):
        rows = []

        # Backends run sequentially. Only cases within one backend run concurrently,
        # so every worker observes the same RERANK_BACKEND setting.
        with override_settings(RERANK_BACKEND=backend):
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self.run_case,
                        backend,
                        index,
                        case,
                    ): case.get("id") or f"case_{index:03d}"
                    for index, case in enumerate(cases, start=1)
                }

                for future in as_completed(futures):
                    case_id = futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = self.build_worker_error_row(
                            backend=backend,
                            case_id=case_id,
                            exc=exc,
                        )

                    rows.append(row)
                    self.print_case(row)

        rows.sort(key=lambda item: item["id"])
        return {
            "backend": backend,
            "summary": self.build_summary(rows),
            "results": rows,
        }

    def run_case(self, backend, index, case):
        case_id = case.get("id") or f"case_{index:03d}"
        question = case["question"]
        session_id = f"rerank-ablation-{backend}-{case_id}-{timezone.now().timestamp()}"
        started_at = perf_counter()
        close_old_connections()

        try:
            result = run_rag_query(
                question=question,
                session_id=session_id,
            )
            error = None
        except Exception as exc:
            result = {}
            error = f"{type(exc).__name__}: {exc}"
        finally:
            # Each worker owns its Django DB connection. Closing it here avoids
            # leaking thread-local SQLite connections across pooled tasks.
            close_old_connections()

        latency = perf_counter() - started_at
        contexts = result.get("contexts", [])
        actual_sources = [
            item.get("source_ref", "")
            for item in contexts
        ]
        expected_sources = case.get("expected_sources_any") or []
        has_source_labels = bool(expected_sources)
        first_relevant_rank = find_first_relevant_rank(
            expected_sources=expected_sources,
            actual_sources=actual_sources,
        )

        hit_at_5 = (
            any(
                source in actual_sources[:5]
                for source in expected_sources
            )
            if has_source_labels
            else None
        )
        reciprocal_rank = (
            1 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )
        judge_pass = result.get("judge", {}).get("pass") is True
        fallback_used = any(
            (item.get("rerank_features") or {}).get("mode")
            == "weighted_fallback"
            for item in contexts
        )

        if error:
            passed = False
        elif has_source_labels:
            passed = bool(hit_at_5 and judge_pass)
        else:
            passed = judge_pass

        return {
            "id": case_id,
            "backend": backend,
            "question": question,
            "query_id": result.get("query_id"),
            "expected_sources": expected_sources,
            "actual_sources": actual_sources,
            "has_source_labels": has_source_labels,
            "hit_at_5": hit_at_5,
            "first_relevant_rank": first_relevant_rank,
            "reciprocal_rank": round(reciprocal_rank, 4),
            "judge_pass": judge_pass,
            "fallback_used": fallback_used,
            "latency_seconds": round(latency, 4),
            "passed": passed,
            "error": error,
        }

    def build_worker_error_row(self, backend, case_id, exc):
        return {
            "id": case_id,
            "backend": backend,
            "question": "",
            "query_id": None,
            "expected_sources": [],
            "actual_sources": [],
            "has_source_labels": False,
            "hit_at_5": None,
            "first_relevant_rank": None,
            "reciprocal_rank": 0.0,
            "judge_pass": False,
            "fallback_used": False,
            "latency_seconds": 0.0,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    def print_case(self, row):
        status = "PASS" if row["passed"] else "FAIL"
        self.stdout.write(
            f"{row['backend']} {row['id']} {status} "
            f"hit@5={row['hit_at_5']} "
            f"rr={row['reciprocal_rank']:.3f} "
            f"judge={row['judge_pass']} "
            f"time={row['latency_seconds']:.2f}s "
            f"fallback={row['fallback_used']}"
        )

    def build_summary(self, rows):
        total = len(rows)
        labeled_rows = [
            row for row in rows if row["has_source_labels"]
        ]
        labeled_count = len(labeled_rows)

        hit_at_5 = (
            sum(row["hit_at_5"] is True for row in labeled_rows)
            / labeled_count
            if labeled_count
            else 0.0
        )
        mrr = (
            sum(row["reciprocal_rank"] for row in labeled_rows)
            / labeled_count
            if labeled_count
            else 0.0
        )
        judge_pass_rate = (
            sum(row["judge_pass"] for row in rows) / total
            if total
            else 0.0
        )
        pass_rate = (
            sum(row["passed"] for row in rows) / total
            if total
            else 0.0
        )
        average_latency = (
            sum(row["latency_seconds"] for row in rows) / total
            if total
            else 0.0
        )

        return {
            "total_cases": total,
            "labeled_cases": labeled_count,
            "passed_cases": sum(row["passed"] for row in rows),
            "pass_rate": round(pass_rate, 4),
            "hit_at_5": round(hit_at_5, 4),
            "mrr": round(mrr, 4),
            "judge_pass_rate": round(judge_pass_rate, 4),
            "average_latency_seconds": round(average_latency, 4),
            "fallback_count": sum(row["fallback_used"] for row in rows),
            "error_count": sum(bool(row["error"]) for row in rows),
        }

    def print_summary(self, experiments):
        self.stdout.write("")
        self.stdout.write("===== Rerank ablation summary =====")

        for experiment in experiments:
            summary = experiment["summary"]
            self.stdout.write(
                f"{experiment['backend']}: "
                f"pass={summary['passed_cases']}/{summary['total_cases']}, "
                f"Hit@5={summary['hit_at_5']:.3f}, "
                f"MRR={summary['mrr']:.3f}, "
                f"judge={summary['judge_pass_rate']:.3f}, "
                f"latency={summary['average_latency_seconds']:.2f}s, "
                f"fallback={summary['fallback_count']}, "
                f"errors={summary['error_count']}"
            )


def find_first_relevant_rank(expected_sources, actual_sources):
    if not expected_sources:
        return None

    for rank, source in enumerate(actual_sources, start=1):
        if source in expected_sources:
            return rank

    return None
