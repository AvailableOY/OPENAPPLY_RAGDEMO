import csv
import json
import time
from datetime import datetime
from pathlib import Path

from django.core.management.base import BaseCommand

from rag.services.rag_chain import run_rag_query


class Command(BaseCommand):
    help = "测试外部测试集，支持 JSON, JSONL, CSV 格式。"

    def add_arguments(self, parser):
        parser.add_argument(
            "--input",
            required=True,
            help="Input testset file. Supports JSON, JSONL, and CSV.",
        )
        parser.add_argument(
            "--output-dir",
            default=None,
            help="Directory to save summary.json and details.jsonl.",
        )
        parser.add_argument(
            "--session-prefix",
            default=None,
            help="Session id prefix for saved QueryLog records.",
        )

    def handle(self, *args, **options):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = Path(options["input"])

        if not input_path.is_absolute():
            input_path = Path.cwd() / input_path

        output_dir = resolve_output_dir(options["output_dir"], run_id)
        session_prefix = options["session_prefix"] or f"external-{run_id}"

        cases = load_cases(input_path)

        if not cases:
            raise ValueError(
                "No valid test cases found. "
                "Expected question/query/input/text field."
            )

        run_cases(
            command=self,
            cases=cases,
            output_dir=output_dir,
            session_prefix=session_prefix,
        )


def resolve_output_dir(output_dir, run_id):
    if output_dir:
        path = Path(output_dir)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    return (
        Path.cwd()
        / "output"
        / "eval"
        / f"external_run_{run_id}"
    )


def load_cases(input_path):
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        return load_json_cases(path)

    if suffix == ".jsonl":
        return load_jsonl_cases(path)

    if suffix == ".csv":
        return load_csv_cases(path)

    raise ValueError(f"Unsupported input file type: {suffix}")


def load_json_cases(path):
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        data = (
            data.get("cases")
            or data.get("questions")
            or data.get("results")
            or []
        )

    cases = []

    for index, item in enumerate(data, start=1):
        if isinstance(item, str):
            cases.append(
                {
                    "id": f"case_{index:03d}",
                    "question": item,
                }
            )
            continue

        if isinstance(item, dict):
            question = extract_question(item)
            if question:
                cases.append(
                    {
                        **item,
                        "id": item.get("id") or f"case_{index:03d}",
                        "question": question,
                    }
                )

    return cases


def load_jsonl_cases(path):
    cases = []

    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            question = extract_question(item)

            if question:
                cases.append(
                    {
                        **item,
                        "id": item.get("id") or f"case_{index:03d}",
                        "question": question,
                    }
                )

    return cases


def load_csv_cases(path):
    cases = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for index, row in enumerate(reader, start=1):
            question = extract_question(row)

            if question:
                cases.append(
                    {
                        **row,
                        "id": row.get("id") or f"case_{index:03d}",
                        "question": question,
                    }
                )

    return cases


def extract_question(item):
    return (
        item.get("question")
        or item.get("query")
        or item.get("input")
        or item.get("text")
    )


def run_cases(command, cases, output_dir, session_prefix):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    details_path = output_path / "details.jsonl"
    summary_path = output_path / "summary.json"

    results = []
    started_at = datetime.now().isoformat()

    with details_path.open("w", encoding="utf-8") as details_file:
        for index, case in enumerate(cases, start=1):
            row = run_one_case(
                command=command,
                index=index,
                case=case,
                session_prefix=session_prefix,
            )

            results.append(row)
            details_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            details_file.flush()

    summary = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "total": len(results),
        "success": sum(1 for item in results if not item.get("error")),
        "failed": sum(1 for item in results if item.get("error")),
        "details_file": str(details_path),
        "results": results,
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    command.stdout.write("")
    command.stdout.write(
        command.style.SUCCESS(
            f"Done. total={summary['total']}, "
            f"success={summary['success']}, failed={summary['failed']}"
        )
    )
    command.stdout.write(f"summary: {summary_path}")
    command.stdout.write(f"details: {details_path}")


def run_one_case(command, index, case, session_prefix):
    case_id = case.get("id") or f"case_{index:03d}"
    question = case["question"]
    session_id = f"{session_prefix}-{case_id}"

    command.stdout.write("")
    command.stdout.write(f"===== {case_id} =====")
    command.stdout.write(question)

    started = time.time()

    try:
        result = run_rag_query(
            question=question,
            session_id=session_id,
        )

        contexts = result.get("contexts", [])
        source_refs = [
            item.get("source_ref", "")
            for item in contexts
        ]

        row = {
            "id": case_id,
            "question": question,
            "query_id": result.get("query_id"),
            "rewritten_query": result.get("rewritten_query"),
            "intent_result": result.get("intent_result", {}),
            "filters": result.get("filters", {}),
            "strategy": result.get("strategy", {}),
            "sources": source_refs,
            "answer": result.get("answer"),
            "judge": result.get("judge"),
            "attempt": result.get("attempt"),
            "elapsed_seconds": round(time.time() - started, 2),
            "error": None,
        }

        command.stdout.write(f"query_id: {row['query_id']}")
        command.stdout.write(f"intent: {row['intent_result'].get('intent')}")
        command.stdout.write(f"filters: {row['filters']}")
        command.stdout.write(f"sources: {row['sources'][:5]}")
        command.stdout.write(f"judge: {row['judge']}")

        return row

    except Exception as exc:
        row = {
            "id": case_id,
            "question": question,
            "query_id": None,
            "elapsed_seconds": round(time.time() - started, 2),
            "error": repr(exc),
        }

        command.stdout.write(command.style.ERROR(f"ERROR: {row['error']}"))
        return row