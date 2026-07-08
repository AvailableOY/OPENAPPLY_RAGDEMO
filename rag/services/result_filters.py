import json
import re


def parse_json_list_if_possible(value):
    if not isinstance(value, str):
        return value

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value

    return parsed


def metadata_value_contains(value, expected):
    if not value:
        return False

    value = parse_json_list_if_possible(value)

    if isinstance(value, list):
        return any(expected in str(item) for item in value)

    return expected in str(value)


def parse_float_from_text(value):
    match = re.search(r"[0-9]+(?:\.[0-9]+)?", str(value))
    if not match:
        return None

    return float(match.group(0))


def parse_cost_wan(value):
    match = re.search(r"([0-9]+)\s*万", str(value))
    if not match:
        return None

    return int(match.group(1))


def build_searchable_text(item, metadata):
    fields = [
        metadata.get("programme", ""),
        metadata.get("clusters", ""),
        metadata.get("clusters_enriched", ""),
        metadata.get("value_tags", ""),
        metadata.get("value_tags_enriched", ""),
        item.get("content", ""),
    ]

    return " ".join(str(field) for field in fields)


def match_major(searchable_text, major):
    if major == "计算机":
        keywords = [
            "计算机",
            "CS",
            "Computer Science",
            "Engineering/CS",
            "engineering, CS",
        ]
        return any(keyword in searchable_text for keyword in keywords)

    return major in searchable_text


def apply_metadata_filters(candidates, filters):
    if not filters:
        return candidates

    filtered = []

    for item in candidates:
        metadata = item.get("metadata") or {}

        if filters.get("country"):
            if metadata.get("country") != filters["country"]:
                continue

        if filters.get("value_tag"):
            value_tags = metadata.get("value_tags") or metadata.get("value_tags_enriched")
            if not metadata_value_contains(value_tags, filters["value_tag"]):
                continue

        if filters.get("ielts_max") is not None:
            school_ielts = parse_float_from_text(metadata.get("ielts", ""))

            if school_ielts is not None and school_ielts > filters["ielts_max"]:
                continue

        if filters.get("budget_max_rmb_wan") is not None:
            school_cost = parse_cost_wan(metadata.get("cost", ""))

            if school_cost is not None and school_cost > filters["budget_max_rmb_wan"]:
                continue

        if filters.get("major"):
            searchable_text = build_searchable_text(item, metadata)

            if not match_major(searchable_text, filters["major"]):
                continue

        filtered.append(item)

    return filtered