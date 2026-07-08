import re


def extract_query_filters(question):
    filters = {}

    if "英国" in question or "UK" in question.upper():
        filters["country"] = "英国"
    elif "美国" in question or "US" in question.upper() or "USA" in question.upper():
        filters["country"] = "美国"
    elif "加拿大" in question:
        filters["country"] = "加拿大"
    elif "澳大利亚" in question or "澳洲" in question:
        filters["country"] = "澳大利亚"

    if "落户" in question or "人才政策" in question:
        filters["value_tag"] = "落户/人才政策友好"

    if "计算机" in question or "CS" in question.upper() or "Computer Science".lower() in question.lower():
        filters["major"] = "计算机"

    ielts_match = re.search(
        r"(?:雅思|IELTS)\s*([0-9](?:\.[0-9])?)",
        question,
        re.IGNORECASE,
    )
    if ielts_match:
        filters["ielts_max"] = float(ielts_match.group(1))

    budget_match = re.search(r"([0-9]+)\s*万", question)
    if budget_match and any(word in question for word in ["预算", "费用", "学费", "以内", "不超过"]):
        filters["budget_max_rmb_wan"] = int(budget_match.group(1))

    return filters