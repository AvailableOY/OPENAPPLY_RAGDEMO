import json

from langchain_core.output_parsers import StrOutputParser

from rag.services.generator import get_chat_model
from rag.services.prompts import INTENT_CLASSIFIER_PROMPT


INTENT_LIST_SCHOOLS = "list_schools"
INTENT_CONTINUE_LIST = "continue_list"
INTENT_SCHOOL_DETAIL = "school_detail"
INTENT_COMPARE_SCHOOLS = "compare_schools"
INTENT_FILTER_REQUIREMENT = "filter_requirement"
INTENT_GENERAL_QA = "general_qa"

VALID_INTENTS = {
    INTENT_LIST_SCHOOLS,
    INTENT_CONTINUE_LIST,
    INTENT_SCHOOL_DETAIL,
    INTENT_COMPARE_SCHOOLS,
    INTENT_FILTER_REQUIREMENT,
    INTENT_GENERAL_QA,
}


SCHOOL_ALIAS_KEYWORDS = [
    "UCL",
    "伦敦大学学院",
    "曼大",
    "曼彻斯特",
    "曼彻斯特大学",
    "剑桥",
    "牛津",
    "帝国理工",
    "爱丁堡",
    "华威",
    "布里斯托",
    "谢菲尔德",
    "格拉斯哥",
    "利兹",
    "杜伦",
    "LSE",
]


CONTINUE_KEYWORDS = [
    "继续",
    "还有吗",
    "下一页",
    "更多",
    "再来",
    "接着",
    "往下",
]


COMPARE_KEYWORDS = [
    "对比",
    "怎么选",
    "选哪个",
    "哪个好",
    "哪所更好",
    "哪个更",
    "更适合",
    "区别",
    "差别",
]


DETAIL_KEYWORDS = [
    "介绍",
    "详情",
    "怎么样",
    "如何",
    "申请要求",
    "语言要求",
    "费用",
    "学费",
    "项目情况",
]


LIST_KEYWORDS = [
    "哪些学校",
    "有哪些学校",
    "有哪些院校",
    "推荐",
    "适合",
    "院校列表",
    "学校名单",
    "可以申请哪些",
    "能申请哪些",
    "看看",
]


FILTER_KEYWORDS = [
    "雅思",
    "IELTS",
    "托福",
    "TOEFL",
    "预算",
    "不超过",
    "以内",
    "学费",
    "费用",
    "便宜",
    "门槛",
    "成绩一般",
    "低一点",
    "低一些",
    "好申请",
    "保底",
    "不用面试",
    "无需面试",
    "不需要面试",
    "无面试",
]

STRONG_FILTER_KEYWORDS = [
    "雅思",
    "IELTS",
    "托福",
    "TOEFL",
    "预算",
    "不超过",
    "以内",
    "门槛",
    "成绩一般",
    "不用面试",
    "无需面试",
    "不需要面试",
    "无面试",
]


def mentions_school(text):
    if any(keyword in text for keyword in ["大学", "学校", "院校"]):
        return True

    return any(alias in text for alias in SCHOOL_ALIAS_KEYWORDS)


def build_rule_result(intent, reason):
    return {
        "intent": intent,
        "method": "rule",
        "confidence":0.95,
        "reason": reason,
    }


def classify_intent_by_rules(question):
    text = question.strip()
    upper_text = text.upper()

    if any(keyword in text for keyword in CONTINUE_KEYWORDS):
        return build_rule_result(
            INTENT_CONTINUE_LIST,
            "命中继续查看更多结果的关键词",
        )

    if any(keyword in text for keyword in COMPARE_KEYWORDS):
        return build_rule_result(
            INTENT_COMPARE_SCHOOLS,
            "命中对比或选择类关键词",
        )

    if any(keyword in text for keyword in DETAIL_KEYWORDS) and mentions_school(text):
        return build_rule_result(
            INTENT_SCHOOL_DETAIL,
            "命中学校详情类关键词，并提到了具体学校",
        )

    if any(keyword in text for keyword in STRONG_FILTER_KEYWORDS) or "IELTS" in upper_text:
        return build_rule_result(
            INTENT_FILTER_REQUIREMENT,
            "命中强筛选条件类关键词",
        )

    if any(keyword in text for keyword in LIST_KEYWORDS):
        return build_rule_result(
            INTENT_LIST_SCHOOLS,
            "命中学校列表或推荐类关键词",
        )

    if any(keyword in text for keyword in FILTER_KEYWORDS) or "IELTS" in upper_text:
        return build_rule_result(
            INTENT_FILTER_REQUIREMENT,
            "命中筛选条件类关键词",
        )

    return None


def classify_intent_by_llm(question):
    chain = INTENT_CLASSIFIER_PROMPT | get_chat_model(temperature=0) | StrOutputParser()

    raw = chain.invoke(
        {
            "question": question,
            "rule_result": "规则未命中",
        }
    ).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "intent": INTENT_GENERAL_QA,
            "method": "llm_parse_failed",
            "confidence": 0.0,
            "reason": f"LLM 返回的不是合法 JSON: {raw[:200]}",
        }

    intent = result.get("intent", INTENT_GENERAL_QA)

    if intent not in VALID_INTENTS:
        intent = INTENT_GENERAL_QA

    try:
        confidence = float(result.get("confidence", 0.70))
    except (TypeError, ValueError):
        confidence = 0.70

    confidence = max(0.0, min(confidence, 0.95))

    return {
        "intent": intent,
        "method": "llm_fallback",
        "confidence": confidence,
        "reason": result.get("reason", ""),
    }


def classify_intent(question):
    rule_result = classify_intent_by_rules(question)

    if rule_result is not None:
        return rule_result

    return classify_intent_by_llm(question)