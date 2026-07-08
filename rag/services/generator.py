import json

from django.conf import settings
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from rag.services.prompts import ANSWER_PROMPT, JUDGE_PROMPT, REWRITE_PROMPT,TASK_MEMORY_PROMPT
from rag.services.prompts import (
    COMPARE_SCHOOLS_PROMPT,
    FILTER_REQUIREMENT_PROMPT,
    GENERAL_QA_PROMPT,
    LIST_SCHOOLS_PROMPT,
    SCHOOL_DETAIL_PROMPT,
)

def get_answer_prompt(intent):
    if intent == "list_schools":
        return LIST_SCHOOLS_PROMPT

    if intent == "school_detail":
        return SCHOOL_DETAIL_PROMPT

    if intent == "compare_schools":
        return COMPARE_SCHOOLS_PROMPT

    if intent == "filter_requirement":
        return FILTER_REQUIREMENT_PROMPT

    if intent == "general_qa":
        return GENERAL_QA_PROMPT

    return ANSWER_PROMPT

def get_chat_model(temperature=0.2):
    """统一创建 LangChain ChatOpenAI 客户端，兼容 OpenAI 风格的第三方 base_url。"""
    kwargs = {
        "model": settings.OPENAI_MODEL,
        "api_key": settings.OPENAI_API_KEY,
        "temperature": temperature,
    }

    if getattr(settings, "OPENAI_BASE_URL", None):
        kwargs["base_url"] = settings.OPENAI_BASE_URL

    return ChatOpenAI(**kwargs)


def rewrite_query(question):
    """把用户原始问题改写成更适合检索的 query。"""
    chain = REWRITE_PROMPT | get_chat_model(temperature=0) | StrOutputParser()
    return chain.invoke({"question": question}).strip()


def generate_answer(
    question,
    contexts_text,
    intent="general_qa",
    filters=None,
    display_top_k=5,
    remaining_count=0,
):
    prompt = get_answer_prompt(intent)
    chain = prompt | get_chat_model(temperature=0.2) | StrOutputParser()

    return chain.invoke(
        {
            "question": question,
            "contexts": contexts_text,
            "filters": filters or {},
            "display_top_k": display_top_k,
            "remaining_count": remaining_count,
        }
    ).strip()


def judge_answer(question, contexts_text, answer):
    """用 LLM 检查回答是否忠实于资料、是否带引用、是否存在编造。"""
    chain = JUDGE_PROMPT | get_chat_model(temperature=0) | StrOutputParser()
    raw = chain.invoke(
        {
            "question": question,
            "contexts": contexts_text,
            "answer": answer,
        }
    ).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "pass": False,
            "reason": f"Judge returned invalid JSON: {raw}",
        }


def extract_task_memory(question):
    chain = TASK_MEMORY_PROMPT | get_chat_model(temperature=0) | StrOutputParser()
    raw = chain.invoke({"question": question}).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}