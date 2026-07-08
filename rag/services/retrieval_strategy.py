from rag.services.intent_classifier import (
    INTENT_COMPARE_SCHOOLS,
    INTENT_CONTINUE_LIST,
    INTENT_FILTER_REQUIREMENT,
    INTENT_LIST_SCHOOLS,
    INTENT_SCHOOL_DETAIL,
)


def build_retrieval_strategy(intent, filters=None):
    filters = filters or {}

    if intent == INTENT_LIST_SCHOOLS:
        return {
            "retrieval_top_k": 40,
            "context_top_k": 20,
            "display_top_k": 5,
            "enable_pagination": True,
            "use_previous_results": False,
        }

    if intent == INTENT_CONTINUE_LIST:
        return {
            "retrieval_top_k": 0,
            "context_top_k": 5,
            "display_top_k": 5,
            "enable_pagination": True,
            "use_previous_results": True,
        }

    if intent == INTENT_COMPARE_SCHOOLS:
        return {
            "retrieval_top_k": 20,
            "context_top_k": 8,
            "display_top_k": 8,
            "enable_pagination": False,
            "use_previous_results": False,
        }

    if intent == INTENT_SCHOOL_DETAIL:
        return {
            "retrieval_top_k": 12,
            "context_top_k": 5,
            "display_top_k": 5,
            "enable_pagination": False,
            "use_previous_results": False,
        }

    if intent == INTENT_FILTER_REQUIREMENT:
        return {
            "retrieval_top_k": 30,
            "context_top_k": 12,
            "display_top_k": 5,
            "enable_pagination": True,
            "use_previous_results": False,
        }

    return {
        "retrieval_top_k": 20,
        "context_top_k": 8,
        "display_top_k": 5,
        "enable_pagination": False,
        "use_previous_results": False,
    }