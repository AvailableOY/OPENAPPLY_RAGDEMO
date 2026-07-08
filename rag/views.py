import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from rag.services.rag_chain import run_rag_query


@csrf_exempt
def rag_query_api(request):
    """Postman 调试入口：调用当前 fake RAG 流程并返回完整链路结果。"""
    if request.method != "POST":
        return JsonResponse(
            {"error": "Only POST is allowed."},
            status=405,
        )

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse(
            {"error": "Invalid JSON body."},
            status=400,
        )

    question = payload.get("question")
    session_id = payload.get("session_id", "postman-demo-session")

    if not question:
        return JsonResponse(
            {"error": "question is required."},
            status=400,
        )

    result = run_rag_query(
        question=question,
        session_id=session_id,
    )

    # ensure_ascii=False 保证 Postman 中直接显示中文。
    return JsonResponse(
        {
            "query_id": result["query_id"],
            "question": result["question"],
            "rewritten_query": result["rewritten_query"],
            "answer": result["answer"],
            "judge": result["judge"],
            "attempt": result["attempt"],
            "contexts": result["contexts"],
            "task_memory": result.get("task_memory", {}),
            "intent_result": result.get("intent_result", {}),
            "filters": result.get("filters", {}),
            "strategy": result.get("strategy", {}),
        },
        json_dumps_params={"ensure_ascii": False},
    )
