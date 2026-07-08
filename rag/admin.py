from django.contrib import admin

from rag.models import (
    ConversationSession,
    Document,
    DocumentChunk,
    IngestionJob,
    Message,
    QueryLog,
    RetrievedChunkLog,
)


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "file_name", "file_type", "status", "created_at")
    search_fields = ("title", "file_name", "source_path")
    list_filter = ("status", "file_type")


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "chunk_index", "page_number", "source_ref")
    search_fields = ("content", "source_ref")


@admin.register(IngestionJob)
class IngestionJobAdmin(admin.ModelAdmin):
    list_display = ("id", "source_path", "status", "document_count", "chunk_count", "started_at")
    list_filter = ("status",)


@admin.register(ConversationSession)
class ConversationSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "session_id", "title", "created_at", "updated_at")
    search_fields = ("session_id", "title")


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "role", "created_at")
    search_fields = ("content",)
    list_filter = ("role",)


@admin.register(QueryLog)
class QueryLogAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "question_preview",
        "rewritten_query_preview",
        "answer_preview",
        "intent",
        "model_name",
        "created_at",
    )
    search_fields = ("question", "answer", "rewritten_query")
    list_filter = ("intent", "model_name")
    readonly_fields = (
        "question",
        "rewritten_query",
        "answer",
        "retrieval_params",
        "metadata",
        "created_at",
    )

    def question_preview(self, obj):
        return obj.question[:50]

    question_preview.short_description = "Question"

    def rewritten_query_preview(self, obj):
        return obj.rewritten_query[:50]

    rewritten_query_preview.short_description = "Rewritten query"

    def answer_preview(self, obj):
        return obj.answer[:100]

    answer_preview.short_description = "Answer"


@admin.register(RetrievedChunkLog)
class RetrievedChunkLogAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "query",
        "chunk",
        "rank",
        "bm25_score",
        "vector_score",
        "rrf_score",
        "rerank_score",
        "used_in_answer",
    )
    list_filter = ("used_in_answer",)
    search_fields = ("snapshot_content", "snapshot_source_ref")