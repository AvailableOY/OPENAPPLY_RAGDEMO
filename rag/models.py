from django.db import models
from django.utils import timezone


class Document(models.Model):
    """Original source document uploaded or placed in the data directory."""

    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_INDEXED = "indexed"
    STATUS_FAILED = "failed"

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_INDEXED, "Indexed"),
        (STATUS_FAILED, "Failed"),
    ]

    title = models.CharField(max_length=255)
    source_path = models.TextField()
    file_name = models.CharField(max_length=255)
    file_type = models.CharField(max_length=32, blank=True)
    file_hash = models.CharField(max_length=64, unique=True)
    status = models.CharField(
        max_length=32,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
    )
    metadata = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["file_hash"]),
            models.Index(fields=["status"]),
        ]

    def __str__(self):
        return self.title


class DocumentChunk(models.Model):
    """A searchable text chunk with traceable source metadata."""

    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="chunks",
    )
    chunk_index = models.PositiveIntegerField()
    content = models.TextField()
    token_count = models.PositiveIntegerField(default=0)
    char_count = models.PositiveIntegerField(default=0)
    page_number = models.PositiveIntegerField(null=True, blank=True)
    section_title = models.CharField(max_length=255, blank=True)
    source_ref = models.CharField(max_length=512, blank=True)
    vector_id = models.CharField(max_length=128, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["document_id", "chunk_index"]
        unique_together = [("document", "chunk_index")]
        indexes = [
            models.Index(fields=["document", "chunk_index"]),
            models.Index(fields=["vector_id"]),
        ]

    def __str__(self):
        return f"{self.document.title}#{self.chunk_index}"


class IngestionJob(models.Model):
    """A record of one document ingestion run."""

    STATUS_RUNNING = "running"
    STATUS_SUCCESS = "success"
    STATUS_FAILED = "failed"

    STATUS_CHOICES = [
        (STATUS_RUNNING, "Running"),
        (STATUS_SUCCESS, "Success"),
        (STATUS_FAILED, "Failed"),
    ]

    source_path = models.TextField()
    status = models.CharField(
        max_length=32,
        choices=STATUS_CHOICES,
        default=STATUS_RUNNING,
    )
    document_count = models.PositiveIntegerField(default=0)
    chunk_count = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True)
    started_at = models.DateTimeField(default=timezone.now)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self):
        return f"IngestionJob({self.status}) {self.source_path}"


class ConversationSession(models.Model):
    """Lightweight session for short-term multi-turn context."""

    session_id = models.CharField(max_length=64, unique=True)
    title = models.CharField(max_length=255, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return self.session_id


class Message(models.Model):
    """A user or assistant message in a conversation session."""

    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"

    ROLE_CHOICES = [
        (ROLE_USER, "User"),
        (ROLE_ASSISTANT, "Assistant"),
        (ROLE_SYSTEM, "System"),
    ]

    session = models.ForeignKey(
        ConversationSession,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    role = models.CharField(max_length=32, choices=ROLE_CHOICES)
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["session", "created_at"]),
        ]

    def __str__(self):
        return f"{self.session.session_id}:{self.role}"


class QueryLog(models.Model):
    """One user question and the generated grounded answer."""

    session = models.ForeignKey(
        ConversationSession,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="queries",
    )
    question = models.TextField()
    rewritten_query = models.TextField(blank=True)
    intent = models.CharField(max_length=64, blank=True)
    answer = models.TextField(blank=True)
    model_name = models.CharField(max_length=128, blank=True)
    retrieval_params = models.JSONField(default=dict, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["created_at"]),
            models.Index(fields=["intent"]),
        ]

    def __str__(self):
        return self.question[:80]


class RetrievedChunkLog(models.Model):
    """A retrieved chunk and all ranking scores for one query."""

    query = models.ForeignKey(
        QueryLog,
        on_delete=models.CASCADE,
        related_name="retrieved_chunks",
    )
    chunk = models.ForeignKey(
        DocumentChunk,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="retrieval_logs",
    )
    rank = models.PositiveIntegerField()
    bm25_score = models.FloatField(null=True, blank=True)
    vector_score = models.FloatField(null=True, blank=True)
    rrf_score = models.FloatField(null=True, blank=True)
    rerank_score = models.FloatField(null=True, blank=True)
    used_in_answer = models.BooleanField(default=False)
    snapshot_content = models.TextField(blank=True)
    snapshot_source_ref = models.CharField(max_length=512, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["query_id", "rank"]
        unique_together = [("query", "rank")]
        indexes = [
            models.Index(fields=["query", "rank"]),
            models.Index(fields=["used_in_answer"]),
        ]

    def __str__(self):
        return f"Query#{self.query_id} rank {self.rank}"
