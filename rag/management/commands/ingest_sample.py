from django.core.management.base import BaseCommand
from django.utils import timezone

from rag.models import Document, DocumentChunk, IngestionJob
from rag.services.loaders import load_markdown_file
from rag.services.splitter import split_school_markdown


class Command(BaseCommand):
    help = "Ingest OpenApply sample markdown into DocumentChunk."

    def add_arguments(self, parser):
        parser.add_argument("file_path", type=str)

    def handle(self, *args, **options):
        file_path = options["file_path"]

        job = IngestionJob.objects.create(source_path=file_path)

        try:
            loaded = load_markdown_file(file_path)
            chunks = split_school_markdown(
                text=loaded["text"],
                source_file=loaded["file_name"],
            )

            document, _ = Document.objects.update_or_create(
                file_hash=loaded["file_hash"],
                defaults={
                    "title": loaded["title"],
                    "source_path": loaded["source_path"],
                    "file_name": loaded["file_name"],
                    "file_type": loaded["file_type"],
                    "status": Document.STATUS_PROCESSING,
                    "metadata": {
                        "loader": "markdown_school_sample",
                    },
                },
            )

            document.chunks.all().delete()

            for index, chunk in enumerate(chunks, start=1):
                DocumentChunk.objects.create(
                    document=document,
                    chunk_index=index,
                    content=chunk["content"],
                    token_count=0,
                    char_count=len(chunk["content"]),
                    section_title=chunk["section_title"],
                    source_ref=chunk["source_ref"],
                    metadata={
                        **chunk["metadata"],
                        "raw_content": chunk["raw_content"],
                    },
                )

            document.status = Document.STATUS_INDEXED
            document.save(update_fields=["status", "updated_at"])

            job.status = IngestionJob.STATUS_SUCCESS
            job.document_count = 1
            job.chunk_count = len(chunks)
            job.finished_at = timezone.now()
            job.save()

            self.stdout.write(
                self.style.SUCCESS(
                    f"Ingest success. document_id={document.id}, chunks={len(chunks)}"
                )
            )

        except Exception as exc:
            job.status = IngestionJob.STATUS_FAILED
            job.error_message = str(exc)
            job.finished_at = timezone.now()
            job.save()

            raise