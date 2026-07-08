from django.contrib import admin
from django.urls import path

from rag.views import rag_query_api

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/rag/rag-query/", rag_query_api),
]