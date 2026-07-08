from datetime import timedelta

from django.conf import settings
from django.utils import timezone


DEFAULT_TASK_MEMORY = {
    "domain": "international_education",
    "country": "",
    "degree": "",
    "school": "",
    "major": "",
    "city": "",
    "focus": [],
    "user_goal": "",
    "constraints": [],
    "updated_from": "",
    "confidence": 0.0,
    "updated_at": "",
    "expires_at": "",
}


CARRY_OVER_FIELDS = {
    "domain",
    "country",
    "degree",
    "major",
    "focus",
    "user_goal",
    "constraints",
    "confidence",
}


STRICT_ENTITY_FIELDS = {
    "school",
    "city",
}


FOLLOWUP_KEYWORDS = [
    "继续",
    "还有吗",
    "下一页",
    "更多",
    "这个学校",
    "这所学校",
    "它",
    "刚才",
    "上面",
]


def is_followup_question(question):
    return any(keyword in question for keyword in FOLLOWUP_KEYWORDS)


def is_memory_expired(task_memory):
    expires_at = task_memory.get("expires_at")
    if not expires_at:
        return False

    try:
        expires_at_dt = timezone.datetime.fromisoformat(expires_at)
    except ValueError:
        return True

    if timezone.is_naive(expires_at_dt):
        expires_at_dt = timezone.make_aware(expires_at_dt)

    return timezone.now() > expires_at_dt


def get_empty_task_memory():
    return DEFAULT_TASK_MEMORY.copy()


def get_task_memory(session):
    if session is None:
        return get_empty_task_memory()

    metadata = session.metadata or {}
    task_memory = metadata.get("task_memory") or {}

    memory = get_empty_task_memory()
    memory.update(task_memory)

    if is_memory_expired(memory):
        return get_empty_task_memory()

    return memory


def save_task_memory(session, task_memory):
    if session is None:
        return None

    now = timezone.now()
    ttl_hours = getattr(settings, "TASK_MEMORY_TTL_HOURS", 6)
    expires_at = now + timedelta(hours=ttl_hours)

    task_memory = task_memory.copy()
    task_memory["updated_at"] = now.isoformat()
    task_memory["expires_at"] = expires_at.isoformat()

    metadata = session.metadata or {}
    metadata["task_memory"] = task_memory
    session.metadata = metadata
    session.save(update_fields=["metadata", "updated_at"])
    return session


def merge_task_memory(old_memory, new_memory, question=""):
    merged = old_memory.copy()
    followup = is_followup_question(question)

    for key in CARRY_OVER_FIELDS:
        value = new_memory.get(key)
        if value not in ("", None, [], {}):
            merged[key] = value

    for key in STRICT_ENTITY_FIELDS:
        value = new_memory.get(key)

        if value not in ("", None, [], {}):
            merged[key] = value
        elif not followup:
            merged[key] = ""

    return merged