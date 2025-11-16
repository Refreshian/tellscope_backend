import redis
from datetime import datetime

redis_client = redis.Redis(host="localhost", port=6379, db=0)

def safe_update_progress(task_id, new_progress, status="processing", error=None, stage=None, stage_details=None):
    new_progress = max(0, min(100, int(new_progress)))
    current_progress = 0
    try:
        current_progress = int(redis_client.hget(f"task:{task_id}", "progress") or 0)
    except Exception:
        current_progress = 0
    new_progress = max(current_progress, new_progress)
    mapping = {
        "progress": str(new_progress),
        "status": status,
        "last_update": datetime.now().isoformat()
    }
    if error:
        mapping["error"] = error
    if stage:
        mapping["stage"] = stage
    if stage_details is not None:
        mapping["stage_details"] = stage_details
    if status in ["completed", "failed"]:
        mapping[f"{status}_at"] = datetime.now().isoformat()
    redis_client.hset(f"task:{task_id}", mapping=mapping)
    redis_client.expire(f"task:{task_id}", 3600)