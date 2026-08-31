"""Read-only MLOps surface: serving lock, jobs, prompts, drain, Prometheus metrics."""
from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Response

from mlops.gateway import gateway_metrics
from mlops.jobs import list_jobs
from mlops.lock import public_lock
from mlops.prompts import list_prompts
from mlops.runtime import reap_stale_jobs, running_batches, stale_jobs

router = APIRouter(prefix="/mlops", tags=["mlops"])


@router.get("/lock")
def serving_lock():
    return public_lock()


@router.get("/jobs")
def jobs():
    return {"jobs": list_jobs()}


@router.get("/busy")
def busy():
    return {"jobs": running_batches()}


@router.get("/ready")
def ready(response: Response):
    """503 if a live GPU batch holds Qwen — check before FastAPI restart."""
    holders = running_batches()
    if holders:
        response.status_code = 503
        return {"ready": False, "jobs": holders}
    return {"ready": True, "jobs": []}


@router.get("/prompts")
def prompts():
    return {"prompts": list_prompts()}


@router.post("/drain")
def drain():
    """Mark stale journal leftovers as stale. Live GPU holders are left alone."""
    return {"reaped": reap_stale_jobs(), "stale": stale_jobs()}


def _prom_line(name: str, labels: dict, value) -> str:
    parts = ",".join(f'{k}="{v}"' for k, v in labels.items())
    inner = f"{{{parts}}}" if parts else ""
    return f"{name}{inner} {value}"


@router.get("/metrics")
def metrics():
    jobs = list_jobs()
    busy_jobs = running_batches()
    by_ps = Counter((job.get("product") or "unknown", job.get("status") or "unknown") for job in jobs)
    stale = len(stale_jobs())
    gw = gateway_metrics()
    lines = [
        "# HELP tellscope_jobs Jobs in the unified journal",
        "# TYPE tellscope_jobs gauge",
    ]
    for (product, status), count in sorted(by_ps.items()):
        lines.append(_prom_line("tellscope_jobs", {"product": product, "status": status}, count))
    lines += [
        "# HELP tellscope_gpu_busy 1 if a live heavy GPU job holds Qwen",
        "# TYPE tellscope_gpu_busy gauge",
        f"tellscope_gpu_busy {1 if busy_jobs else 0}",
        "# HELP tellscope_gpu_holders Live heavy GPU jobs holding Qwen",
        "# TYPE tellscope_gpu_holders gauge",
        f"tellscope_gpu_holders {len(busy_jobs)}",
        "# HELP tellscope_jobs_stale Active-looking jobs without a fresh heartbeat",
        "# TYPE tellscope_jobs_stale gauge",
        f"tellscope_jobs_stale {stale}",
        "# HELP tellscope_gateway_calls_total LLM gateway calls since process start",
        "# TYPE tellscope_gateway_calls_total counter",
    ]
    for key, count in sorted(gw["calls"].items()):
        provider, profile = (key.split("|", 1) + ["-"])[:2]
        lines.append(_prom_line("tellscope_gateway_calls_total", {"provider": provider, "profile": profile}, count))
    lines += [
        "# HELP tellscope_gateway_errors_total LLM gateway errors since process start",
        "# TYPE tellscope_gateway_errors_total counter",
    ]
    for key, count in sorted(gw["errors"].items()):
        provider, profile = (key.split("|", 1) + ["-"])[:2]
        lines.append(_prom_line("tellscope_gateway_errors_total", {"provider": provider, "profile": profile}, count))
    lines += [
        "# HELP tellscope_gateway_latency_ms_sum Sum of gateway call latency",
        "# TYPE tellscope_gateway_latency_ms_sum counter",
    ]
    for key, total in sorted(gw["latency_ms"].items()):
        provider, profile = (key.split("|", 1) + ["-"])[:2]
        lines.append(_prom_line("tellscope_gateway_latency_ms_sum", {"provider": provider, "profile": profile}, round(total, 1)))
    body = "\n".join(lines) + "\n"
    return Response(content=body, media_type="text/plain; version=0.0.4")
