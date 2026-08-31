from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from .catalog import Catalog, CONFIG, ROOT
from .classify import classify_messages, rewrite_insights_aitunnel
from .excel_export import export_excel
from .ingest import load_folder
from .metrics import build_report
from .pptx_export import build_pptx

try:
    from mlops.lineage import build_run, cache_fingerprint, file_hash, write_run
    from mlops.lock import generate_cfg
except ImportError:
    build_run = None
    cache_fingerprint = None
    file_hash = None
    write_run = None
    generate_cfg = None


def load_settings(path: Path | None = None) -> dict:
    p = path or (CONFIG / "settings.yaml")
    if p.exists():
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {}


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    period: str = "",
    tellscope: bool = False,
    progress=None,
    job_id: str = "",
) -> dict:
    settings = load_settings()
    use_vllm = tellscope or os.environ.get("MOSINFORM_VLLM", "0") == "1"
    if use_vllm:
        settings.setdefault("tellscope", {})
        settings["tellscope"]["enabled"] = True
        model = (generate_cfg()["model"] if generate_cfg else os.environ.get("VLLM_MODEL", "Qwen/Qwen3-32B-FP8"))
        base = (generate_cfg()["base_url"] if generate_cfg else os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000"))
        settings["tellscope"].setdefault("base_url", base)
        settings["tellscope"].setdefault("model", model)
        settings["tellscope"].setdefault("prompt_id", "classify_v1")
    if period:
        settings["period_label"] = period
    catalog = Catalog()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"

    if progress:
        progress("чтение выгрузок")
    messages, meta = load_folder(Path(input_dir), catalog)
    if progress:
        progress(f"разобрано {len(messages)} текстов, разметка")
    messages = classify_messages(messages, catalog, settings, cache_dir=cache_dir, progress=progress)
    bundle = build_report(messages, catalog, meta, period_label=period or settings.get("period_label") or "")
    if progress:
        progress("формулировки слайдов")
    overrides = rewrite_insights_aitunnel(bundle, catalog, settings)
    if overrides:
        settings["insight_overrides"] = overrides

    xlsx_path = output_dir / "mosinform_rating.xlsx"
    pptx_path = output_dir / "mosinform_rating.pptx"
    if progress:
        progress("сборка Excel и PPTX")
    export_excel(bundle, catalog, xlsx_path)
    build_pptx(bundle, catalog, pptx_path, settings)
    result = {
        "messages": len(bundle.messages),
        "objects": len(bundle.object_stats),
        "untagged": sum(1 for m in bundle.messages if not m.object_ids),
        "xlsx": str(xlsx_path),
        "pptx": str(pptx_path),
        "notes": bundle.notes,
        "missing": bundle.missing_metrics,
        "top": [(catalog.label(s.object_id), s.messages) for s in bundle.object_stats[:10]],
    }
    if build_run:
        ts_cfg = settings.get("tellscope") or {}
        prompt_id = ts_cfg.get("prompt_id") or "classify_v1"
        catalog_hash = file_hash(CONFIG / "objects.yaml") if file_hash else ""
        cache_key = ts_cfg.get("_cache_key") or (
            cache_fingerprint(ts_cfg.get("model") or "", prompt_id, catalog_hash) if cache_fingerprint else ""
        )
        lineage = build_run(
            product="mosinform",
            route="/mosinform-rating",
            job_id=job_id,
            prompt_id=prompt_id,
            catalog_version=str(settings.get("catalog_version") or "2026.08"),
            catalog_hash=catalog_hash,
            cache_key=cache_key,
            extra={
                "aitunnel_model": ((settings.get("aitunnel") or {}).get("model") or os.environ.get("AITUNNEL_MODEL") or ""),
                "aitunnel_prompt_id": ((settings.get("aitunnel") or {}).get("prompt_id") or "insights_v1"),
                "cache_legacy": bool(ts_cfg.get("_cache_legacy")),
                "period": period or settings.get("period_label") or "",
                "status": "done",
            },
        )
        write_run(output_dir / "run.json", lineage)
        result["lineage"] = lineage
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
