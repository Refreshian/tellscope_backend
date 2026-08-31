from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx

from .catalog import Catalog
from .models import Message

try:
    from mlops.gateway import GatewayError, chat as llm_chat
    from mlops.lineage import cache_fingerprint, file_hash
    from mlops.lock import generate_cfg, prompt_id as lock_prompt_id
    from mlops.prompts import render_prompt
except ImportError:  # local CLI without mlops on PYTHONPATH
    GatewayError = None
    llm_chat = None
    cache_fingerprint = None
    file_hash = None
    generate_cfg = None
    lock_prompt_id = None
    render_prompt = None

INITIATED_MARKERS = (
    "пресс-служб",
    "сообщили в департамент",
    "сообщил департамент",
    "сообщается на сайте",
    "официальный сайт",
    "в ходе рабочего",
    "по словам руководителя",
    "приводятся в сообщении слова",
    "сообщила пресс-служба",
    "рассказал руководитель",
)

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def classify_messages(
    messages: list[Message],
    catalog: Catalog,
    settings: dict,
    cache_dir: Path | None = None,
    progress=None,
) -> list[Message]:
    ts_cfg = (settings or {}).get("tellscope") or {}
    prompt_id = ts_cfg.get("prompt_id") or (lock_prompt_id("mosinform_classify", "classify_v1") if lock_prompt_id else "classify_v1")
    model = ts_cfg.get("model") or (generate_cfg()["model"] if generate_cfg else os.environ.get("VLLM_MODEL") or "Qwen/Qwen3-32B-FP8")
    catalog_hash = ""
    if file_hash:
        from .catalog import CONFIG

        catalog_hash = file_hash(CONFIG / "objects.yaml")
    fp = cache_fingerprint(model, prompt_id, catalog_hash) if cache_fingerprint else "legacy"
    cache: dict[str, dict] = {}
    versioned_path = None
    if cache_dir:
        versioned_path = Path(cache_dir) / f"classify_{fp}.jsonl"
        legacy_path = Path(cache_dir) / "tellscope.jsonl"
        chosen = versioned_path if versioned_path.exists() else legacy_path
        if chosen.exists():
            for line in chosen.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("id"):
                        cache[rec["id"]] = rec
        ts_cfg["_cache_key"] = fp
        ts_cfg["_cache_legacy"] = chosen == legacy_path and not versioned_path.exists()

    pending: list[Message] = []
    for msg in messages:
        _heuristic(msg, catalog)
        cached = cache.get(msg.id)
        if cached:
            _apply_model(msg, cached)
            continue
        pending.append(msg)

    if ts_cfg.get("enabled") and pending:
        if progress:
            progress(f"LLM-разметка {len(pending)} текстов")
        ts_cfg["prompt_id"] = prompt_id
        results = _vllm_classify(pending, catalog, ts_cfg, progress=progress)
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            out_path = versioned_path or (Path(cache_dir) / "tellscope.jsonl")
            with out_path.open("a", encoding="utf-8") as fh:
                for rec in results:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        by_id = {r["id"]: r for r in results if r.get("id")}
        for msg in pending:
            if msg.id in by_id:
                _apply_model(msg, by_id[msg.id])
    return messages


def _heuristic(msg: Message, catalog: Catalog) -> None:
    blob = f"{msg.title}\n{msg.text}"
    inherited = list(msg.object_ids)
    title_hits = catalog.match_objects(msg.title)
    body_hits = catalog.match_objects(blob)
    msg.object_ids = list(dict.fromkeys([*inherited, *title_hits, *body_hits]))
    msg.speaker_ids = catalog.match_persons(blob)
    for pid in msg.speaker_ids:
        person = catalog.person_by_id.get(pid)
        if person and person.object_id and person.object_id not in msg.object_ids:
            msg.object_ids.append(person.object_id)
    for oid in msg.object_ids:
        obj = catalog.by_id.get(oid)
        if not obj:
            continue
        in_title = any(a.lower() in msg.title.lower() for a in [obj.short, obj.name, *obj.aliases] if a)
        if in_title:
            msg.role_by_object[oid] = "main"
        elif blob.lower().count((obj.short or "").lower()) <= 1 and not in_title:
            msg.role_by_object[oid] = "episodic"
        else:
            msg.role_by_object[oid] = "background"
    low = blob.lower()
    msg.initiated = any(m in low for m in INITIATED_MARKERS)
    msg.sentiment = msg.sentiment or "neutral"
    msg.classified_by = "heuristic"


def _apply_model(msg: Message, rec: dict) -> None:
    if rec.get("object_ids"):
        msg.object_ids = list(dict.fromkeys([*msg.object_ids, *rec["object_ids"]]))
    if rec.get("speaker_ids"):
        msg.speaker_ids = list(dict.fromkeys([*msg.speaker_ids, *rec["speaker_ids"]]))
    if rec.get("sentiment") in {"positive", "neutral", "negative"}:
        msg.sentiment = rec["sentiment"]
    if rec.get("role_by_object"):
        msg.role_by_object.update(rec["role_by_object"])
    elif rec.get("role") and msg.object_ids:
        for oid in msg.object_ids:
            msg.role_by_object.setdefault(oid, rec["role"])
    if rec.get("initiated") is not None:
        msg.initiated = bool(rec["initiated"])
    if rec.get("topics"):
        msg.topics = rec["topics"]
    msg.classified_by = "vllm"


def _vllm_classify(messages: list[Message], catalog: Catalog, cfg: dict, progress=None) -> list[dict]:
    model = cfg.get("model") or os.environ.get("VLLM_MODEL") or "Qwen/Qwen3-32B-FP8"
    batch_size = max(1, min(int(cfg.get("batch_size") or 4), 4))
    timeout = float(cfg.get("timeout_sec") or 180)
    text_limit = int(cfg.get("text_limit") or 1000)
    max_tokens = int(cfg.get("max_tokens") or 400)
    allowed = ", ".join(f"{o.id} ({o.short})" for o in catalog.objects)
    prompt_id = cfg.get("prompt_id") or "classify_v1"
    if render_prompt:
        system = render_prompt(prompt_id, allowed=allowed)
    else:
        system = (
            "Ты аналитик медиаприсутствия органов власти Москвы. "
            "/no_think\n"
            "Верни ТОЛЬКО JSON вида {\"items\":[...]} без рассуждений и без markdown. "
            f"object_ids только из списка: {allowed}. "
            "Поля item: id, object_ids (массив), sentiment (positive|neutral|negative), "
            "role (main|episodic|background), initiated (true если похоже на релиз/пресс-службу)."
        )
    out: list[dict] = []
    total = len(messages)
    for i in range(0, total, batch_size):
        chunk = messages[i : i + batch_size]
        items = _classify_chunk(model, system, chunk, text_limit, max_tokens, timeout)
        out.extend(items)
        done = min(i + batch_size, total)
        if progress and (done == total or done % 20 == 0 or i == 0):
            progress(f"размечено {done}/{total}")
    return out


def _classify_chunk(
    model: str,
    system: str,
    chunk: list[Message],
    text_limit: int,
    max_tokens: int,
    timeout: float,
) -> list[dict]:
    if not chunk:
        return []
    items = _post_classify(model, system, chunk, text_limit, max_tokens, timeout)
    if items is not None:
        return items
    if len(chunk) > 1:
        mid = len(chunk) // 2
        return _classify_chunk(model, system, chunk[:mid], text_limit, max_tokens, timeout) + _classify_chunk(
            model, system, chunk[mid:], text_limit, max_tokens, timeout
        )
    if text_limit > 400:
        return _classify_chunk(model, system, chunk, 400, min(max_tokens, 192), timeout)
    return []


def _post_classify(
    model: str,
    system: str,
    chunk: list[Message],
    text_limit: int,
    max_tokens: int,
    timeout: float,
) -> list[dict] | None:
    payload_items = [
        {"id": m.id, "title": m.title, "source": m.source, "text": (m.text or "")[:text_limit]}
        for m in chunk
    ]
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    try:
        if llm_chat:
            result = llm_chat(
                provider="vllm",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload_items, ensure_ascii=False)},
                ],
                model=model,
                temperature=0.1,
                max_tokens=max_tokens,
                timeout=timeout,
                extra=extra,
            )
            content = result.content
        else:
            base = (os.environ.get("VLLM_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
            resp = httpx.Client(timeout=timeout).post(
                f"{base}/v1/chat/completions",
                json={
                    "model": model,
                    "temperature": 0.1,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(payload_items, ensure_ascii=False)},
                    ],
                    **extra,
                },
            )
            if resp.status_code == 400:
                return None
            resp.raise_for_status()
            content = (((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
        parsed = _parse_json(content)
        items = parsed.get("items") if isinstance(parsed, dict) else parsed
        return items if isinstance(items, list) else []
    except Exception as exc:
        if GatewayError and isinstance(exc, GatewayError) and exc.status_code == 400:
            return None
        if isinstance(exc, httpx.HTTPError):
            return None
        if GatewayError and isinstance(exc, GatewayError):
            return None
        return None


def _parse_json(text: str):
    text = (text or "").replace("```json", "```")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "```" in text:
        text = text.split("```", 2)[1]
    match = JSON_RE.search(text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def rewrite_insights_aitunnel(bundle, catalog, settings: dict) -> dict:
    cfg = (settings or {}).get("aitunnel") or {}
    prompt_id = cfg.get("prompt_id") or "insights_v1"
    top = [
        {
            "name": catalog.label(s.object_id),
            "messages": s.messages,
            "main_role": s.main_role,
            "top2": round(s.top2_share, 2),
            "media": dict(list(s.media.items())[:5]),
        }
        for s in bundle.object_stats[:8]
    ]
    headlines = {}
    for st in bundle.object_stats[:5]:
        titles = [m.title for m in bundle.messages if st.object_id in m.object_ids and m.title][:6]
        headlines[catalog.label(st.object_id)] = titles
    if render_prompt:
        prompt = render_prompt(
            prompt_id,
            period=bundle.period_label,
            stats=json.dumps(top, ensure_ascii=False),
            headlines=json.dumps(headlines, ensure_ascii=False),
        )
    else:
        prompt = (
            "Ты готовишь слайды рейтинга медиаприсутствия ОИВ Москвы в деловом стиле пилота Мосинформ. "
            "По цифрам и заголовкам напиши JSON с ключами volume, speakers, media, tone, role, concentration, initiated, observations. "
            "Каждый ключ кроме observations — массив из ровно 4 коротких фраз на русском (1–2 предложения). "
            "observations — массив из 3 объектов {title, fact, meaning, full}. Без воды и эмодзи.\n\n"
            f"Период: {bundle.period_label}\nСтатистика: {json.dumps(top, ensure_ascii=False)}\n"
            f"Заголовки: {json.dumps(headlines, ensure_ascii=False)}"
        )
    try:
        if llm_chat:
            result = llm_chat(
                provider="aitunnel",
                profile="mosinform_insights",
                messages=[
                    {"role": "system", "content": "Отвечай только валидным JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2500,
                timeout=90,
            )
            data = _parse_json(result.content)
            return data if isinstance(data, dict) else {}
        key = os.environ.get("AITUNNEL_API_KEY") or cfg.get("api_key")
        if not key:
            return {}
        base = (cfg.get("base_url") or os.environ.get("AITUNNEL_BASE_URL") or "https://api.aitunnel.ru/v1").rstrip("/")
        model = cfg.get("model") or os.environ.get("AITUNNEL_MODEL") or "gpt-4.1-mini"
        with httpx.Client(timeout=90) as client:
            resp = client.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "temperature": 0.3,
                    "max_tokens": 2500,
                    "messages": [
                        {"role": "system", "content": "Отвечай только валидным JSON."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()
            content = (((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
            data = _parse_json(content)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
