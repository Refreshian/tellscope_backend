from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx

from .catalog import Catalog
from .models import Message

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
    cache: dict[str, dict] = {}
    if cache_dir:
        cache_path = Path(cache_dir) / "tellscope.jsonl"
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    cache[rec["id"]] = rec

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
        results = _vllm_classify(pending, catalog, ts_cfg, progress=progress)
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with (Path(cache_dir) / "tellscope.jsonl").open("a", encoding="utf-8") as fh:
                for rec in results:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        by_id = {r["id"]: r for r in results}
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
    base = (cfg.get("base_url") or os.environ.get("VLLM_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
    model = cfg.get("model") or os.environ.get("VLLM_MODEL") or "Qwen/Qwen3-32B-FP8"
    batch_size = int(cfg.get("batch_size") or 4)
    timeout = float(cfg.get("timeout_sec") or 180)
    allowed = ", ".join(f"{o.id} ({o.short})" for o in catalog.objects)
    system = (
        "Ты аналитик медиаприсутствия органов власти Москвы. "
        "/no_think\n"
        "Верни ТОЛЬКО JSON вида {\"items\":[...]} без рассуждений и без markdown. "
        f"object_ids только из списка: {allowed}. "
        "Поля item: id, object_ids (массив), sentiment (positive|neutral|negative), "
        "role (main|episodic|background), initiated (true если похоже на релиз/пресс-службу)."
    )
    out: list[dict] = []
    with httpx.Client(timeout=timeout) as client:
        for i in range(0, len(messages), batch_size):
            chunk = messages[i : i + batch_size]
            payload_items = [
                {"id": m.id, "title": m.title, "source": m.source, "text": (m.text or "")[:1800]}
                for m in chunk
            ]
            body = {
                "model": model,
                "temperature": 0.1,
                "max_tokens": 1200,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload_items, ensure_ascii=False)},
                ],
                "chat_template_kwargs": {"enable_thinking": False},
            }
            try:
                resp = client.post(f"{base}/v1/chat/completions", json=body)
                resp.raise_for_status()
                content = (((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
                parsed = _parse_json(content)
                items = parsed.get("items") if isinstance(parsed, dict) else parsed
                if isinstance(items, list):
                    out.extend(items)
            except Exception as exc:
                if progress:
                    progress(f"партия {i // batch_size + 1}: {exc}")
            if progress and i and i % 40 == 0:
                progress(f"размечено {min(i + batch_size, len(messages))}/{len(messages)}")
    return out


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
    key = os.environ.get("AITUNNEL_API_KEY") or ((settings or {}).get("aitunnel") or {}).get("api_key")
    if not key:
        return {}
    base = (
        ((settings or {}).get("aitunnel") or {}).get("base_url")
        or os.environ.get("AITUNNEL_BASE_URL")
        or "https://api.aitunnel.ru/v1"
    ).rstrip("/")
    model = (
        ((settings or {}).get("aitunnel") or {}).get("model")
        or os.environ.get("AITUNNEL_MODEL")
        or "gpt-4.1-mini"
    )
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
    prompt = (
        "Ты готовишь слайды рейтинга медиаприсутствия ОИВ Москвы в деловом стиле пилота Мосинформ. "
        "По цифрам и заголовкам напиши JSON с ключами volume, speakers, media, tone, role, concentration, initiated, observations. "
        "Каждый ключ кроме observations — массив из ровно 4 коротких фраз на русском (1–2 предложения). "
        "observations — массив из 3 объектов {title, fact, meaning, full}. Без воды и эмодзи.\n\n"
        f"Период: {bundle.period_label}\nСтатистика: {json.dumps(top, ensure_ascii=False)}\n"
        f"Заголовки: {json.dumps(headlines, ensure_ascii=False)}"
    )
    try:
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
