from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mlops.gateway import GatewayError, achat

router = APIRouter(tags=["data analytics"])

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_THINK_OPEN = re.compile(r"<think>.*", re.S | re.I)
_THINK_TAG = re.compile(r"</?think>", re.I)


class ChainSummaryIn(BaseModel):
    text: str = ""
    query: str = ""
    theme: str = ""


def _clean_llm(text: str) -> str:
    cleaned = _THINK_BLOCK.sub("", text or "")
    cleaned = _THINK_OPEN.sub("", cleaned)
    cleaned = _THINK_TAG.sub("", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


@router.post("/information-graph/chain-summary")
async def chain_summary(body: ChainSummaryIn):
    text = (body.text or "").strip()[:1200]
    query = (body.query or "").strip()
    theme = (body.theme or "").strip()
    if not text:
        return {"summary": ""}

    user = (
        "/no_think\n"
        f"Тема набора данных: {theme or 'не указана'}\n"
        f"Поисковый объект из поля поиска: {query or 'не указан'}\n\n"
        f"Текст первого сообщения цепочки:\n{text}\n\n"
        "Сформулируй краткое содержание исходного сообщения (2–3 предложения) "
        "применительно к теме и поисковому объекту. "
        "Каждое вхождение поискового объекта в ответе оберни в двойные скобки, "
        "например [[Платон]]. Без преамбулы, без пунктов, без рассуждений, только содержание."
    )
    try:
        result = await achat(
            provider="vllm",
            messages=[
                {
                    "role": "system",
                    "content": "/no_think\nТы аналитик русскоязычных соцмедиа. Отвечай сразу готовым текстом, без размышлений.",
                },
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=140,
            timeout=25,
            extra={"chat_template_kwargs": {"enable_thinking": False}},
            profile="information_spread_summary",
        )
    except GatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"summary": _clean_llm(result.content or "")}
