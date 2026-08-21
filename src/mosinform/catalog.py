from __future__ import annotations

import re
from pathlib import Path

import yaml

from .models import CatalogObject, Person

_PKG = Path(__file__).resolve().parent
_CANDIDATES = [
    _PKG / "config",
    _PKG.parents[1] / "config",
    Path("/home/dev/tellscope_app/tellscope_backend/mosinform/config"),
]
CONFIG = next((p for p in _CANDIDATES if (p / "objects.yaml").exists()), _CANDIDATES[0])
ROOT = CONFIG.parent


class Catalog:
    def __init__(self, objects_path: Path | None = None, media_path: Path | None = None):
        objects_path = objects_path or CONFIG / "objects.yaml"
        media_path = media_path or CONFIG / "media.yaml"
        raw = yaml.safe_load(objects_path.read_text(encoding="utf-8"))
        self.objects: list[CatalogObject] = [
            CatalogObject(
                id=item["id"],
                kind=item["kind"],
                name=item["name"],
                short=item["short"],
                aliases=list(item.get("aliases") or []),
                head=item.get("head"),
            )
            for item in raw.get("objects") or []
        ]
        self.persons: list[Person] = [
            Person(
                id=item["id"],
                name=item["name"],
                short=item["short"],
                role=item.get("role") or "",
                aliases=list(item.get("aliases") or []),
                object_id=item.get("object_id"),
            )
            for item in raw.get("persons") or []
        ]
        self.by_id = {o.id: o for o in self.objects}
        self.person_by_id = {p.id: p for p in self.persons}

        media_raw = yaml.safe_load(media_path.read_text(encoding="utf-8"))
        self.outlets = media_raw.get("outlets") or []
        self.key_media: list[str] = list(media_raw.get("key_media") or [])
        self._alias_index: list[tuple[str, str, int]] = []
        for obj in self.objects:
            terms = [obj.name, *(obj.aliases or [])]
            if obj.head:
                parts = obj.head.replace("ё", "е").split()
                if parts and len(parts[0]) >= 6:
                    terms.append(parts[0])
            for term in terms:
                term = (term or "").strip()
                if len(term) < 4:
                    continue
                self._alias_index.append((obj.id, term.lower().replace("ё", "е"), len(term)))
        self._alias_index.sort(key=lambda x: x[2], reverse=True)

        self._person_index: list[tuple[str, str, int]] = []
        for person in self.persons:
            for term in [person.short, person.name, *person.aliases]:
                t = term.strip().lower().replace("ё", "е")
                if len(t) >= 4:
                    self._person_index.append((person.id, t, len(t)))
        self._person_index.sort(key=lambda x: x[2], reverse=True)

    def match_objects(self, text: str) -> list[str]:
        hay = (text or "").lower().replace("ё", "е")
        found: list[str] = []
        for oid, term, _ in self._alias_index:
            if oid in found:
                continue
            if term in hay:
                found.append(oid)
        return found

    def match_persons(self, text: str) -> list[str]:
        hay = (text or "").lower().replace("ё", "е")
        found: list[str] = []
        for pid, term, _ in self._person_index:
            if pid in found:
                continue
            if re.search(r"(?<![а-яa-z])" + re.escape(term) + r"(?![а-яa-z])", hay):
                found.append(pid)
        return found

    def normalize_source(self, raw: str) -> tuple[str, str]:
        low = (raw or "").lower()
        for outlet in self.outlets:
            for alias in outlet.get("aliases") or []:
                if alias.lower() in low:
                    return outlet["canonical"], outlet.get("contour") or "official"
        name = re.sub(r"\s*\([^)]*\)\s*", " ", raw or "").strip()
        name = re.split(r"[#,]", name)[0].strip() or (raw or "Неизвестно")
        contour = "independent" if any(x in low for x in ("t.me/", "telegram", "dzen", "vk.com", "блог")) else "official"
        return name, contour

    def label(self, object_id: str) -> str:
        obj = self.by_id.get(object_id)
        return obj.short if obj else object_id
