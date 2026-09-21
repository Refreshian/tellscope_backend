#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Синхронизация страниц документации Tellscope с Wiki.js.

Как это работает: страницы лежат в markdown-файлах в /opt/tellscope-wiki/pages,
рядом — manifest.json с заголовками и описаниями. Скрипт входит в wiki служебной
учётной записью (доступ в /root/.tellscope/wiki-sync.json) и приводит страницы
в wiki к тому, что лежит в файлах: новые создаёт, изменённые обновляет.

Примеры:
  sudo /home/dev/tellscope_app/tellscope_backend/venv_py312_clean/bin/python \
      /opt/tellscope-wiki/wiki_sync.py --dry-run      # показать, что изменится
  ... wiki_sync.py                                    # применить
  ... wiki_sync.py --only nabory-dannyh               # одна страница
  ... wiki_sync.py --home-nav                         # обновить навигацию на Home
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:8080"
PAGES_DIR = "/opt/tellscope-wiki/pages"
MANIFEST = "/opt/tellscope-wiki/manifest.json"
CREDS = "/root/.tellscope/wiki-sync.json"
NAV_START = "<!-- nav:start -->"
NAV_END = "<!-- nav:end -->"


class Wiki:
    """Минимальный клиент Wiki.js: вход по логину и GraphQL-вызовы."""

    def __init__(self, base: str, email: str, password: str):
        self.base = base.rstrip("/")
        self.email = email
        self.password = password
        self.jwt = ""

    def call(self, query: str, variables: dict | None = None) -> dict:
        payload = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.jwt:
            headers["Cookie"] = "jwt=" + self.jwt
        request = urllib.request.Request(self.base + "/graphql", data=payload, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return {"httpError": exc.code, "body": exc.read().decode("utf-8", "replace")[:300]}

    def login(self) -> bool:
        result = self.call(
            "mutation ($u: String!, $p: String!, $s: String!) {"
            " authentication { login(username: $u, password: $p, strategy: $s) {"
            " jwt responseResult { succeeded message } } } }",
            {"u": self.email, "p": self.password, "s": "local"})
        login = (((result.get("data") or {}).get("authentication") or {}).get("login") or {})
        self.jwt = login.get("jwt") or ""
        if not self.jwt:
            print("не удалось войти:", json.dumps(result, ensure_ascii=False)[:300])
            return False
        return True

    def pages(self) -> dict:
        result = self.call("{ pages { list { id path title locale isPublished } } }")
        items = (((result.get("data") or {}).get("pages") or {}).get("list")) or []
        return {row["path"]: row for row in items}

    def content(self, page_id: int) -> str:
        result = self.call("query ($id: Int!) { pages { single(id: $id) { content } } }", {"id": page_id})
        single = (((result.get("data") or {}).get("pages") or {}).get("single")) or {}
        return single.get("content") or ""

    def create(self, row: dict, content: str) -> dict:
        return self.call(
            "mutation ($content: String!, $description: String!, $editor: String!,"
            " $isPublished: Boolean!, $isPrivate: Boolean!, $locale: String!, $path: String!,"
            " $tags: [String]!, $title: String!) { pages { create(content: $content,"
            " description: $description, editor: $editor, isPublished: $isPublished,"
            " isPrivate: $isPrivate, locale: $locale, path: $path, tags: $tags, title: $title)"
            " { responseResult { succeeded message } page { id path } } } }",
            {"content": content, "description": row.get("description", ""), "editor": "markdown",
             "isPublished": True, "isPrivate": False, "locale": row.get("locale", "en"),
             "path": row["path"], "tags": [], "title": row.get("title", row["path"])})

    def update(self, page_id: int, row: dict, content: str) -> dict:
        return self.call(
            "mutation ($id: Int!, $content: String!, $description: String!, $editor: String!,"
            " $isPublished: Boolean!, $isPrivate: Boolean!, $locale: String!, $path: String!,"
            " $tags: [String]!, $title: String!) { pages { update(id: $id, content: $content,"
            " description: $description, editor: $editor, isPublished: $isPublished,"
            " isPrivate: $isPrivate, locale: $locale, path: $path, tags: $tags, title: $title)"
            " { responseResult { succeeded message } page { id path } } } }",
            {"id": page_id, "content": content, "description": row.get("description", ""),
             "editor": "markdown", "isPublished": True, "isPrivate": False,
             "locale": row.get("locale", "en"), "path": row["path"], "tags": [],
             "title": row.get("title", row["path"])})


def load_settings() -> dict:
    path = CREDS
    if not os.path.isfile(path):
        print("нет файла доступа %s — запустите под sudo" % path)
        sys.exit(2)
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_pages() -> list:
    with io.open(MANIFEST, encoding="utf-8") as fh:
        manifest = json.load(fh)
    out = []
    for row in manifest:
        path = os.path.join(PAGES_DIR, row["file"])
        if not os.path.isfile(path):
            print("нет файла:", path)
            continue
        with io.open(path, encoding="utf-8") as fh:
            row = dict(row)
            row["content"] = fh.read()
        out.append(row)
    return out


def sync(wiki: Wiki, pages: list, only: str = "", dry: bool = False) -> dict:
    existing = wiki.pages()
    stats = {"created": [], "updated": [], "skipped": [], "failed": []}
    for row in pages:
        path = row["path"]
        if only and path != only:
            continue
        page = existing.get(path)
        if not page:
            if dry:
                stats["created"].append(path)
                continue
            result = wiki.create(row, row["content"])
            status = ((((result.get("data") or {}).get("pages") or {}).get("create") or {})
                      .get("responseResult") or {})
            (stats["created"] if status.get("succeeded") else stats["failed"]).append(
                path if status.get("succeeded") else "%s: %s" % (path, status.get("message")))
            continue
        current = wiki.content(page["id"])
        if current.strip() == row["content"].strip():
            stats["skipped"].append(path)
            continue
        if dry:
            stats["updated"].append(path)
            continue
        result = wiki.update(page["id"], row, row["content"])
        status = ((((result.get("data") or {}).get("pages") or {}).get("update") or {})
                  .get("responseResult") or {})
        (stats["updated"] if status.get("succeeded") else stats["failed"]).append(
            path if status.get("succeeded") else "%s: %s" % (path, status.get("message")))
    return stats


def home_nav(wiki: Wiki, pages: list, dry: bool = False) -> str:
    """Собирает на главной странице блок со ссылками на разделы (в начале страницы).

    Блок ставится в начало: открыв wiki, сразу видно список разделов, а не только
    вводный текст. Повторный запуск заменяет блок, дублей не появляется.
    """
    existing = wiki.pages()
    home = existing.get("home")
    if not home:
        print("страница home не найдена")
        return "нет home"
    content = wiki.content(home["id"])

    groups = {}
    for row in pages:
        if row["path"] == "home" or not row.get("nav"):
            continue
        groups.setdefault(row.get("nav_group", "Разделы"), []).append(row)

    lines = [NAV_START, "## Разделы документации", "",
             "Выберите раздел — на каждой странице описан порядок работы, показатели и типовые "
             "ситуации. Ниже остался вводный обзор сервиса.", ""]
    for group, rows in groups.items():
        lines.append("### " + group)
        lines.append("")
        for row in rows:
            lines.append("* [%s](/%s) — %s" % (row.get("title", row["path"]), row["path"],
                                               row.get("description", "")))
        lines.append("")

    block = "\n".join(lines).rstrip() + "\n" + NAV_END
    if NAV_START in content and NAV_END in content:
        start = content.index(NAV_START)
        end = content.index(NAV_END) + len(NAV_END)
        rest = (content[:start] + "\n\n" + content[end:]).strip()
    else:
        rest = content.strip()
    # прежний заголовок первого уровня опускаем до второго: на странице один главный заголовок
    rest = re.sub(r"^#\s+(?!#)", "## ", rest, count=1)
    # блок всегда в начале страницы: открыв wiki, сразу видно разделы
    new_content = block + ("\n\n" + rest if rest else "")
    if new_content.strip() == content.strip():
        return "без изменений"
    if dry:
        return "будет обновлено (%d знаков)" % len(new_content)
    row = {"path": "home", "title": home.get("title") or "Home Page",
           "description": "", "locale": home.get("locale") or "ru"}
    result = wiki.update(home["id"], row, new_content)
    status = ((((result.get("data") or {}).get("pages") or {}).get("update") or {})
              .get("responseResult") or {})
    return "обновлено" if status.get("succeeded") else "ошибка: %s" % status.get("message")


def main() -> None:
    parser = argparse.ArgumentParser(description="Синхронизация документации с Wiki.js")
    parser.add_argument("--dry-run", action="store_true", help="только показать изменения")
    parser.add_argument("--only", default="", help="обновить одну страницу по пути")
    parser.add_argument("--home-nav", action="store_true", help="обновить навигацию на Home")
    args = parser.parse_args()

    settings = load_settings()
    wiki = Wiki(settings.get("base", DEFAULT_BASE), settings["email"], settings["password"])
    if not wiki.login():
        sys.exit(3)
    pages = load_pages()
    print("страниц в файлах: %d" % len(pages))

    stats = sync(wiki, pages, only=args.only, dry=args.dry_run)
    print("создано: %d %s" % (len(stats["created"]), stats["created"][:5]))
    print("обновлено: %d %s" % (len(stats["updated"]), stats["updated"][:5]))
    print("без изменений: %d" % len(stats["skipped"]))
    if stats["failed"]:
        print("ОШИБКИ: %s" % stats["failed"][:5])
    if args.home_nav or not args.only:
        print("навигация на Home:", home_nav(wiki, pages, dry=args.dry_run))


if __name__ == "__main__":
    main()
