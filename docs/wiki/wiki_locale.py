#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Работа с локалями wiki: установка, перенос страниц, выбор основной.

Примеры (от root на сервере):
  python wiki_locale.py --list                     # какие локали установлены
  python wiki_locale.py --install ru               # установить локаль
  python wiki_locale.py --migrate en ru            # перенести страницы из en в ru
  python wiki_locale.py --default ru               # сделать локаль основной
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wiki_sync  # noqa: E402


def connect() -> "wiki_sync.Wiki":
    settings = wiki_sync.load_settings()
    wiki = wiki_sync.Wiki(settings.get("base", "http://127.0.0.1:8080"),
                          settings["email"], settings["password"])
    if not wiki.login():
        sys.exit(3)
    return wiki


def locales(wiki) -> list:
    result = wiki.call("{ localization { locales { code name isRTL } } }")
    return (((result.get("data") or {}).get("localization") or {}).get("locales")) or []


def pages(wiki) -> list:
    result = wiki.call("{ pages { list { id path locale title } } }")
    return (((result.get("data") or {}).get("pages") or {}).get("list")) or []


def main() -> None:
    parser = argparse.ArgumentParser(description="Локали wiki Tellscope")
    parser.add_argument("--list", action="store_true", help="показать установленные локали и страницы")
    parser.add_argument("--install", default="", help="установить локаль, например ru")
    parser.add_argument("--migrate", nargs=2, metavar=("ОТКУДА", "КУДА"), help="перенести страницы")
    parser.add_argument("--default", default="", help="сделать локаль основной")
    args = parser.parse_args()

    wiki = connect()

    if args.install:
        result = wiki.call('mutation { localization { downloadLocale(locale: "%s") '
                           '{ responseResult { succeeded message } } } }' % args.install)
        print("установка %s:" % args.install, json.dumps(result, ensure_ascii=False)[:300])

    if args.migrate:
        source, target = args.migrate
        result = wiki.call('mutation { pages { migrateToLocale(sourceLocale: "%s", targetLocale: "%s") '
                           '{ responseResult { succeeded message } } } }' % (source, target))
        print("перенос %s → %s:" % (source, target), json.dumps(result, ensure_ascii=False)[:300])

    if args.default:
        result = wiki.call('mutation { localization { updateLocale(locale: "%s", autoUpdate: true, '
                           'namespacing: false, namespaces: ["%s"]) { responseResult { succeeded message } } } }'
                           % (args.default, args.default))
        print("основная локаль %s:" % args.default, json.dumps(result, ensure_ascii=False)[:300])

    installed = locales(wiki)
    print("локали: %s" % ", ".join("%s (%s)" % (row["code"], row["name"]) for row in installed))
    from collections import Counter
    counts = Counter(row["locale"] for row in pages(wiki))
    print("страницы: %s" % dict(counts))


if __name__ == "__main__":
    main()
