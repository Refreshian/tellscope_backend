from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mosinform",
        description="Собрать рейтинг медиаприсутствия: выгрузки Медиалогии/СКАН → Excel + PPTX",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="обработать папку с файлами и собрать отчёт")
    run.add_argument("input_dir", type=Path, help="папка с docx/xlsx выгрузками")
    run.add_argument("-o", "--output", type=Path, default=Path("output"), help="куда положить Excel и PPTX")
    run.add_argument("--period", default="", help='подписи периода, например "Июль 2026"')
    run.add_argument("--tellscope", action="store_true", help="включить разметку через Tellscope")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        if not args.input_dir.exists():
            print(f"нет папки {args.input_dir}", file=sys.stderr)
            return 2
        result = run_pipeline(args.input_dir, args.output, period=args.period, tellscope=args.tellscope)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
