import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_folder_naming import build_run_folder_name, parse_run_folder_name


def _build_cmd(args: argparse.Namespace) -> int:
    name = build_run_folder_name(
        kind=args.kind,
        stage=args.stage,
        base=args.base,
        method=args.method,
        adapter=args.adapter,
        quant=args.quant,
        ctx=args.ctx,
        rows=args.rows,
        run_id=args.run_id,
    )
    print(name)
    return 0


def _parse_cmd(args: argparse.Namespace) -> int:
    parsed = parse_run_folder_name(args.name)
    if not parsed:
        print("not_canonical")
        return 1
    print(json.dumps(parsed, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build/parse canonical run folder names")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build canonical folder name")
    b.add_argument("--kind", required=True)
    b.add_argument("--stage", required=True)
    b.add_argument("--base", required=True)
    b.add_argument("--method", required=True)
    b.add_argument("--adapter", required=True)
    b.add_argument("--quant", required=True)
    b.add_argument("--ctx", required=True)
    b.add_argument("--rows", required=True)
    b.add_argument("--run-id", required=True)
    b.set_defaults(func=_build_cmd)

    p = sub.add_parser("parse", help="Parse canonical folder name")
    p.add_argument("--name", required=True)
    p.set_defaults(func=_parse_cmd)
    return ap


def main() -> int:
    args = _parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
