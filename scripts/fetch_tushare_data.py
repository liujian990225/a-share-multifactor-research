from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mf_strategy.config import load_config
from mf_strategy.tushare_loader import fetch_tushare_to_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Tushare data and normalize it for the multi-factor backtest.")
    parser.add_argument("--config", default="configs/config_tushare.yaml", help="Path to Tushare YAML config.")
    parser.add_argument("--token", default=None, help="Optional Tushare token. Prefer environment variable TUSHARE_TOKEN.")
    parser.add_argument("--force", action="store_true", help="Ignore normalized CSV cache and download again.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    paths = fetch_tushare_to_csv(config, token=args.token, force=args.force)
    print("\nGenerated files:")
    print(f"  prices       : {paths.prices}")
    print(f"  fundamentals : {paths.fundamentals}")
    print(f"  benchmark    : {paths.benchmark}")
    print(f"  membership   : {paths.membership}")
    print("\nNext step:")
    print("  python -m mf_strategy.cli --config configs/config_tushare.yaml")


if __name__ == "__main__":
    main()
