from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mf_strategy.cli import run_pipeline


if __name__ == "__main__":
    run_pipeline(PROJECT_ROOT / "configs" / "config_tushare.yaml")
