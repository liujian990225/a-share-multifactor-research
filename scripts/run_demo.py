from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mf_strategy.cli import run_pipeline


if __name__ == "__main__":
    run_pipeline(ROOT / "configs" / "config_demo.yaml")
