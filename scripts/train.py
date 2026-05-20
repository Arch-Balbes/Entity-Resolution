#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import run_training  # noqa: E402


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()
