from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate_lstm_fd001_test import run_test_evaluation


def main() -> None:
    result = run_test_evaluation()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
