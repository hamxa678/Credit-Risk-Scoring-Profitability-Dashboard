import os
import subprocess
import sys

required = [
    "src/models/train.py",
    "dashboard/app.py",
    "tests/test_evaluate.py",
]
missing = [p for p in required if not os.path.exists(p)]
if missing:
    print("Missing:", missing)
    sys.exit(1)
print("Smoke test passed.")
