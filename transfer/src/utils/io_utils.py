import os
import json
from pathlib import Path

import numpy as np


def to_jsonable(x):
    """Convert common non-JSON types (numpy, paths) into JSON-serializable Python types."""
    if x is None:
        return None

    # numpy scalars
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)

    # numpy arrays
    if isinstance(x, np.ndarray):
        return x.tolist()

    # paths
    if isinstance(x, Path):
        return str(x)

    # containers
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]

    # fallback: leave as-is if already JSON-friendly (str/int/float/bool)
    # otherwise stringify to prevent crashes
    if isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(to_jsonable(row)) + '\n')
