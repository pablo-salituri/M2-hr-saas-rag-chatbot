"""Persistence utilities for historical query results."""

import json
from datetime import datetime
from pathlib import Path


def save_query_result(result: dict) -> str:
    """Persist a query result as a timestamped JSON file.

    Args:
        result: Query payload with question, answer, chunks, and evaluation.

    Returns:
        Path to the saved JSON file as a string.
    """
    project_root = Path(__file__).resolve().parent.parent
    historical_dir = project_root / "outputs" / "historical"
    historical_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    filename = now.strftime("%Y%m%d_%H%M%S.json")
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")

    payload = {"timestamp": timestamp, **result}
    file_path = historical_dir / filename
    file_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return str(file_path)
