"""Append-only JSONL experience memory with validation and summaries."""
import json
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperienceMemory:
    def __init__(self, path: str = "zeus_memory.jsonl"):
        self.path = Path(path)

    def record(self, state: Dict[str, Any], action: str, reward: float,
               outcome: str, features: Optional[Dict[str, float]] = None) -> None:
        if not isinstance(state, dict):
            raise ValueError("state must be a dictionary")
        if not isinstance(action, str) or not action:
            raise ValueError("action must be a non-empty string")
        if not isinstance(outcome, str) or not outcome:
            raise ValueError("outcome must be a non-empty string")
        if features is not None and not isinstance(features, dict):
            raise ValueError("features must be a dictionary")
        try:
            numeric_reward = float(reward)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("reward must be a finite number") from exc
        if isinstance(reward, bool) or not isfinite(numeric_reward):
            raise ValueError("reward must be a finite number")
        clean_features = {}
        for name, value in (features or {}).items():
            if not isinstance(name, str) or not name:
                raise ValueError("feature names must be non-empty strings")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"feature {name!r} must be finite numeric data") from exc
            if isinstance(value, bool) or not isfinite(number):
                raise ValueError(f"feature {name!r} must be finite numeric data")
            clean_features[name] = number
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "action": action,
            "features": clean_features,
            "reward": max(-1.0, min(1.0, numeric_reward)),
            "outcome": outcome,
        }
        # Serialize before opening the file: invalid state data cannot leave a partial line.
        try:
            line = json.dumps(entry, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("state must contain JSON-serializable finite values") from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        entries = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Corrupt memory at line {line_number}: {exc.msg}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"Corrupt memory at line {line_number}: expected an object")
                entries.append(item)
        return entries

    def summary(self) -> Dict[str, Any]:
        entries = self.all()
        if not entries:
            return {"total": 0, "average_reward": 0.0, "positive_rate": 0.0,
                    "outcomes": {}}
        rewards = [float(item.get("reward", 0.0)) for item in entries]
        outcomes: Dict[str, int] = {}
        for item in entries:
            key = str(item.get("outcome", "unknown"))
            outcomes[key] = outcomes.get(key, 0) + 1
        return {"total": len(entries),
                "average_reward": sum(rewards) / len(rewards),
                "positive_rate": sum(r > 0 for r in rewards) / len(rewards),
                "outcomes": outcomes}
