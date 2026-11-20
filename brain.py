"""Explainable, bounded decision engine for safe simulations."""
from dataclasses import dataclass
from math import exp, isfinite
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class Decision:
    action: str
    score: float
    reason: str
    confidence: float
    evidence: tuple


def _number(value: Any, label: str) -> float:
    """Convert a numeric input and reject NaN/infinity explicitly."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


class ZeusBrain:
    """Score candidate actions, learn bounded weights, and use similar outcomes.

    Confidence is a relative score-gap heuristic, not a probability of success.
    """
    def __init__(self, learning_rate: float = 0.1, min_weight: float = -2.0,
                 max_weight: float = 2.0, experience_scale: float = 0.25):
        self.learning_rate = _number(learning_rate, "learning_rate")
        self.min_weight = _number(min_weight, "min_weight")
        self.max_weight = _number(max_weight, "max_weight")
        self.experience_scale = _number(experience_scale, "experience_scale")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.min_weight >= self.max_weight:
            raise ValueError("min_weight must be less than max_weight")
        if not 0 <= self.experience_scale <= 1:
            raise ValueError("experience_scale must be between 0 and 1")
        self.weights: Dict[str, float] = {
            "move_toward_goal": 0.5,
            "avoid_hazard": 0.5,
            "wait": 0.0,
        }
        self.feedback_count = 0

    @staticmethod
    def _similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
        keys = set(left) | set(right)
        if not keys:
            return 1.0
        pairs = [(_number(left.get(k, 0.0), f"left[{k}]"),
                  _number(right.get(k, 0.0), f"right[{k}]")) for k in keys]
        magnitude = max((max(abs(a), abs(b)) for a, b in pairs), default=0.0)
        if magnitude == 0.0:
            return 1.0
        # Scale before subtraction/summation to avoid overflow on large finite inputs.
        numerator = sum(abs(a / magnitude - b / magnitude) for a, b in pairs)
        denominator = sum(abs(a / magnitude) + abs(b / magnitude) for a, b in pairs)
        return max(0.0, 1.0 - numerator / (denominator + 1.0 / magnitude))

    @staticmethod
    def _validate_features(features: Mapping[str, Any], label: str) -> Dict[str, float]:
        if not isinstance(features, Mapping):
            raise ValueError(f"{label} must be a mapping of feature names to numbers")
        clean: Dict[str, float] = {}
        for name, value in features.items():
            if not isinstance(name, str) or not name:
                raise ValueError("feature names must be non-empty strings")
            clean[name] = _number(value, f"{label}.{name}")
        return clean

    def decide(self, features_by_action: Dict[str, Dict[str, float]],
               experiences: Optional[List[Dict[str, Any]]] = None) -> Optional[Decision]:
        if not isinstance(features_by_action, Mapping):
            raise ValueError("features_by_action must be a mapping")
        if not features_by_action:
            return None
        if experiences is not None and not isinstance(experiences, list):
            raise ValueError("experiences must be a list")

        ranked = []
        for action, raw_features in features_by_action.items():
            if not isinstance(action, str) or not action:
                raise ValueError("action names must be non-empty strings")
            features = self._validate_features(raw_features, f"features[{action}]")
            base = _number(sum(self.weights.get(name, 0.0) * value
                               for name, value in features.items()), "base score")
            weighted_reward = 0.0
            similarity_total = 0.0
            matches = 0
            for item in experiences or []:
                if not isinstance(item, Mapping) or item.get("action") != action:
                    continue
                past = item.get("features")
                if not isinstance(past, Mapping):
                    continue
                similarity = self._similarity(features, past)
                if similarity <= 0:
                    continue
                reward = max(-1.0, min(1.0, _number(item.get("reward", 0.0), "experience.reward")))
                weighted_reward += similarity * reward
                similarity_total += similarity
                matches += 1
            evidence_score = weighted_reward / similarity_total if similarity_total else 0.0
            adjustment = self.experience_scale * evidence_score
            score = _number(base + adjustment, "decision score")
            evidence = (f"feature score {base:.3f}",
                        f"similar-experience adjustment {adjustment:+.3f} ({matches} weighted matches)")
            ranked.append(Decision(action, score,
                                   f"{evidence[0]}; {evidence[1]}", 0.5, evidence))

        # Stable ordering preserves caller-supplied order for exact ties.
        ranked.sort(key=lambda item: item.score, reverse=True)
        top = ranked[0]
        if len(ranked) > 1:
            gap = max(0.0, top.score - ranked[1].score)
            confidence = 1.0 / (1.0 + exp(-min(gap, 20.0)))
        else:
            confidence = 0.5  # No alternative means no comparative evidence.
        return Decision(top.action, top.score, top.reason, confidence, top.evidence)

    def learn(self, action: str, reward: float, features: Dict[str, float]) -> None:
        """Update observed feature weights using bounded, validated feedback."""
        if not isinstance(action, str) or not action:
            raise ValueError("action must be a non-empty string")
        clean = self._validate_features(features, "features")
        bounded_reward = max(-1.0, min(1.0, _number(reward, "reward")))
        # Calculate first, then commit, so invalid input cannot partially update state.
        updates = {}
        for name, value in clean.items():
            updated = self.weights.get(name, 0.0) + self.learning_rate * bounded_reward * value
            updates[name] = max(self.min_weight, min(self.max_weight, updated))
        self.weights.update(updates)
        self.feedback_count += 1

    def status(self) -> Dict[str, Any]:
        return {"feedback_count": self.feedback_count,
                "weights": dict(self.weights),
                "learning_rate": self.learning_rate,
                "experience_scale": self.experience_scale}
