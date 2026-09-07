import time
from typing import Dict, List, Tuple


class EMARiskEngine:
    """
    Dynamic Exponential Moving Average (EMA) Risk Engine.
    Smoothes single-frame neural score spikes to prevent false positives from background noise/coughs.
    Formula: RunningRisk(t) = alpha * RunningRisk(t-1) + (1 - alpha) * FrameScore
    Default alpha = 0.70 (70% previous history weight, 30% new window weight).
    """

    def __init__(self, alpha: float = 0.70):
        self.alpha = alpha
        self.running_risk = 0.05  # Initialize at safe baseline 5%
        self.history: List[Tuple[float, float, str]] = []  # (timestamp, score, tier)

    def update_risk(self, frame_score: float) -> Tuple[float, str]:
        """
        Updates the running risk score with a new frame score.
        Returns (running_risk_pct, risk_tier).
        """
        # Apply 70/30 EMA smoothing formula
        self.running_risk = (self.alpha * self.running_risk) + ((1.0 - self.alpha) * frame_score)
        self.running_risk = float(np_clip(self.running_risk, 0.0, 1.0))

        tier = self.get_risk_tier(self.running_risk)
        self.history.append((time.time(), self.running_risk, tier))

        return round(self.running_risk * 100.0, 2), tier

    @staticmethod
    def get_risk_tier(score: float) -> str:
        """Categorizes score into 3-tier action levels."""
        pct = score * 100.0 if score <= 1.0 else score
        if pct < 40.0:
            return "GREEN"
        elif pct < 75.0:
            return "AMBER"
        else:
            return "RED"

    def get_history(self) -> List[Dict]:
        """Returns formatted timestamped risk history for visualization graphs."""
        return [
            {"timestamp": ts, "risk_pct": round(score * 100.0, 2), "tier": tier}
            for ts, score, tier in self.history
        ]

    def reset(self):
        """Reset risk score and history."""
        self.running_risk = 0.05
        self.history.clear()


def np_clip(val: float, min_val: float, max_val: float) -> float:
    """Helper clip function."""
    return max(min_val, min(val, max_val))
