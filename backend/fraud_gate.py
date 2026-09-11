import time
from typing import Dict, Any


class FraudPreventionGate:
    """
    Active Banking Fraud Prevention Gate.
    Monitors call risk tier and triggers automated security holds on sensitive transaction operations.
    """

    def __init__(self, high_risk_threshold_pct: float = 75.0):
        self.high_risk_threshold_pct = high_risk_threshold_pct
        self.state = "APPROVED"  # APPROVED, CAUTION, LOCKED_FREEZE
        self.is_frozen = False
        self.freeze_timestamp = None
        self.lock_reason = None
        self.secondary_2fa_triggered = False

    def evaluate_gate(self, running_risk_pct: float, tier: str) -> Dict[str, Any]:
        """
        Evaluates risk score against threshold and updates gate state.
        """
        if tier == "RED" or running_risk_pct >= self.high_risk_threshold_pct:
                self.is_frozen = True
                self.state = "LOCKED_FREEZE"
                self.freeze_timestamp = time.time()
                import hashlib
                self.evidence_hash = f"SHA256-{hashlib.sha256(f'{self.freeze_timestamp}:{running_risk_pct}'.encode()).hexdigest()[:16]}"
                self.lock_reason = f"CRITICAL VOICE CLONE DETECTED (Risk Score: {running_risk_pct:.1f}%)"
                self.secondary_2fa_triggered = True

        elif tier == "AMBER" and not self.is_frozen:
            self.state = "CAUTION"
            self.lock_reason = f"Elevated Risk / Acoustic Distortion (Risk Score: {running_risk_pct:.1f}%)"

        elif tier == "GREEN" and not self.is_frozen:
            self.state = "APPROVED"
            self.lock_reason = "Authentic Voice Verified"

        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Returns active fraud prevention state status."""
        return {
            "state": self.state,
            "is_frozen": self.is_frozen,
            "transaction_button_enabled": not self.is_frozen,
            "lock_reason": self.lock_reason,
            "freeze_timestamp": self.freeze_timestamp,
            "evidence_hash": getattr(self, "evidence_hash", None),
            "secondary_2fa_required": self.secondary_2fa_triggered,
            "action_prompt": "TRANSACTION APPROVAL LOCKED: Voice Cloning Attack Detected. Triggering Out-of-Band 2FA Callback."
            if self.is_frozen
            else "Transaction Authorized.",
        }

    def manual_override_unfreeze(self, supervisor_id: str) -> Dict[str, Any]:
        """Allows authorized supervisor to unlock transaction after 2FA validation."""
        self.is_frozen = False
        self.state = "APPROVED"
        self.lock_reason = f"Manually verified & unlocked by Supervisor #{supervisor_id}"
        self.secondary_2fa_triggered = False
        return self.get_status()

    def reset(self):
        """Reset gate state."""
        self.state = "APPROVED"
        self.is_frozen = False
        self.freeze_timestamp = None
        self.lock_reason = None
        self.secondary_2fa_triggered = False
