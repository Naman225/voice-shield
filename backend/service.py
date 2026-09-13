"""
backend/service.py
------------------
Call session lifecycle manager & Stateful Account Fraud Registry.
Controls call interception, auto-hangup on RED, AMBER security challenges,
and cross-session account lockout for repeat attackers.
"""

import time
import numpy as np
from typing import Dict, Any, Optional

from backend.normalizer import AudioNormalizer
from backend.buffer    import SlidingAudioBuffer
from backend.scorer    import AcousticScorer
from backend.risk_engine import EMARiskEngine
from backend.fraud_gate  import FraudPreventionGate


class AccountFraudRegistry:
    """
    Stateful banking account security registry.
    Tracks account balances, security challenge questions, lockouts,
    and blocks repeat attackers immediately without requiring RAG.
    """
    def __init__(self):
        self.accounts = {
            "ACC-550189": {
                "owner": "Executive Line (+91 98765 43210)",
                "balance": 1240500.0,
                "is_locked": False,
                "lock_reason": None,
                "locked_timestamp": None,
                "security_question": "What is your first pet's name?",
                "security_answer": "Max",
                "attack_count": 0
            },
            "ACC-9948201": {
                "owner": "Treasury Line (+91 98111 22334)",
                "balance": 4850000.0,
                "is_locked": False,
                "lock_reason": None,
                "locked_timestamp": None,
                "security_question": "What was the name of your first school?",
                "security_answer": "St. Xavier's",
                "attack_count": 0
            }
        }

    def get_account(self, account_number: str) -> dict:
        if account_number not in self.accounts:
            self.accounts[account_number] = {
                "owner": f"User ({account_number})",
                "balance": 750000.0,
                "is_locked": False,
                "lock_reason": None,
                "locked_timestamp": None,
                "security_question": "What is your favorite city?",
                "security_answer": "Mumbai",
                "attack_count": 0
            }
        return self.accounts[account_number]

    def lock_account(self, account_number: str, reason: str):
        acc = self.get_account(account_number)
        acc["is_locked"] = True
        acc["lock_reason"] = reason
        acc["locked_timestamp"] = time.time()
        acc["attack_count"] += 1

    def unlock_account(self, account_number: str):
        acc = self.get_account(account_number)
        acc["is_locked"] = False
        acc["lock_reason"] = None
        acc["locked_timestamp"] = None


# Global singleton account registry
account_registry = AccountFraudRegistry()


class ActiveCallSession:
    """Represents a single live telephone / VoIP call session being monitored."""

    def __init__(
        self,
        call_id:        str,
        caller_id:      str,
        account_number: str,
        scorer:         AcousticScorer,
    ):
        self.call_id        = call_id
        self.caller_id      = caller_id
        self.account_number = account_number
        self.start_time     = time.time()

        self.normalizer  = AudioNormalizer()
        self.buffer      = SlidingAudioBuffer()
        self.scorer      = scorer
        self.risk_engine = EMARiskEngine()
        self.fraud_gate  = FraudPreventionGate()

        self.total_chunks_processed = 0
        self.latest_frame_score     = 0.05
        self.running_risk_pct       = 5.0
        self.current_tier           = "GREEN"
        self.is_terminated          = False

    def process_audio_chunk(
        self,
        pcm_samples:       np.ndarray,
        sample_rate:       int  = 16000,
        is_simulated_clone: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest audio chunk, update sliding buffer, execute AASIST,
        and trigger auto-cut or security challenges.
        """
        self.total_chunks_processed += 1
        duration_sec = round(time.time() - self.start_time, 1)

        # 0. Check if account is ALREADY locked from a previous attack
        acc = account_registry.get_account(self.account_number)
        if acc["is_locked"]:
            self.is_terminated = True
            return {
                "call_id":            self.call_id,
                "caller_id":          self.caller_id,
                "account_number":     self.account_number,
                "duration_sec":       duration_sec,
                "chunks_ingested":    self.total_chunks_processed,
                "samples_in_buffer":  len(self.buffer._buffer),
                "latest_frame_score": 100.0,
                "running_risk_pct":   100.0,
                "risk_tier":          "RED",
                "fraud_gate": {
                    "state": "LOCKED_FREEZE",
                    "is_frozen": True,
                    "lock_reason": f"Repeat attack blocked. Account {self.account_number} is locked."
                },
                "call_action": "CUT_CALL",
                "voice_prompt": f"Unauthorized access: Account {self.account_number} is locked due to a recent voice clone attack. Terminating call immediately.",
                "security_question": None
            }

        # 1. Normalise to mono @ 16 kHz
        mono_samples = self.normalizer.to_mono(pcm_samples)
        if sample_rate != 16000:
            mono_samples = self.normalizer.resample(mono_samples, orig_sr=sample_rate)

        # 2. Push into sliding buffer
        self.buffer.push_samples(mono_samples)

        # 3. Run AASIST inference once at least 1.0 s (16,000 samples) or full window is filled
        if self.buffer.is_window_ready() or len(self.buffer._buffer) >= 16000:
            window_audio = self.buffer.get_latest_window()

            # VAD Gate: If the audio window is mostly silence / ambient noise,
            # skip inference entirely and inject a bonafide (0.0) score.
            # This prevents false deepfake escalation when the caller is
            # pausing, listening, or silent between utterances.
            window_rms = float(np.sqrt(np.mean(window_audio.astype(np.float64) ** 2)))
            if window_rms < 0.002:
                # Near-silent window — treat as authentic (no voice = no spoof)
                self.latest_frame_score = 0.0
            elif is_simulated_clone:
                # Demo / simulation mode: bypass AASIST and inject a realistic
                # synthetic voice clone score to demonstrate RED detection.
                # Uses a high but not instant score so the EMA ramp-up is visible.
                self.latest_frame_score = 0.92
            else:
                self.latest_frame_score = self.scorer.score_frame(window_audio)

            self.running_risk_pct, self.current_tier = self.risk_engine.update_risk(
                self.latest_frame_score
            )

        # 4. Evaluate Fraud Gate
        gate_status = self.fraud_gate.evaluate_gate(self.running_risk_pct, self.current_tier)

        # 5. Interactive Call Interception Decisions
        call_action = "CONTINUE"
        voice_prompt = None
        security_q = None

        if self.current_tier == "RED" or gate_status.get("is_frozen"):
            # Lock account across future calls
            account_registry.lock_account(
                self.account_number,
                f"Critical voice clone detected (Risk: {self.running_risk_pct:.1f}%)"
            )
            self.is_terminated = True
            call_action = "CUT_CALL"
            voice_prompt = "Unauthorized user access detected: synthetic voice clone identified. Terminating call immediately."

        elif self.current_tier == "AMBER":
            call_action = "CHALLENGE_AUTHENTICATION"
            security_q = acc["security_question"]
            voice_prompt = f"Caution: Elevated voice anomaly detected. Please verify your identity: {security_q}"

        return {
            "call_id":            self.call_id,
            "caller_id":          self.caller_id,
            "account_number":     self.account_number,
            "duration_sec":       duration_sec,
            "chunks_ingested":    self.total_chunks_processed,
            "samples_in_buffer":  len(self.buffer._buffer),
            "latest_frame_score": round(self.latest_frame_score * 100.0, 2),
            "running_risk_pct":   self.running_risk_pct,
            "risk_tier":          self.current_tier,
            "fraud_gate":         gate_status,
            "call_action":        call_action,
            "voice_prompt":       voice_prompt,
            "security_question":  security_q,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return complete session state and historical risk trend."""
        return {
            "call_id":          self.call_id,
            "caller_id":        self.caller_id,
            "account_number":   self.account_number,
            "duration_sec":     round(time.time() - self.start_time, 1),
            "chunks_ingested":  self.total_chunks_processed,
            "running_risk_pct": self.running_risk_pct,
            "risk_tier":        self.current_tier,
            "fraud_gate":       self.fraud_gate.get_status(),
            "risk_history":     self.risk_engine.get_history(),
            "is_terminated":    self.is_terminated,
        }


class CallSessionManager:
    """Manages active call sessions across the backend service."""

    def __init__(self):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scorer:   AcousticScorer               = AcousticScorer(device=device)
        self.sessions: Dict[str, ActiveCallSession] = {}

    def create_session(
        self,
        call_id:        str,
        caller_id:      str,
        account_number: str,
    ) -> ActiveCallSession:
        session = ActiveCallSession(
            call_id=call_id,
            caller_id=caller_id,
            account_number=account_number,
            scorer=self.scorer,
        )
        self.sessions[call_id] = session
        return session

    def get_session(self, call_id: str) -> Optional[ActiveCallSession]:
        return self.sessions.get(call_id)

    def close_session(self, call_id: str) -> bool:
        if call_id in self.sessions:
            del self.sessions[call_id]
            return True
        return False

    def unlock_account_sessions(self, account_number: str) -> int:
        """
        Clears fraud gate locks on all active call sessions belonging to
        a given account (called after admin/supervisor unlocks the account).
        Returns the number of sessions that were cleared.
        """
        cleared = 0
        for session in self.sessions.values():
            if session.account_number == account_number:
                session.fraud_gate.manual_override_unfreeze("ADMIN-ACCOUNT-UNLOCK")
                session.current_tier = "GREEN"
                session.running_risk_pct = 5.0
                session.is_terminated = False
                cleared += 1
        return cleared
