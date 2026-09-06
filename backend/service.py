import time
import numpy as np
from typing import Dict, Any, Optional

from backend.normalizer import AudioNormalizer
from backend.buffer import SlidingAudioBuffer
from backend.scorer import AcousticScorer
from backend.risk_engine import EMARiskEngine
from backend.fraud_gate import FraudPreventionGate


class ActiveCallSession:
    """Represents a single live telephone / VoIP call session being monitored."""

    def __init__(self, call_id: str, caller_id: str, account_number: str):
        self.call_id = call_id
        self.caller_id = caller_id
        self.account_number = account_number
        self.start_time = time.time()

        self.normalizer = AudioNormalizer()
        self.buffer = SlidingAudioBuffer()
        self.scorer = AcousticScorer()
        self.risk_engine = EMARiskEngine()
        self.fraud_gate = FraudPreventionGate()

        self.total_chunks_processed = 0
        self.latest_frame_score = 0.05
        self.running_risk_pct = 5.0
        self.current_tier = "GREEN"

    def process_audio_chunk(
        self, pcm_samples: np.ndarray, sample_rate: int = 16000, is_simulated_clone: bool = False
    ) -> Dict[str, Any]:
        """
        Ingests a raw PCM audio chunk, updates sliding buffer, executes inference frame
        if 1.0s hop interval is reached, updates EMA risk score, and evaluates fraud gate.
        """
        self.total_chunks_processed += 1

        # 1. Normalize mono & sample rate
        mono_samples = self.normalizer.to_mono(pcm_samples)
        if sample_rate != 16000:
            mono_samples = self.normalizer.resample(mono_samples, orig_sr=sample_rate)

        # 2. Push to sliding buffer
        self.buffer.push_samples(mono_samples)

        # 3. Check if window snapshot should be evaluated (or evaluate on every chunk if buffer ready)
        if self.buffer.is_window_ready():
            window_audio = self.buffer.get_latest_window()
            # 4. Score frame
            self.latest_frame_score = self.scorer.score_frame(
                window_audio, is_simulated_clone=is_simulated_clone
            )
            # 5. Update EMA risk score
            self.running_risk_pct, self.current_tier = self.risk_engine.update_risk(
                self.latest_frame_score
            )

        # 6. Evaluate Active Fraud Prevention Gate
        gate_status = self.fraud_gate.evaluate_gate(self.running_risk_pct, self.current_tier)

        duration_sec = round(time.time() - self.start_time, 1)

        return {
            "call_id": self.call_id,
            "caller_id": self.caller_id,
            "account_number": self.account_number,
            "duration_sec": duration_sec,
            "chunks_ingested": self.total_chunks_processed,
            "samples_in_buffer": len(self.buffer._buffer),
            "latest_frame_score": round(self.latest_frame_score * 100.0, 2),
            "running_risk_pct": self.running_risk_pct,
            "risk_tier": self.current_tier,
            "fraud_gate": gate_status,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Returns complete session state summary and historical risk trend."""
        duration_sec = round(time.time() - self.start_time, 1)
        return {
            "call_id": self.call_id,
            "caller_id": self.caller_id,
            "account_number": self.account_number,
            "duration_sec": duration_sec,
            "chunks_ingested": self.total_chunks_processed,
            "running_risk_pct": self.running_risk_pct,
            "risk_tier": self.current_tier,
            "fraud_gate": self.fraud_gate.get_status(),
            "risk_history": self.risk_engine.get_history(),
        }


class CallSessionManager:
    """Manages active call sessions across the backend service."""

    def __init__(self):
        self.sessions: Dict[str, ActiveCallSession] = {}

    def create_session(self, call_id: str, caller_id: str, account_number: str) -> ActiveCallSession:
        session = ActiveCallSession(call_id, caller_id, account_number)
        self.sessions[call_id] = session
        return session

    def get_session(self, call_id: str) -> Optional[ActiveCallSession]:
        return self.sessions.get(call_id)

    def close_session(self, call_id: str) -> bool:
        if call_id in self.sessions:
            del self.sessions[call_id]
            return True
        return False
