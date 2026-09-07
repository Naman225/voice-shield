"""
backend/service.py
------------------
Call session lifecycle manager.

Changes from Nikunj's original:
- AcousticScorer (AASIST model) is loaded ONCE in CallSessionManager
  and shared across all sessions to avoid re-loading the model per call.
- ActiveCallSession receives the shared scorer via dependency injection.
"""

import time
import numpy as np
from typing import Dict, Any, Optional

from backend.normalizer import AudioNormalizer
from backend.buffer    import SlidingAudioBuffer
from backend.scorer    import AcousticScorer
from backend.risk_engine import EMARiskEngine
from backend.fraud_gate  import FraudPreventionGate


class ActiveCallSession:
    """Represents a single live telephone / VoIP call session being monitored."""

    def __init__(
        self,
        call_id:        str,
        caller_id:      str,
        account_number: str,
        scorer:         AcousticScorer,          # shared, pre-loaded model
    ):
        self.call_id        = call_id
        self.caller_id      = caller_id
        self.account_number = account_number
        self.start_time     = time.time()

        self.normalizer  = AudioNormalizer()
        self.buffer      = SlidingAudioBuffer()
        self.scorer      = scorer                # injected — NOT re-loaded here
        self.risk_engine = EMARiskEngine()
        self.fraud_gate  = FraudPreventionGate()

        self.total_chunks_processed = 0
        self.latest_frame_score     = 0.05
        self.running_risk_pct       = 5.0
        self.current_tier           = "GREEN"

    def process_audio_chunk(
        self,
        pcm_samples:       np.ndarray,
        sample_rate:       int  = 16000,
        is_simulated_clone: bool = False,   # kept for backward-compat, ignored by AASIST scorer
    ) -> Dict[str, Any]:
        """
        Ingest a raw PCM chunk, update sliding buffer, run AASIST inference
        when the 4.04 s window is ready, update EMA risk, and evaluate the
        fraud prevention gate.
        """
        self.total_chunks_processed += 1

        # 1. Normalise to mono @ 16 kHz
        mono_samples = self.normalizer.to_mono(pcm_samples)
        if sample_rate != 16000:
            mono_samples = self.normalizer.resample(mono_samples, orig_sr=sample_rate)

        # 2. Push into sliding buffer
        self.buffer.push_samples(mono_samples)

        # 3. Run AASIST inference once the 4.04 s window is filled
        if self.buffer.is_window_ready():
            window_audio = self.buffer.get_latest_window()
            self.latest_frame_score = self.scorer.score_frame(window_audio)

            # 4. Update EMA risk score → tier
            self.running_risk_pct, self.current_tier = self.risk_engine.update_risk(
                self.latest_frame_score
            )

        # 5. Evaluate active fraud prevention gate
        gate_status  = self.fraud_gate.evaluate_gate(self.running_risk_pct, self.current_tier)
        duration_sec = round(time.time() - self.start_time, 1)

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
        }


class CallSessionManager:
    """
    Manages active call sessions across the backend service.

    The AASIST model is loaded ONCE here and shared across all sessions.
    This avoids the ~3-second model-load penalty per incoming call.
    """

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
            scorer=self.scorer,              # pass the shared model in
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
