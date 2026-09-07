import os
import sys
import json
import uuid
import numpy as np
from typing import List, Optional

# Ensure repository root is in sys.path so 'backend' and 'src' resolve regardless of working directory
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend.service import CallSessionManager

app = FastAPI(
    title="AI-Powered Voice Cloning Detection & Prevention API",
    description="SIH 2026 Real-Time Impersonation Defense & Active Banking Fraud Gate Backend Service",
    version="1.0.0",
)

# Enable CORS for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Session Manager
session_manager = CallSessionManager()


# -------------------------------------------------------------
# PYDANTIC SCHEMAS
# -------------------------------------------------------------
class StartCallRequest(BaseModel):
    call_id: Optional[str] = Field(default_factory=lambda: f"CALL-{uuid.uuid4().hex[:8].upper()}")
    caller_id: str = Field(default="Executive Desk #8941", description="Phone number or caller ID")
    account_number: str = Field(default="ACC-9948201", description="Target bank account ID")

    model_config = {
        "json_schema_extra": {
            "example": {
                "call_id": "CALL-DEMO-001",
                "caller_id": "CEO Executive Line (+91 98765 43210)",
                "account_number": "ACC-550189",
            }
        }
    }


class AudioChunkRequest(BaseModel):
    call_id: Optional[str] = Field(default="CALL-DEMO-001", description="Call session ID returned by /call/start")
    samples: List[float] = Field(default_factory=lambda: [0.0, 0.045, 0.09, 0.13, 0.17, 0.21, 0.24, 0.27, 0.3, 0.32], description="Float32 PCM audio samples in range [-1.0, 1.0]")
    sample_rate: int = Field(default=16000, description="Audio sample rate (e.g., 8000, 16000, 44100)")
    is_simulated_clone: bool = Field(default=False, description="Set True to simulate a deepfake attack for testing")

    model_config = {
        "json_schema_extra": {
            "example": {
                "call_id": "CALL-DEMO-001",
                "samples": [0.0, 0.045, 0.09, 0.13, 0.17, 0.21, 0.24, 0.27, 0.3, 0.32],
                "sample_rate": 16000,
                "is_simulated_clone": False,
            }
        }
    }

    @field_validator("samples")
    @classmethod
    def samples_must_not_be_empty(cls, v: List[float]) -> List[float]:
        if len(v) == 0:
            raise ValueError("'samples' must not be empty. Provide at least 1 PCM float32 audio sample.")
        return v


class UnfreezeRequest(BaseModel):
    supervisor_id: str = Field(default="SUP-7701", description="Authorized supervisor credential ID")


# -------------------------------------------------------------
# REST ENDPOINTS
# -------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "AI-Powered Real-Time Voice Cloning Detection Backend",
        "status": "ONLINE",
        "version": "1.0.0",
        "target_accent": "Indian English (IndieFake Adaptive Engine)",
        "documentation": "/docs",
    }


@app.post("/api/v1/call/start")
def start_call(req: Optional[StartCallRequest] = None):
    """Initializes a new real-time call monitoring session."""
    if req is None:
        req = StartCallRequest()
    session = session_manager.create_session(
        call_id=req.call_id,
        caller_id=req.caller_id,
        account_number=req.account_number,
    )
    return {
        "status": "SESSION_INITIALIZED",
        "call_id": session.call_id,
        "caller_id": session.caller_id,
        "account_number": session.account_number,
        "buffer_target": "64,600 samples (~4.04s at 16kHz)",
        "hop_step": "16,000 samples (1.0s)",
    }


@app.post("/api/v1/call/chunk")
def ingest_chunk(req: Optional[AudioChunkRequest] = None):
    """Ingests a raw PCM audio chunk, updates sliding buffer, calculates EMA risk, and updates fraud gate."""
    if req is None:
        req = AudioChunkRequest()
    session = session_manager.get_session(req.call_id)
    if not session:
        # Auto-create session if not present so chunk processing never fails
        session = session_manager.create_session(
            call_id=req.call_id,
            caller_id="Executive Desk #8941",
            account_number="ACC-9948201",
        )

    pcm_array = np.array(req.samples, dtype=np.float32)
    result = session.process_audio_chunk(
        pcm_samples=pcm_array,
        sample_rate=req.sample_rate,
        is_simulated_clone=req.is_simulated_clone,
    )
    return result


@app.get("/api/v1/call/status/{call_id}")
def get_call_status(call_id: str):
    """Fetches live call metrics, EMA risk score, history trend, and fraud prevention gate status."""
    session = session_manager.get_session(call_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Call session '{call_id}' not found.")
    return session.get_summary()


@app.post("/api/v1/call/unfreeze/{call_id}")
def unfreeze_transaction(call_id: str, req: Optional[UnfreezeRequest] = None):
    """Allows an authorized supervisor to unlock a frozen transaction after manual 2FA validation."""
    if req is None:
        req = UnfreezeRequest()
    session = session_manager.get_session(call_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Call session '{call_id}' not found.")

    updated_gate = session.fraud_gate.manual_override_unfreeze(req.supervisor_id)
    return {
        "status": "TRANSACTION_UNFROZEN",
        "call_id": call_id,
        "fraud_gate": updated_gate,
    }


@app.delete("/api/v1/call/{call_id}")
def end_call(call_id: str):
    """Ends call monitoring session and clears buffers."""
    success = session_manager.close_session(call_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Call session '{call_id}' not found.")
    return {"status": "SESSION_CLOSED", "call_id": call_id}


# -------------------------------------------------------------
# WEBSOCKET REAL-TIME AUDIO STREAMING ENDPOINT
# -------------------------------------------------------------
@app.websocket("/ws/call/{call_id}")
async def websocket_call_stream(websocket: WebSocket, call_id: str):
    """
    Bi-directional WebSocket real-time audio stream.
    Client sends JSON packets containing audio sample arrays or control messages.
    Backend returns real-time risk updates, tier badges, and active fraud gate alerts.
    """
    await websocket.accept()

    session = session_manager.get_session(call_id)
    if not session:
        session = session_manager.create_session(
            call_id=call_id,
            caller_id="VoIP Incoming Channel",
            account_number="ACC-DEFAULT",
        )

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            samples = message.get("samples", [])
            sample_rate = message.get("sample_rate", 16000)
            is_simulated_clone = message.get("is_simulated_clone", False)

            if samples:
                pcm_array = np.array(samples, dtype=np.float32)
                response = session.process_audio_chunk(
                    pcm_samples=pcm_array,
                    sample_rate=sample_rate,
                    is_simulated_clone=is_simulated_clone,
                )
                await websocket.send_json(response)
            else:
                await websocket.send_json({"status": "NO_SAMPLES_RECEIVED"})

    except WebSocketDisconnect:
        print(f"[WebSocket] Disconnected from call stream: {call_id}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
