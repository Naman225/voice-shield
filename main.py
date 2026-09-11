"""
Voice Shield 🛡️
AI-Powered Real-Time Voice Cloning Detection & Financial Fraud Prevention Engine
Smart India Hackathon (SIH 2026) Official Technical Backend Service
"""

import os
import sys
import io
import time
import json
import uuid
import random
import numpy as np
import soundfile as sf
from typing import Optional

# Ensure repository root is in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

from backend.service import CallSessionManager

app = FastAPI(
    title="Voice Shield 🛡️ AI Voice Cloning Detection & Fraud Prevention API",
    description="SIH 2026: Real-Time Impersonation Defense & Active Banking Fraud Gate Backend",
    version="2.0.0",
)

# Enterprise Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "microphone=(self)"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# Enable CORS for frontend and dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files & Dashboard UI
STATIC_DIR = os.path.join(_CURRENT_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global Session Manager (loads AASIST model once into GPU/CPU memory)
session_manager = CallSessionManager()

# In-memory OTP storage for step-up secondary verification
_ACTIVE_OTPS = {}


# -------------------------------------------------------------
# PYDANTIC SCHEMAS
# -------------------------------------------------------------
class StartCallRequest(BaseModel):
    call_id: Optional[str] = Field(default_factory=lambda: f"CALL-{uuid.uuid4().hex[:8].upper()}")
    caller_id: str = Field(default="Executive Desk #8941", description="Caller phone number or identifier")
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


class UnfreezeRequest(BaseModel):
    supervisor_id: str = Field(default="SUP-7701", description="Authorized supervisor credential ID")


class BankTransferRequest(BaseModel):
    call_id: str = Field(..., description="Active call monitoring session ID")
    account_number: str = Field(default="ACC-550189", description="Source bank account ID")
    beneficiary: str = Field(default="ABC Corp Ltd", description="Recipient beneficiary")
    amount: float = Field(default=2500000.0, description="Transfer amount in INR")


class VerifyOTPRequest(BaseModel):
    otp: str = Field(..., description="6-digit One Time Password for secondary verification")


class VerifySecurityQuestionRequest(BaseModel):
    answer: str = Field(..., description="Answer to security question")


class AccountInquiryRequest(BaseModel):
    call_id: str = Field(..., description="Active call monitoring session ID")
    account_number: str = Field(default="ACC-550189", description="Source bank account ID to inquire")


class SimulateRequest(BaseModel):
    mode: str = Field(
        ...,
        description="Simulation mode: CLONE (injects RED), AMBER (injects AMBER), RESET (back to GREEN)"
    )


# -------------------------------------------------------------
# CORE WEB & DASHBOARD ROUTES
# -------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Voice Shield 🛡️ Real-Time Voice Cloning Detection & Fraud Prevention",
        "status": "ONLINE",
        "version": "2.0.0",
        "target_accent": "Indian English (IndieFake Adaptive AASIST Model)",
        "dashboard_ui": "/dashboard",
        "documentation": "/docs",
    }


@app.get("/dashboard", response_class=FileResponse)
def get_dashboard():
    """Serves the interactive Voice Shield Live Monitoring SOC Dashboard."""
    dashboard_file = os.path.join(STATIC_DIR, "dashboard.html")
    if not os.path.isfile(dashboard_file):
        raise HTTPException(status_code=404, detail="Dashboard UI file not found.")
    return FileResponse(dashboard_file)


# -------------------------------------------------------------
# CALL SESSION MANAGEMENT & AUDIO FORENSICS
# -------------------------------------------------------------
@app.post("/api/v1/call/start")
def start_call(req: Optional[StartCallRequest] = None):
    """Initializes a new real-time voice call monitoring session."""
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


@app.post("/api/v1/call/upload-audio")
async def upload_audio_file(
    file: UploadFile = File(..., description="Audio file (.wav, .mp3, .ogg, .flac) to analyze"),
    call_id: Optional[str] = Form(None, description="Optional call session ID. Auto-generated if omitted."),
    caller_id: Optional[str] = Form("Forensic Inspector", description="Caller phone or identifier"),
    account_number: Optional[str] = Form("ACC-FORENSIC-INSPECT", description="Target bank account ID"),
    stream_by_chunks: bool = Form(True, description="If True, streams through sliding buffer in 1.0s increments to simulate live call progression.")
):
    """
    Uploads a recorded audio file to verify voice authenticity with fine-tuned AASIST.
    Evaluates multi-frame acoustic windows to accurately detect synthetic speech and real voices.
    """
    try:
        content = await file.read()
        audio_data, sr = sf.read(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio file '{file.filename}': {e}")

    if call_id is None:
        call_id = f"INSPECT-{uuid.uuid4().hex[:8].upper()}"

    # Use isolated inspection account to prevent bank lock contamination
    inspect_account = "ACC-FORENSIC-INSPECT" if (call_id.startswith("INSPECT-") or account_number == "ACC-550189") else account_number

    session = session_manager.get_session(call_id)
    if not session:
        session = session_manager.create_session(
            call_id=call_id,
            caller_id=caller_id,
            account_number=inspect_account,
        )

    # Convert to mono and resample to 16kHz
    mono = session.normalizer.to_mono(audio_data)
    if sr != 16000:
        mono = session.normalizer.resample(mono, orig_sr=sr)

    total_samples = len(mono)
    duration_sec = round(total_samples / 16000.0, 2)

    # Multi-frame acoustic forensic analysis
    win_len = 64600  # ~4.04s
    hop = 16000      # 1.0s hop
    frame_scores = []

    if total_samples < win_len:
        repeats = int(math.ceil(win_len / max(1, total_samples)))
        w = np.tile(mono, repeats)[:win_len]
        s = session.scorer.score_frame(w)
        frame_scores.append(s)
        session.risk_engine.update_risk(s)
    else:
        for st in range(0, total_samples - win_len + 1, hop):
            w = mono[st : st + win_len]
            if np.sqrt(np.mean(w.astype(np.float64) ** 2)) < 0.015:
                continue
            s = session.scorer.score_frame(w)
            frame_scores.append(s)
            session.risk_engine.update_risk(s)

    if not frame_scores:
        # Fallback to direct fixed length if all windows skipped
        framed = session.normalizer.fix_length(mono)
        s = session.scorer.score_frame(framed)
        frame_scores.append(s)
        session.risk_engine.update_risk(s)

    peak_score = max(frame_scores)
    mean_score = sum(frame_scores) / len(frame_scores)

    # Calibrated forensic classification
    if peak_score >= 0.70 or mean_score >= 0.55:
        risk_tier = "RED"
        running_risk_pct = round(max(peak_score * 100.0, 75.0), 1)
        verdict = "CRITICAL_VOICE_CLONE_DETECTED"
        gate_state = "LOCKED_FREEZE"
        is_frozen = True
    elif peak_score >= 0.40:
        risk_tier = "AMBER"
        running_risk_pct = round(peak_score * 100.0, 1)
        verdict = "SUSPICIOUS_VOICE_AMBER"
        gate_state = "CAUTION"
        is_frozen = False
    else:
        risk_tier = "GREEN"
        running_risk_pct = round(max(mean_score * 100.0, 5.0), 1)
        verdict = "AUTHENTIC_HUMAN_CALLER"
        gate_state = "APPROVED"
        is_frozen = False

    session.running_risk_pct = running_risk_pct
    session.current_tier = risk_tier
    session.latest_frame_score = peak_score

    res = {
        "call_id": call_id,
        "caller_id": caller_id,
        "account_number": inspect_account,
        "duration_sec": duration_sec,
        "chunks_ingested": len(frame_scores),
        "samples_in_buffer": total_samples,
        "latest_frame_score": round(peak_score * 100.0, 2),
        "running_risk_pct": running_risk_pct,
        "risk_tier": risk_tier,
        "fraud_gate": {
            "state": gate_state,
            "is_frozen": is_frozen,
            "lock_reason": "Forensic audio scan: " + ("Deepfake clone confirmed" if is_frozen else ("Acoustic anomaly detected" if risk_tier == "AMBER" else "Authentic human speech verified"))
        },
        "call_action": "CUT_CALL" if is_frozen else ("CHALLENGE_AUTHENTICATION" if risk_tier == "AMBER" else "CONTINUE"),
        "uploaded_filename": file.filename,
        "audio_duration_sec": duration_sec,
        "input_sample_rate": sr,
        "verdict": verdict,
    }
    return res


@app.get("/api/v1/call/status/{call_id}")
def get_call_status(call_id: str):
    """Fetches live call telemetry, EMA risk score, historical trend, and fraud prevention gate status."""
    session = session_manager.get_session(call_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Call session '{call_id}' not found.")
    return session.get_summary()


@app.post("/api/v1/call/unfreeze/{call_id}")
def unfreeze_transaction(call_id: str, req: Optional[UnfreezeRequest] = None):
    """Allows an authorized supervisor to unlock a frozen transaction after manual out-of-band validation."""
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
    """Ends call monitoring session and clears memory buffers."""
    success = session_manager.close_session(call_id)
    _ACTIVE_OTPS.pop(call_id, None)
    if not success:
        raise HTTPException(status_code=404, detail=f"Call session '{call_id}' not found.")
    return {"status": "SESSION_CLOSED", "call_id": call_id}


# -------------------------------------------------------------
# ACTIVE BANKING FRAUD PREVENTION & STEP-UP 2FA (PS REQUIREMENT)
# -------------------------------------------------------------
@app.post("/api/v1/bank/transfer")
def execute_wire_transfer(req: BankTransferRequest):
    """
    Executes a high-value bank wire transfer.
    Strictly protected by Voice Shield AI fraud gate:
    - 🔴 RED (Voice Clone): Returns 403 Forbidden and physically blocks transfer.
    - 🟡 AMBER (Suspicious Prosody): Returns 202 Accepted and triggers Step-Up OTP.
    - 🟢 GREEN (Authentic Human): Returns 200 OK and authorizes fund transfer.
    """
    session = session_manager.get_session(req.call_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Active call session '{req.call_id}' not found.")

    gate = session.fraud_gate.get_status()
    risk_pct = session.running_risk_pct

    if gate.get("is_frozen") or session.current_tier == "RED":
        return JSONResponse(
            status_code=403,
            content={
                "status": "TRANSFER_BLOCKED",
                "reason": "ACTIVE_VOICE_CLONE_ATTACK_DETECTED",
                "risk_score": f"{risk_pct:.1f}%",
                "lock_reason": gate.get("lock_reason"),
                "action": "Wire transfer physically locked. Out-of-band supervisor intervention required.",
                "evidence_hash": f"SHA256-{uuid.uuid4().hex}"
            }
        )

    if session.current_tier == "AMBER":
        otp_code = f"{random.randint(100000, 999999)}"
        _ACTIVE_OTPS[req.call_id] = otp_code
        print(f"\n[STEP-UP OTP DISPATCH] 📲 Sent OTP {otp_code} for call session {req.call_id} (Account: {req.account_number})")
        return JSONResponse(
            status_code=202,
            content={
                "status": "PENDING_STEP_UP_2FA",
                "message": "Elevated voice risk detected. 6-digit OTP dispatched to registered mobile.",
                "call_id": req.call_id,
                "risk_score": f"{risk_pct:.1f}%",
                "demo_otp": otp_code,
            }
        )

    # 🟢 GREEN SAFE
    receipt_id = f"TXN-{uuid.uuid4().hex[:10].upper()}"
    return {
        "status": "TRANSFER_SUCCESSFUL",
        "receipt_id": receipt_id,
        "amount": req.amount,
        "beneficiary": req.beneficiary,
        "source_account": req.account_number,
        "voice_integrity_score": f"{100.0 - risk_pct:.1f}% (Authentic Human Caller)",
        "timestamp": time.time(),
    }


@app.post("/api/v1/call/send-otp/{call_id}")
def send_otp(call_id: str):
    """Dispatches a step-up authentication OTP for a flagged call."""
    session = session_manager.get_session(call_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Call session '{call_id}' not found.")
    otp_code = f"{random.randint(100000, 999999)}"
    _ACTIVE_OTPS[call_id] = otp_code
    print(f"\n[OTP DISPATCH] 📲 Generated OTP {otp_code} for call {call_id}")
    return {
        "status": "OTP_SENT",
        "call_id": call_id,
        "otp_code_demo": otp_code,
        "demo_otp": otp_code,
        "message": f"6-digit OTP dispatched for session {call_id}."
    }


@app.post("/api/v1/call/verify-otp/{call_id}")
def verify_otp(call_id: str, req: VerifyOTPRequest):
    """Verifies OTP and unlocks the fraud gate if valid."""
    session = session_manager.get_session(call_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Call session '{call_id}' not found.")

    expected_otp = _ACTIVE_OTPS.get(call_id)
    submitted_otp = req.otp.strip()

    # Valid if matches session OTP or universal demo fallback OTP
    is_valid = (
        (expected_otp and submitted_otp == str(expected_otp).strip())
        or submitted_otp == "847291"
    )

    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid OTP code. Transaction remains locked.")

    # Unfreeze fraud gate and reset tier to GREEN
    session.current_tier = "GREEN"
    session.running_risk_pct = 5.0
    session.is_terminated = False
    session.fraud_gate.manual_override_unfreeze(f"OTP-VERIFIED-{submitted_otp}")
    _ACTIVE_OTPS.pop(call_id, None)

    # Also unlock account in registry if locked
    from backend.service import account_registry
    account_registry.unlock_account(session.account_number)

    return {
        "status": "OTP_VERIFIED_SUCCESS",
        "call_id": call_id,
        "message": "Step-up 2FA successful. Fraud gate unlocked for wire transfer.",
        "fraud_gate": session.fraud_gate.get_status()
    }


@app.post("/api/v1/call/simulate/{call_id}")
def simulate_voice_risk(call_id: str, req: SimulateRequest):
    """
    Demo / Simulation Mode — Directly injects voice risk into a call session.
    Used by the dashboard simulation chips to demonstrate the full detection
    pipeline without requiring real recorded audio.

    Modes:
      CLONE  → Injects synthetic voice clone scores (RED tier, triggers fraud gate)
      AMBER  → Injects elevated risk scores (AMBER tier, triggers OTP/challenge)
      RESET  → Resets session to GREEN (authentic voice baseline)
    """
    session = session_manager.get_session(call_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Call session '{call_id}' not found.")

    mode = req.mode.upper().strip()

    if mode == "CLONE":
        # Simulate consecutive high-score frames so EMA crosses RED threshold (>= 75%)
        for _ in range(6):
            risk_pct, tier = session.risk_engine.update_risk(0.95)
        session.running_risk_pct = risk_pct
        session.current_tier = tier
        session.latest_frame_score = 0.95
        gate_status = session.fraud_gate.evaluate_gate(risk_pct, tier)

        if tier == "RED" or gate_status.get("is_frozen"):
            from backend.service import account_registry
            account_registry.lock_account(
                session.account_number,
                f"Voice clone attack simulated (Risk: {risk_pct:.1f}%)"
            )
            session.is_terminated = True

        return {
            "status": "SIMULATION_CLONE_INJECTED",
            "risk_tier": tier,
            "running_risk_pct": risk_pct,
            "latest_frame_score": 92.0,
            "fraud_gate": gate_status,
            "call_action": "CUT_CALL" if tier == "RED" else "CONTINUE",
            "voice_prompt": "Unauthorized access detected: synthetic voice clone identified. Terminating call immediately."
        }

    elif mode == "AMBER":
        # Simulate 2 moderately-elevated frames to reach AMBER
        for _ in range(3):
            risk_pct, tier = session.risk_engine.update_risk(0.60)
        session.running_risk_pct = risk_pct
        session.current_tier = tier
        session.latest_frame_score = 0.60
        gate_status = session.fraud_gate.evaluate_gate(risk_pct, tier)
        from backend.service import account_registry
        acc = account_registry.get_account(session.account_number)
        otp_code = f"{random.randint(100000, 999999)}"
        _ACTIVE_OTPS[call_id] = otp_code
        print(f"\n[SIMULATION OTP] 📲 Registered OTP {otp_code} for simulated AMBER call {call_id}")
        return {
            "status": "SIMULATION_AMBER_INJECTED",
            "risk_tier": tier,
            "running_risk_pct": risk_pct,
            "latest_frame_score": 60.0,
            "fraud_gate": gate_status,
            "call_action": "CHALLENGE_AUTHENTICATION",
            "security_question": acc.get("security_question"),
            "demo_otp": otp_code,
            "voice_prompt": f"Caution: Elevated voice anomaly detected. Please verify your identity: {acc.get('security_question', 'What is your pet name?')}"
        }

    elif mode == "RESET":
        # Reset to clean GREEN baseline
        session.risk_engine.reset()
        session.running_risk_pct = 5.0
        session.current_tier = "GREEN"
        session.latest_frame_score = 0.05
        session.is_terminated = False
        session.fraud_gate.reset()
        gate_status = session.fraud_gate.get_status()
        return {
            "status": "SIMULATION_RESET",
            "risk_tier": "GREEN",
            "running_risk_pct": 5.0,
            "latest_frame_score": 5.0,
            "fraud_gate": gate_status,
            "call_action": "CONTINUE",
            "voice_prompt": None
        }

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown simulation mode '{req.mode}'. Use: CLONE, AMBER, or RESET"
        )


@app.post("/api/v1/call/inquiry")
def inquire_account_details(req: AccountInquiryRequest):
    """
    Simulates inquiring about confidential bank account balance and details.
    Protected by Voice Shield real-time acoustic forensics:
    - 🔴 RED (Voice Clone): Returns 403, triggers auto-cut voice prompt, locks account.
    - 🟡 AMBER (Caution): Returns 202, challenges with security question before balance reveal.
    - 🟢 GREEN (Authentic Human): Returns 200 with verified account balance.
    """
    session = session_manager.get_session(req.call_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Active call session '{req.call_id}' not found.")

    from backend.service import account_registry
    acc = account_registry.get_account(req.account_number)

    if acc["is_locked"] or session.current_tier == "RED":
        return JSONResponse(
            status_code=403,
            content={
                "status": "INQUIRY_DENIED_UNAUTHORIZED",
                "call_action": "CUT_CALL",
                "reason": "VOICE_CLONE_ATTACK_DETECTED",
                "voice_prompt": "Unauthorized access detected: synthetic voice clone identified. Terminating call immediately.",
                "message": "Account inquiry blocked due to voice clone detection. Session terminated."
            }
        )

    if session.current_tier == "AMBER":
        return JSONResponse(
            status_code=202,
            content={
                "status": "INQUIRY_CHALLENGE_REQUIRED",
                "call_action": "CHALLENGE_AUTHENTICATION",
                "security_question": acc["security_question"],
                "voice_prompt": f"Caution: Voice anomaly detected. Please verify your identity: {acc['security_question']}",
                "message": "Security verification required before confidential account details are disclosed."
            }
        )

    return {
        "status": "INQUIRY_SUCCESSFUL",
        "account_number": req.account_number,
        "owner": acc["owner"],
        "balance": f"₹ {acc['balance']:,.2f}",
        "account_status": "ACTIVE_VERIFIED",
        "voice_integrity": f"{100.0 - session.running_risk_pct:.1f}% (Authentic Human Caller)",
        "message": "Identity verified. Confidential details cleared."
    }


@app.post("/api/v1/call/verify-question/{call_id}")
def verify_security_question(call_id: str, req: VerifySecurityQuestionRequest):
    """
    Verifies the user's answer to their security question (e.g. pet name, school).
    - If correct: Clears caution state and authorizes transactions/inquiries.
    - If incorrect: Escalates to RED, cuts call, and locks account.
    """
    session = session_manager.get_session(call_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Active call session '{call_id}' not found.")

    from backend.service import account_registry
    acc = account_registry.get_account(session.account_number)

    if req.answer.strip().lower() != acc["security_answer"].strip().lower():
        session.current_tier = "RED"
        session.fraud_gate.evaluate_gate(99.0, "RED")
        account_registry.lock_account(session.account_number, "Failed security question challenge under voice caution.")
        return JSONResponse(
            status_code=403,
            content={
                "status": "CHALLENGE_FAILED_UNAUTHORIZED",
                "call_action": "CUT_CALL",
                "voice_prompt": "Security verification failed. Unauthorized access detected. Terminating call immediately.",
                "message": "Incorrect answer. Account locked and call cut."
            }
        )

    session.fraud_gate.manual_override_unfreeze("SECURITY-QUESTION-VERIFIED")
    session.current_tier = "GREEN"
    session.running_risk_pct = 5.0
    return {
        "status": "CHALLENGE_PASSED",
        "call_action": "CONTINUE",
        "voice_prompt": "Identity successfully verified. You may proceed.",
        "message": "Security check passed. Transaction and inquiry authorized."
    }


@app.post("/api/v1/account/unlock/{account_number}")
def unlock_account(account_number: str):
    """Supervisor endpoint to unlock an account locked due to repeat attacks."""
    from backend.service import account_registry
    account_registry.unlock_account(account_number)
    session_manager.unlock_account_sessions(account_number)
    return {"status": "ACCOUNT_UNLOCKED", "account_number": account_number}


# -------------------------------------------------------------
# ADMIN OVERSIGHT & SUPERVISOR CONTROL PANEL
# -------------------------------------------------------------
@app.get("/api/v1/admin/accounts")
def admin_get_all_accounts():
    """
    Returns full state of all accounts in the fraud registry.
    Used by the Admin & Security Hub in the dashboard.
    """
    from backend.service import account_registry
    return {
        "status": "OK",
        "accounts": account_registry.accounts
    }


@app.post("/api/v1/admin/account/unlock/{account_number}")
def admin_unlock_account(account_number: str):
    """
    Supervisor re-authorization: unlocks a frozen account and clears
    all active call session fraud gate locks for that account.
    """
    from backend.service import account_registry
    account_registry.unlock_account(account_number)
    session_manager.unlock_account_sessions(account_number)
    return {
        "status": "ACCOUNT_UNLOCKED",
        "account_number": account_number,
        "message": f"Account {account_number} has been cleared by admin override."
    }


@app.post("/api/v1/admin/account/lock/{account_number}")
def admin_lock_account(account_number: str):
    """
    Manual supervisor lockout — for testing fraud gate behavior or
    emergency freeze of a suspected compromised account.
    """
    from backend.service import account_registry
    account_registry.lock_account(account_number, "Manual admin lockout via supervisor console.")
    return {
        "status": "ACCOUNT_LOCKED",
        "account_number": account_number,
        "message": f"Account {account_number} manually frozen by admin override."
    }


# -------------------------------------------------------------
# WEBSOCKET REAL-TIME AUDIO STREAMING (FOR MIC & VOIP)
# -------------------------------------------------------------
@app.websocket("/ws/call/{call_id}")
async def websocket_call_stream(websocket: WebSocket, call_id: str):
    """
    Bi-directional WebSocket real-time audio stream.
    Receives streaming audio chunks from browser microphone or VoIP PBX.
    Returns real-time AASIST risk updates, tier badges, and active fraud gate alerts.
    """
    await websocket.accept()

    query_params = websocket.query_params
    acc_num = query_params.get("account", "ACC-550189")
    caller = query_params.get("caller_id", "Executive Line (+91 98765 43210)")

    session = session_manager.get_session(call_id)
    if not session:
        session = session_manager.create_session(
            call_id=call_id,
            caller_id=caller,
            account_number=acc_num,
        )
    else:
        session.account_number = acc_num
        session.caller_id = caller

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            samples = message.get("samples", [])
            sample_rate = message.get("sample_rate", 16000)
            is_simulated_clone = bool(message.get("is_simulated_clone", False))
            if "account_number" in message:
                session.account_number = message["account_number"]
            if "caller_id" in message:
                session.caller_id = message["caller_id"]

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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir=_CURRENT_DIR)
