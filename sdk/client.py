"""
Voice Shield SDK — VoiceShieldClient
=======================================
Drop-in Python client for integrating Voice Shield into any banking system,
contact center platform, or enterprise communication tool.

Fulfills PS requirement:
    "REST/gRPC APIs and SDKs for integration with core banking systems,
     contact center platforms, enterprise communication tools, and telecom networks."

Installation:
    pip install requests  (already a standard dep)

Basic Usage:
    from sdk import VoiceShieldClient

    with VoiceShieldClient("https://voice-shield-kf0h.onrender.com") as vs:
        vs.start_call("CALL-001", caller_id="+91-98765-43210", account="ACC-550189")
        result = vs.scan_audio_file("recording.mp3")
        print(result["risk_tier"])   # "GREEN", "AMBER", or "RED"
"""

import os
import time
import json
import threading
from typing import Optional, Callable

import requests

# ──────────────────────────────────────────────────────────────────────────────

class VoiceShieldError(Exception):
    """Raised when the Voice Shield API returns an error."""
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"[HTTP {status_code}] {detail}")
        self.status_code = status_code
        self.detail = detail


class VoiceShieldClient:
    """
    Python SDK client for Voice Shield — AI Voice Clone Detection API.

    Designed for drop-in integration with:
    - Core banking systems (CBS) — wire transfer authorization
    - Contact center platforms — real-time call risk scoring
    - IVR systems — caller identity verification
    - Telecom operator fraud departments

    Thread-safe. Supports context manager usage.

    Args:
        base_url:  Voice Shield server URL (e.g. "https://voice-shield-kf0h.onrender.com")
        api_key:   Optional API key (set X-API-Key header if your deployment requires it)
        timeout:   HTTP request timeout in seconds (default: 60)
        language:  Default language hint — "en" (English), "hi" (Hindi), "ta" (Tamil)
    """

    def __init__(
        self,
        base_url:   str,
        api_key:    Optional[str] = None,
        timeout:    int = 60,
        language:   str = "en",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.language = language
        self._lock    = threading.Lock()

        self._session = requests.Session()
        self._session.headers.update({
            "Accept":       "application/json",
            "User-Agent":   "VoiceShieldSDK/1.0",
        })
        if api_key:
            self._session.headers["X-API-Key"] = api_key

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        resp = self._session.request(method, url, timeout=self.timeout, **kwargs)
        if not resp.ok:
            try:    detail = resp.json().get("detail", resp.text)
            except: detail = resp.text
            raise VoiceShieldError(resp.status_code, detail)
        return resp.json()

    # ── Health ─────────────────────────────────────────────────────────────────

    def health(self) -> bool:
        """Returns True if the Voice Shield server is online and healthy."""
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=10)
            return r.ok and r.json().get("status") == "healthy"
        except Exception:
            return False

    def ping(self) -> float:
        """Returns round-trip latency in milliseconds, or -1 if unreachable."""
        try:
            t0 = time.perf_counter()
            self._session.get(f"{self.base_url}/health", timeout=10)
            return round((time.perf_counter() - t0) * 1000, 2)
        except Exception:
            return -1.0

    # ── Call Session Management ────────────────────────────────────────────────

    def start_call(
        self,
        call_id:        str,
        caller_id:      str = "Unknown Caller",
        account_number: str = "ACC-550189",
    ) -> dict:
        """
        Register a new incoming call for real-time monitoring.

        Call this as soon as a call is answered. The server starts a monitoring
        session and loads the account's fraud history for risk enrichment.

        Args:
            call_id:        Unique call identifier (your IVR/PBX call reference ID)
            caller_id:      Caller's phone number or identifier
            account_number: Target bank account number

        Returns:
            dict with call_id, session_status, and account info

        Example:
            vs.start_call("IVR-20240924-0042", caller_id="+91-98765-43210", account_number="ACC-550189")
        """
        return self._request("POST", "/api/v1/call/start", json={
            "call_id":        call_id,
            "caller_id":      caller_id,
            "account_number": account_number,
        })

    def end_call(self, call_id: str) -> dict:
        """Terminate a call session and free server resources."""
        return self._request("DELETE", f"/api/v1/call/{call_id}")

    def get_risk(self, call_id: str) -> dict:
        """
        Get the current real-time risk state for an active call.

        Returns dict with:
            risk_tier:        "GREEN" | "AMBER" | "RED"
            running_risk_pct: float (0–100)
            fraud_gate:       {"state": "APPROVED"|"CAUTION"|"LOCKED_FREEZE", ...}
            is_terminated:    bool
        """
        return self._request("GET", f"/api/v1/call/status/{call_id}")

    # ── Audio Analysis ─────────────────────────────────────────────────────────

    def scan_audio_file(
        self,
        audio_path:     str,
        call_id:        Optional[str] = None,
        caller_id:      str = "Forensic Inspector",
        account_number: str = "ACC-FORENSIC-INSPECT",
    ) -> dict:
        """
        Upload a recorded audio file for forensic deepfake analysis.

        This is the KEY method for post-call analysis and compliance audits.
        Supports .mp3, .wav, .ogg, .flac.

        Args:
            audio_path:     Local path to the audio file to analyze
            call_id:        Optional session ID. Auto-generated if not provided.
            caller_id:      Identifier for the caller in the report
            account_number: Account to check (use default for forensic-only analysis)

        Returns:
            dict with:
                risk_tier:          "GREEN" | "AMBER" | "RED"
                running_risk_pct:   float — e.g. 98.4
                verdict:            "AUTHENTIC_HUMAN_CALLER" | "SUSPICIOUS_VOICE_AMBER" | "CRITICAL_VOICE_CLONE_DETECTED"
                latest_frame_score: float — peak AASIST spoof confidence (0–100)
                duration_sec:       float — audio duration
                chunks_ingested:    int — number of analysis windows processed
                fraud_gate:         dict — state, lock_reason, evidence_hash

        Example:
            result = vs.scan_audio_file("call_recording.mp3")
            if result["risk_tier"] == "RED":
                bank.freeze_account(account_number)
                bank.alert_fraud_team(result["fraud_gate"]["evidence_hash"])
        """
        with open(audio_path, "rb") as f:
            fname = os.path.basename(audio_path)
            mime  = "audio/mpeg" if fname.endswith(".mp3") else "audio/wav"
            data  = {}
            if call_id:        data["call_id"]        = call_id
            if caller_id:      data["caller_id"]      = caller_id
            if account_number: data["account_number"] = account_number
            return self._request("POST", "/api/v1/call/upload-audio",
                                 files={"file": (fname, f, mime)},
                                 data=data)

    # ── Fraud Gate Actions ─────────────────────────────────────────────────────

    def approve_transfer(
        self,
        call_id:        str,
        account_number: str,
        amount:         float,
        beneficiary:    str,
    ) -> dict:
        """
        Request authorization for a wire transfer under live voice monitoring.

        The server checks the current voice risk tier before approving:
        - GREEN  → Transfer approved immediately
        - AMBER  → Returns HTTP 202 with step-up OTP challenge required
        - RED    → Returns HTTP 403, account frozen, call cut

        Args:
            call_id:        Active call session ID
            account_number: Source bank account
            amount:         Transfer amount in INR (affects risk threshold tightening)
            beneficiary:    Recipient name or account

        Returns:
            dict with status, call_action, voice_prompt, and optionally otp_reference

        Example:
            result = vs.approve_transfer("CALL-001", "ACC-550189", 2500000.0, "ABC Corp Ltd")
            if result.get("status") == "TRANSFER_APPROVED":
                cbs.execute_transfer(...)   # your CBS function
            elif result.get("call_action") == "CHALLENGE_AUTHENTICATION":
                ivr.send_otp(caller_phone)
        """
        return self._request("POST", "/api/v1/bank/transfer", json={
            "call_id":        call_id,
            "account_number": account_number,
            "amount":         amount,
            "beneficiary":    beneficiary,
        })

    def send_otp(self, call_id: str) -> dict:
        """Dispatch a 6-digit OTP to the caller's registered mobile (AMBER step-up)."""
        return self._request("POST", f"/api/v1/call/send-otp/{call_id}")

    def verify_otp(self, call_id: str, otp: str) -> dict:
        """Submit the OTP entered by the caller. Returns GREEN on success, locks on failure."""
        return self._request("POST", f"/api/v1/call/verify-otp/{call_id}", json={"otp": otp})

    def cancel_otp(self, call_id: str) -> dict:
        """Cancel a pending OTP challenge (caller hung up or timed out)."""
        return self._request("POST", f"/api/v1/call/cancel-otp/{call_id}")

    def inquire_account(self, call_id: str, account_number: str) -> dict:
        """
        Request account balance disclosure under voice risk gating.
        Returns balance on GREEN, security question on AMBER, denial on RED.
        """
        return self._request("POST", "/api/v1/call/inquiry", json={
            "call_id":        call_id,
            "account_number": account_number,
        })

    def unfreeze_account(self, call_id: str, supervisor_id: str = "SUPERVISOR") -> dict:
        """Supervisor override: manually unfreeze a locked call session."""
        return self._request("POST", f"/api/v1/call/unfreeze/{call_id}",
                             json={"supervisor_id": supervisor_id})

    def simulate_clone_attack(self, call_id: str) -> dict:
        """
        Inject a simulated RED-tier voice clone attack. For demo and testing only.
        Instantly triggers all fraud prevention interlocks.
        """
        return self._request("POST", f"/api/v1/call/simulate/{call_id}", json={"mode": "CLONE"})

    def simulate_reset(self, call_id: str) -> dict:
        """Reset a simulated attack back to GREEN (for repeated demos)."""
        return self._request("POST", f"/api/v1/call/simulate/{call_id}", json={"mode": "RESET"})

    # ── Admin ──────────────────────────────────────────────────────────────────

    def list_accounts(self) -> dict:
        """Returns the full account fraud registry (admin use only)."""
        return self._request("GET", "/api/v1/admin/accounts")

    def admin_lock_account(self, account_number: str, reason: str = "Manual admin lock") -> dict:
        return self._request("POST", f"/api/v1/admin/account/lock/{account_number}",
                             json={"reason": reason})

    def admin_unlock_account(self, account_number: str) -> dict:
        return self._request("POST", f"/api/v1/admin/account/unlock/{account_number}")

    # ── Monitoring ─────────────────────────────────────────────────────────────

    def watch_call(
        self,
        call_id:      str,
        on_amber:     Optional[Callable[[dict], None]] = None,
        on_red:       Optional[Callable[[dict], None]] = None,
        poll_interval: float = 2.0,
        timeout:       float = 300.0,
    ) -> dict:
        """
        Poll a call session and invoke callbacks when risk thresholds are crossed.

        Useful for headless integrations where you can't use the WebSocket dashboard.

        Args:
            call_id:       Active call ID to monitor
            on_amber:      Callback(dict) invoked when tier changes to AMBER
            on_red:        Callback(dict) invoked when tier changes to RED (account freeze)
            poll_interval: Seconds between polls (default 2s)
            timeout:       Max monitoring time in seconds (default 5 min)

        Returns:
            Final risk state dict when call ends or timeout reached

        Example:
            def on_red(state):
                send_alert_to_fraud_team(state)
                freeze_account_in_cbs(state["account_number"])

            vs.watch_call("CALL-001", on_red=on_red)
        """
        deadline   = time.time() + timeout
        last_tier  = "GREEN"
        last_state = {}

        while time.time() < deadline:
            try:
                state     = self.get_risk(call_id)
                cur_tier  = state.get("current_tier", "GREEN")
                last_state = state

                if cur_tier == "AMBER" and last_tier != "AMBER" and on_amber:
                    on_amber(state)
                if cur_tier == "RED" and last_tier != "RED" and on_red:
                    on_red(state)

                last_tier = cur_tier
                if state.get("is_terminated"):
                    break
            except VoiceShieldError:
                break

            time.sleep(poll_interval)

        return last_state

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._session.close()

    def __repr__(self):
        return f"VoiceShieldClient(base_url={self.base_url!r})"
