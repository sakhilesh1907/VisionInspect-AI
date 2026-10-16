"""
amd_client.py — AMD Token Factory AI Integration
Connects VisionInspect AI to AMD Radeon Cloud Token Factory models
hosted on AMD Instinct accelerators.

Supported capabilities:
  - Visual Language Defect Diagnosis (DeepSeek-V4-Flash-Vision-Exp / MiniCPM-V)
  - Interactive Human Review Copilot (DeepSeek-V4.1-Flash / Qwen3.8-27B)
  - Shift QA / ISO Audit Report Generation (Qwen3.8-27B)
"""

import os
import time
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
from dotenv import load_dotenv

# Load environment variables from project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

DEFAULT_API_KEY = "rc-1ea138b3676693430ce491ba9730b8d4a50437453b53e842"
DEFAULT_BASE_URL = "https://developer.amd.com.cn/radeon/api/v1"
DEFAULT_VISION_MODEL = "MiniCPM-V-4.7-35B-A3B"
DEFAULT_TEXT_MODEL = "Qwen3.8-Flash-Next"
DEFAULT_REPORT_MODEL = "Qwen3.8-27B"

def get_api_key() -> str:
    return os.getenv("AMD_API_KEY", DEFAULT_API_KEY)

def get_base_url() -> str:
    return os.getenv("AMD_API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

def get_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json"
    }

def check_amd_status() -> Dict[str, Any]:
    """Check connectivity and list models available from AMD Token Factory."""
    url = f"{get_base_url()}/models"
    try:
        resp = requests.get(url, headers=get_headers(), timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            model_ids = [m.get("id") for m in data if "id" in m]
            return {
                "online": True,
                "status_code": 200,
                "models_count": len(model_ids),
                "models": model_ids,
                "vision_model": os.getenv("AMD_VISION_MODEL", DEFAULT_VISION_MODEL),
                "text_model": os.getenv("AMD_TEXT_MODEL", DEFAULT_TEXT_MODEL)
            }
        else:
            return {
                "online": False,
                "status_code": resp.status_code,
                "error": resp.text[:200]
            }
    except Exception as e:
        return {
            "online": False,
            "status_code": 500,
            "error": str(e)
        }

def _post_with_retry(endpoint: str, payload: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
    """Execute POST request with exponential backoff on HTTP 429 rate limit."""
    headers = get_headers()
    url = f"{get_base_url()}/{endpoint.lstrip('/')}"
    delay = 2.0
    last_err = None

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=45)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                # Rate limit hit, backoff and retry
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    raise RuntimeError("AMD Token Factory rate limit (429) exceeded. Please try again shortly.")
            else:
                resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
            else:
                raise last_err

    raise RuntimeError(f"Request failed: {last_err}")

def diagnose_defect_vlm(
    image_b64: str,
    category: str,
    anomaly_score: float,
    verdict: str,
    regions: Optional[List[Dict[str, Any]]] = None,
    custom_notes: str = ""
) -> Dict[str, Any]:
    """
    Run Visual Language Model diagnosis on an inspected sample image.
    Uses DeepSeek-V4-Flash-Vision-Exp or MiniCPM-V.
    """
    # Normalize base64 data url if needed
    if not image_b64.startswith("data:image"):
        data_url = f"data:image/png;base64,{image_b64}"
    else:
        data_url = image_b64

    reg_summary = f"{len(regions)} localized flaw region(s)" if regions else "No distinct bounding boxes isolated"

    system_prompt = (
        "You are an industrial computer vision and quality control AI expert deployed on a factory line. "
        "Your task is to visually examine product images and localized anomaly heatmaps, then provide "
        "an actionable, clear quality engineering diagnosis."
    )

    user_prompt = f"""
Industrial Visual Inspection Context:
- Product Category: {category}
- Model Anomaly Score: {anomaly_score:.4f}
- Conformal Decision: {verdict}
- Detected Regions: {reg_summary}
{f"- Operator Note: {custom_notes}" if custom_notes else ""}

Carefully inspect the visual evidence in the image and respond with:
1. **Defect Description**: Specific visual flaw visible (e.g. surface scratch, contamination speck, structural crack, misaligned seal, hole, deformation).
2. **Defect Severity**: (CRITICAL / MAJOR / MINOR) with a 1-sentence justification.
3. **Probable Root Cause**: Likely manufacturing stage error (molding, conveyor friction, tooling wear, foreign particle ingress).
4. **Actionable Disposition**: (SCRAP / REWORK / PASS WITH CAUTION / ESCALATE).
"""

    model = os.getenv("AMD_VISION_MODEL", DEFAULT_VISION_MODEL)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 400
    }

    t0 = time.perf_counter()
    result = _post_with_retry("chat/completions", payload)
    latency_ms = (time.perf_counter() - t0) * 1000

    choice = result.get("choices", [{}])[0]
    message_content = choice.get("message", {}).get("content", "No analysis returned.")

    return {
        "model_used": model,
        "diagnosis": message_content,
        "latency_ms": round(latency_ms, 2),
        "usage": result.get("usage", {})
    }

def chat_operator_copilot(
    user_query: str,
    category: str,
    anomaly_score: Optional[float] = None,
    history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Operator Copilot assistant for the Human Review Queue.
    Answers quality tolerance questions, standard compliance, and calibration recommendations.
    """
    system_prompt = (
        f"You are the VisionInspect AI Operator Copilot, specialized in industrial anomaly inspection "
        f"for {category} components. You assist production line operators reviewing flagged samples. "
        f"Provide concise, practical manufacturing guidance adhering to ISO 9001 and Six Sigma defect standards."
    )

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for msg in history[-6:]:  # Keep recent context
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    context_prefix = ""
    if anomaly_score is not None:
        context_prefix = f"[Current sample category: {category}, anomaly score: {anomaly_score:.3f}]\n"

    messages.append({"role": "user", "content": f"{context_prefix}{user_query}"})

    model = os.getenv("AMD_TEXT_MODEL", DEFAULT_TEXT_MODEL)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 300
    }

    t0 = time.perf_counter()
    result = _post_with_retry("chat/completions", payload)
    latency_ms = (time.perf_counter() - t0) * 1000

    choice = result.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "No response.")

    return {
        "model_used": model,
        "reply": content,
        "latency_ms": round(latency_ms, 2)
    }

def generate_shift_qa_report(
    inspection_stats: Dict[str, Any],
    category: str,
    recent_verdicts: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Generate an ISO-9001 / Six Sigma Quality Assurance shift report using AMD Qwen3.8-27B.
    """
    model = os.getenv("AMD_REPORT_MODEL", DEFAULT_REPORT_MODEL)
    prompt = f"""
Generate an Executive Quality Assurance (QA) Shift Report for:
- Production Line: {category.title()} Inspection
- Total Inspected: {inspection_stats.get('total_inspected', 0)}
- Conforming (Normal): {inspection_stats.get('normal_count', 0)}
- Confirmed Defects: {inspection_stats.get('defect_count', 0)}
- Human Review Queue (Flagged): {inspection_stats.get('review_count', 0)}
- Mean Inspection Latency: {inspection_stats.get('total_latency_ms', 0) / max(1, inspection_stats.get('total_inspected', 1)):.2f} ms

Produce a structured markdown report including:
1. Executive Summary & Defect Escape Rate
2. Critical Quality Incidents Observed
3. Calibration & Preventive Maintenance Recommendations
4. Sign-off Recommendation (Release Batch / Quarantine Batch)
"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a Chief Quality Engineer preparing an official manufacturing audit log."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 600
    }

    t0 = time.perf_counter()
    result = _post_with_retry("chat/completions", payload)
    latency_ms = (time.perf_counter() - t0) * 1000

    choice = result.get("choices", [{}])[0]
    report_md = choice.get("message", {}).get("content", "Report generation failed.")

    return {
        "model_used": model,
        "report_markdown": report_md,
        "latency_ms": round(latency_ms, 2)
    }
