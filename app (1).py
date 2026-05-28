import os
import json
import base64
import re
import time
import streamlit as st

from openai import OpenAI
from google import genai


# =========================
# Page
# =========================
st.set_page_config(page_title="RetinaGPT", page_icon="👁️", layout="centered")

st.markdown(
    """
    <style>
      .block-container {
          padding-top: 2rem;
          padding-bottom: 2rem;
          max-width: 920px;
      }
      .big-title {
          font-size: 2.0rem;
          font-weight: 800;
          margin-bottom: 0.25rem;
      }
      .card {
          border: 1px solid rgba(0,0,0,0.08);
          border-radius: 14px;
          padding: 16px;
          background: white;
      }
      .divider {
          height: 1px;
          background: rgba(0,0,0,0.08);
          margin: 14px 0;
      }
      .pill {
          display: inline-block;
          padding: 4px 10px;
          border-radius: 999px;
          border: 1px solid rgba(0,0,0,0.10);
          font-size: 0.85rem;
      }
      .muted { color: #6b7280; }
      .subtle-box {
          border-left: 3px solid rgba(0,0,0,0.08);
          padding-left: 12px;
          margin-top: 8px;
      }
      .small-note { font-size: 0.9rem; color: #6b7280; }
      .debate-round {
          border-left: 3px solid #3b82f6;
          padding-left: 12px;
          margin: 8px 0;
          background: #f8faff;
          border-radius: 0 8px 8px 0;
          padding: 10px 12px;
      }
      .agree-badge {
          display: inline-block;
          padding: 2px 8px;
          border-radius: 999px;
          font-size: 0.8rem;
          font-weight: 600;
      }
      .agree-yes { background: #d1fae5; color: #065f46; }
      .agree-no  { background: #fee2e2; color: #991b1b; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">👁️ RetinaGPT</div>', unsafe_allow_html=True)
st.caption("Multi-agent deliberative retinal diagnostic reasoning.")


# =========================
# Keys
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY.")
    st.stop()

if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY.")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# =========================
# Models
# =========================
OPENAI_VISION_MODEL  = "gpt-4o"
OPENAI_ARBITER_MODEL = "gpt-4o-mini"
GEMINI_VISION_MODEL  = "gemini-2.5-flash"


# =========================
# JSON Schemas
# =========================
VISION_JSON_SCHEMA = {
    "top_diagnosis": "string",
    "top_differentials": ["string", "string", "string"],
    "key_evidence": ["string", "string", "string"],
    "confidence": "LOW|MODERATE|HIGH",
    "requested_additional_info": ["string", "string", "string"],
}

CRITIQUE_JSON_SCHEMA = {
    "agree": "true|false",
    "critique": "string — specific imaging or clinical reasoning for agreement/disagreement",
    "revised_diagnosis": "string or null if unchanged",
    "revised_confidence": "LOW|MODERATE|HIGH",
    "revision_type": "evidence_based|sycophantic|maintained",
}


# =========================
# Session state
# =========================
def ss_init():
    defaults = {
        "case_id": 1,
        "images": [],
        "final_report": "",
        "report_history": [],
        "agreement": None,
        "confidence_label": "",
        "confidence_icon": "",
        "chat_history": [],
        "analysis_done": False,
        "uploader_key": 1,
        "last_error_type": None,
        "last_error_raw": "",
        "additional_clinical_note": "",
        "add_uploader_key": 1,
        "additional_modality": "OCT",
        # structured clinical fields
        "age": None,
        "sex": "Select",
        "laterality": "Select",
        "visual_acuity": "",
        "primary_symptom": "",
        "duration": "",
        "relevant_history": "",
        "additional_notes": "",
        "clinical": "",
        # debate log — stores full transcript for research export
        "debate_log": None,
        # arm results for research comparison
        "arm_a_result": None,   # GPT-4o solo
        "arm_b_result": None,   # Gemini solo
        "arm_c_result": None,   # Parallel no-debate
        "arm_d_result": None,   # Full debate
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

ss_init()


def reset_case():
    st.session_state.case_id += 1
    for k in [
        "images", "final_report", "report_history", "agreement",
        "confidence_label", "confidence_icon", "chat_history",
        "last_error_type", "last_error_raw", "additional_clinical_note",
        "additional_modality", "clinical", "debate_log",
        "arm_a_result", "arm_b_result", "arm_c_result", "arm_d_result",
    ]:
        st.session_state[k] = [] if k in ("images", "report_history", "chat_history") else None
    st.session_state.analysis_done = False
    st.session_state.uploader_key += 1
    st.session_state.add_uploader_key += 1
    st.session_state.additional_modality = "OCT"
    st.session_state.confidence_label = ""
    st.session_state.confidence_icon = ""
    st.session_state.age = None
    st.session_state.sex = "Select"
    st.session_state.laterality = "Select"
    for f in ("visual_acuity","primary_symptom","duration","relevant_history","additional_notes","clinical","additional_clinical_note"):
        st.session_state[f] = ""


# =========================
# Helpers
# =========================
def b64_data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def safe_json_extract(text: str):
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def to_jsonable(obj):
    if obj is None: return None
    if isinstance(obj, (str, int, float, bool)): return obj
    if isinstance(obj, (bytes, bytearray)): return f"<bytes:{len(obj)}>"
    if isinstance(obj, dict): return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [to_jsonable(x) for x in obj]
    for attr in ("model_dump", "dict"):
        if hasattr(obj, attr):
            try: return to_jsonable(getattr(obj, attr)())
            except Exception: pass
    if hasattr(obj, "__dict__"):
        try: return to_jsonable(vars(obj))
        except Exception: pass
    return str(obj)


def normalize_dx(dx: str) -> str:
    """Normalize diagnosis string: remove parenthetical abbreviations, lowercase."""
    if not dx: return ""
    dx = re.sub(r"\([^)]{1,10}\)", "", dx)   # strip (DME), (RVO) etc.
    return re.sub(r"[^a-z0-9]+", " ", dx.lower()).strip()


def dx_similarity(a: str, b: str) -> bool:
    """True if two diagnosis strings are clinically equivalent."""
    a = normalize_dx(a)
    b = normalize_dx(b)
    if not a or not b: return False
    if a == b: return True
    if a in b or b in a: return True
    a_words = set(a.split()) - {"with","and","or","the","of","in","a","an","to","due","type"}
    b_words = set(b.split()) - {"with","and","or","the","of","in","a","an","to","due","type"}
    if len(a_words) >= 2 and len(b_words) >= 2:
        if len(a_words & b_words) >= min(2, min(len(a_words), len(b_words))):
            return True
    return False


def normalize_confidence(value: str) -> str:
    v = str(value or "").strip().upper()
    return v if v in {"HIGH", "MODERATE", "LOW"} else "LOW"


def overlap_top2(a: dict, b: dict) -> bool:
    """Check if two model outputs agree on the primary or top differential diagnosis."""
    a1 = normalize_dx((a or {}).get("top_diagnosis", ""))
    b1 = normalize_dx((b or {}).get("top_diagnosis", ""))
    if a1 and b1 and dx_similarity(a1, b1):
        return True
    a2 = [normalize_dx(x) for x in ((a or {}).get("top_differentials") or [])[:2]]
    b2 = [normalize_dx(x) for x in ((b or {}).get("top_differentials") or [])[:2]]
    # Check if either top dx appears in the other's differentials
    if any(dx_similarity(a1, x) for x in b2): return True
    if any(dx_similarity(b1, x) for x in a2): return True
    # Check differential overlap
    for x in a2:
        for y in b2:
            if dx_similarity(x, y): return True
    return False


def derive_confidence_badge(gem_js: dict, oa_js: dict, agreement: bool):
    gc = normalize_confidence((gem_js or {}).get("confidence", "LOW"))
    oc = normalize_confidence((oa_js  or {}).get("confidence", "LOW"))
    if agreement:
        if gc == "HIGH" and oc == "HIGH":
            return "High", "🟢"
        return "Moderate", "🟡"
    return "Low", "🔴"


def uploads_to_images(files):
    return [{"name": f.name, "mime": f.type or "image/jpeg", "data": f.getvalue()} for f in (files or [])]


def merge_images(old, new):
    return list(old or []) + list(new or [])


def get_current_assessment_label():
    n = len(st.session_state.report_history)
    return "Initial assessment" if n == 0 else f"Updated assessment {n}"


def build_clinical_summary():
    lines = []
    if st.session_state.age: lines.append(f"Age: {st.session_state.age}")
    if st.session_state.sex not in (None, "Select"): lines.append(f"Sex: {st.session_state.sex}")
    if st.session_state.laterality not in (None, "Select"): lines.append(f"Laterality: {st.session_state.laterality}")
    for label, key in [
        ("Visual acuity", "visual_acuity"),
        ("Primary symptom", "primary_symptom"),
        ("Duration", "duration"),
        ("Relevant history", "relevant_history"),
        ("Additional notes", "additional_notes"),
    ]:
        val = (st.session_state.get(key) or "").strip()
        if val: lines.append(f"{label}: {val}")
    return "\n".join(lines).strip()


# =========================
# ── ROUND 1: Independent vision calls ──
# =========================
def call_openai_vision(clinical_text: str, images):
    content = [{
        "type": "text",
        "text": (
            "You are an expert retina subspecialist. Carefully examine ALL provided retinal images "
            "(which may include fundus photography, OCT, FAF, FA, OCTA, ICGA, or wide-field imaging) "
            "together with the clinical information.\n\n"
            "CRITICAL RULES:\n"
            "1. You MUST always provide a top_diagnosis — never leave it empty or return 'UNCERTAIN'.\n"
            "2. If the diagnosis is genuinely uncertain, provide your BEST GUESS with LOW confidence "
            "and explain your reasoning in key_evidence.\n"
            "3. You MUST list at least 2 top_differentials — these must be DIFFERENT from top_diagnosis.\n"
            "4. You MUST list at least 2 key_evidence items describing what you see in the images.\n"
            "5. Describe specific morphological findings: fluid (SRF/IRF/CME), exudates, "
            "hemorrhage, drusen, RPE changes, ellipsoid zone status, neovascularization, etc.\n"
            "6. NEVER repeat top_diagnosis inside top_differentials.\n\n"
            "Return STRICT JSON ONLY — no markdown, no explanation — matching this schema exactly:\n"
            f"{json.dumps(VISION_JSON_SCHEMA, ensure_ascii=False)}\n\n"
            f"Clinical info:\n{clinical_text or '(none provided)'}"
        ),
    }]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": b64_data_url(img["mime"], img["data"])}})
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=1000,
        )
        raw = resp.choices[0].message.content or ""
        parsed = safe_json_extract(raw)
        # Clean up: remove top_diagnosis from differentials if present
        if parsed and parsed.get("top_differentials"):
            top_dx = normalize_dx(parsed.get("top_diagnosis",""))
            parsed["top_differentials"] = [
                d for d in parsed["top_differentials"]
                if not dx_similarity(normalize_dx(d), top_dx)
            ]
        # Fallback: if model still returned empty/uncertain, retry once with stronger nudge
        if not parsed or parsed.get("top_diagnosis", "").upper() in ("UNCERTAIN", "", "UNKNOWN"):
            nudge_content = content.copy()
            nudge_content[0] = {
                "type": "text",
                "text": (
                    "You are an expert retina subspecialist. You MUST provide a specific diagnosis.\n"
                    "Look carefully at the images. Describe what you see: any fluid, exudates, "
                    "hemorrhage, drusen, membrane, vascular changes, or structural abnormalities.\n"
                    "Based on these findings, provide your best clinical impression.\n\n"
                    "Return STRICT JSON ONLY matching this schema:\n"
                    f"{json.dumps(VISION_JSON_SCHEMA, ensure_ascii=False)}\n\n"
                    f"Clinical info:\n{clinical_text or '(none provided)'}"
                ),
            }
            resp2 = openai_client.chat.completions.create(
                model=OPENAI_VISION_MODEL,
                messages=[{"role": "user", "content": nudge_content}],
                temperature=0.2,
                max_tokens=1000,
            )
            raw = resp2.choices[0].message.content or ""
            parsed = safe_json_extract(raw)
        return raw, parsed
    except Exception as e:
        return f"OpenAI vision error: {e}", None


def call_gemini_vision(clinical_text: str, images, max_retries: int = 3):
    parts = [{
        "text": (
            "You are a retina specialist. Analyze the provided retinal images and clinical info.\n"
            "Return STRICT JSON ONLY matching this schema:\n"
            f"{json.dumps(VISION_JSON_SCHEMA, ensure_ascii=False)}\n\n"
            f"Clinical info:\n{clinical_text or '(none provided)'}"
        )
    }]
    for img in images:
        parts.append({"inline_data": {"mime_type": img["mime"], "data": base64.b64encode(img["data"]).decode()}})

    last_error = ""
    for attempt in range(max_retries):
        try:
            resp = gemini_client.models.generate_content(
                model=GEMINI_VISION_MODEL,
                contents=[{"role": "user", "parts": parts}],
                config={"temperature": 0, "response_mime_type": "application/json"},
            )
            raw = getattr(resp, "text", None) or ""
            parsed = safe_json_extract(raw)
            if parsed:
                return raw, parsed, None
            last_error = raw or "Gemini returned empty or invalid JSON."
        except Exception as e:
            last_error = f"Gemini vision error: {e}"
            busy = any(x in last_error for x in ["503", "UNAVAILABLE", "high demand", "busy"])
            if busy and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return last_error, None, "busy" if busy else "generic"
    return last_error, None, "busy"


# =========================
# ── ROUND 2: Cross-critique ──
# =========================
def call_openai_critique(clinical_text: str, images, gemini_opinion: dict, oa_r1: dict = None):
    """GPT-4o reads Gemini's Round 1 output and critiques it."""
    own_context = (
        f"\nYour own Round 1 assessment was:\n{json.dumps(oa_r1, ensure_ascii=False, indent=2)}\n"
        if oa_r1 else ""
    )
    crit_content = [{
        "type": "text",
        "text": (
            "You are an expert retina subspecialist conducting a structured peer review.\n"
            "Re-examine ALL retinal images carefully, then evaluate your colleague's assessment.\n\n"
            f"Colleague's assessment:\n{json.dumps(gemini_opinion, ensure_ascii=False, indent=2)}\n"
            f"{own_context}\n"
            "TASK:\n"
            "1. Look at the images again with fresh eyes.\n"
            "2. Identify specific morphological features: fluid type (SRF/IRF/CME), exudate "
            "pattern, hemorrhage distribution, vascular changes, RPE status, drusen, etc.\n"
            "3. Decide whether you AGREE or DISAGREE with your colleague's top diagnosis.\n"
            "4. Write a specific evidence-based critique referencing actual image findings.\n"
            "5. If you revise your diagnosis, cite the EXACT imaging feature that changed your mind.\n"
            "6. Do NOT simply agree because your colleague seems confident — sycophancy is a bias.\n\n"
            "Return STRICT JSON ONLY matching this schema:\n"
            f"{json.dumps(CRITIQUE_JSON_SCHEMA, ensure_ascii=False)}\n\n"
            "revision_type rules:\n"
            "- 'evidence_based': diagnosis changes AND you cite a specific imaging finding\n"
            "- 'sycophantic': diagnosis changes WITHOUT citing a new specific imaging finding\n"
            "- 'maintained': you keep your view with explicit reasoning from the images\n\n"
            f"Clinical info:\n{clinical_text or '(none provided)'}"
        ),
    }]
    for img in images:
        crit_content.append({"type": "image_url", "image_url": {"url": b64_data_url(img["mime"], img["data"])}})
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[{"role": "user", "content": crit_content}],
            temperature=0,
            max_tokens=1000,
        )
        raw = resp.choices[0].message.content or ""
        return raw, safe_json_extract(raw)
    except Exception as e:
        return f"OpenAI critique error: {e}", None


def call_gemini_critique(clinical_text: str, images, openai_opinion: dict, gem_r1: dict = None, max_retries: int = 3):
    """Gemini reads GPT-4o's Round 1 output and critiques it."""
    own_context = (
        f"\nYour own Round 1 assessment was:\n{json.dumps(gem_r1, ensure_ascii=False, indent=2)}\n"
        if gem_r1 else ""
    )
    parts = [{
        "text": (
            "You are an expert retina subspecialist conducting a structured peer review.\n"
            "Re-examine ALL retinal images carefully, then evaluate your colleague's assessment.\n\n"
            f"Colleague's assessment:\n{json.dumps(openai_opinion, ensure_ascii=False, indent=2)}\n"
            f"{own_context}\n"
            "TASK:\n"
            "1. Look at the images again with fresh eyes.\n"
            "2. Identify specific morphological features: fluid type (SRF/IRF/CME), exudate "
            "pattern, hemorrhage distribution, vascular changes, RPE status, drusen, etc.\n"
            "3. Decide whether you AGREE or DISAGREE with your colleague's top diagnosis.\n"
            "4. Write a specific evidence-based critique referencing actual image findings.\n"
            "5. If you revise your diagnosis, cite the EXACT imaging feature that changed your mind.\n"
            "6. Do NOT simply agree because your colleague seems confident — sycophancy is a bias.\n\n"
            "Return STRICT JSON ONLY matching this schema:\n"
            f"{json.dumps(CRITIQUE_JSON_SCHEMA, ensure_ascii=False)}\n\n"
            "revision_type rules:\n"
            "- 'evidence_based': diagnosis changes AND you cite a specific imaging finding\n"
            "- 'sycophantic': diagnosis changes WITHOUT citing a new specific imaging finding\n"
            "- 'maintained': you keep your view with explicit reasoning from the images\n\n"
            f"Clinical info:\n{clinical_text or '(none provided)'}"
        )
    }]
    for img in images:
        parts.append({"inline_data": {"mime_type": img["mime"], "data": base64.b64encode(img["data"]).decode()}})

    last_error = ""
    for attempt in range(max_retries):
        try:
            resp = gemini_client.models.generate_content(
                model=GEMINI_VISION_MODEL,
                contents=[{"role": "user", "parts": parts}],
                config={"temperature": 0, "response_mime_type": "application/json"},
            )
            raw = getattr(resp, "text", None) or ""
            parsed = safe_json_extract(raw)
            if parsed:
                return raw, parsed, None
            last_error = raw or "Gemini critique returned empty."
        except Exception as e:
            last_error = f"Gemini critique error: {e}"
            busy = any(x in last_error for x in ["503", "UNAVAILABLE", "high demand", "busy"])
            if busy and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return last_error, None, "busy" if busy else "generic"
    return last_error, None, "busy"


# =========================
# ── ROUND 3: Revision ──
# =========================
def _revision_prompt(own_r1: dict, critique_received: dict, clinical_text: str) -> str:
    return (
        "You are a retina specialist. Below is your original assessment (Round 1) "
        "and the peer critique you received (Round 2).\n\n"
        f"Your Round 1 assessment:\n{json.dumps(own_r1, ensure_ascii=False, indent=2)}\n\n"
        f"Peer critique:\n{json.dumps(critique_received, ensure_ascii=False, indent=2)}\n\n"
        "Review the images again and produce your FINAL revised assessment.\n"
        "If you change your diagnosis, explain exactly which imaging or clinical finding prompted the change.\n"
        "If you maintain your original diagnosis, explain why the critique did not change your view.\n\n"
        "Return STRICT JSON ONLY matching this schema:\n"
        f"{json.dumps(VISION_JSON_SCHEMA, ensure_ascii=False)}\n\n"
        f"Clinical info:\n{clinical_text or '(none provided)'}"
    )


def call_openai_revision(clinical_text: str, images, oa_r1: dict, gem_critique: dict):
    content = [{"type": "text", "text": _revision_prompt(oa_r1, gem_critique, clinical_text)}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": b64_data_url(img["mime"], img["data"])}})
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        return raw, safe_json_extract(raw)
    except Exception as e:
        return f"OpenAI revision error: {e}", None


def call_gemini_revision(clinical_text: str, images, gem_r1: dict, oa_critique: dict, max_retries: int = 3):
    parts = [{"text": _revision_prompt(gem_r1, oa_critique, clinical_text)}]
    for img in images:
        parts.append({"inline_data": {"mime_type": img["mime"], "data": base64.b64encode(img["data"]).decode()}})

    last_error = ""
    for attempt in range(max_retries):
        try:
            resp = gemini_client.models.generate_content(
                model=GEMINI_VISION_MODEL,
                contents=[{"role": "user", "parts": parts}],
                config={"temperature": 0, "response_mime_type": "application/json"},
            )
            raw = getattr(resp, "text", None) or ""
            parsed = safe_json_extract(raw)
            if parsed:
                return raw, parsed, None
            last_error = raw or "Gemini revision returned empty."
        except Exception as e:
            last_error = f"Gemini revision error: {e}"
            busy = any(x in last_error for x in ["503", "UNAVAILABLE", "high demand", "busy"])
            if busy and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return last_error, None, "busy" if busy else "generic"
    return last_error, None, "busy"


# =========================
# ── SYNTHESIS: Arbiter report ──
# =========================
def build_final_report(clinical_text: str, debate_transcript: dict, is_update: bool):
    payload = to_jsonable({
        "clinical": clinical_text,
        "debate_transcript": debate_transcript,
        "is_update": is_update,
    })

    system = """
You are RetinaGPT Arbiter, a retina subspecialty clinical reasoning assistant.
You receive the FULL debate transcript between two AI retina specialists (3 rounds).
Generate ONE final educational report synthesizing the debate.

FORMAT (Markdown)

**Most likely diagnosis**
- State the most probable diagnosis in **bold**
- Add confidence level (High / Moderate / Low)
- State whether the two specialists reached consensus or maintained disagreement after debate
- Brief justification referencing key debate arguments

**Key imaging findings**
- Bullet list of the most relevant retinal imaging features discussed during debate
- Focus on morphology (RPE changes, ellipsoid zone status, SRF, IRF, PED, hemorrhage, vascular changes)

**Differential diagnosis**
- 2-4 most relevant alternatives
- Note which were debated and why each was accepted or rejected

**Management considerations**
- Evidence-based retina management
- Mention observation vs treatment when relevant
- Include follow-up strategy

**Suggested additional imaging**
- Recommend only modalities NOT already present in the case
- Examples: OCTA, FA, ICGA, FAF, wide-field imaging

**Debate summary**
- 2-3 sentences: what was agreed in Round 1, what was contested, how Round 3 resolved it
- Note revision types: evidence-based revision, maintained position, or sycophantic change

STYLE
- Retina subspecialty terminology
- Concise but clinically meaningful
- Morphology-first reasoning
- Do NOT mention model names (GPT, Gemini etc.)
- If this is an updated assessment, explicitly note how new imaging changed the working diagnosis
"""
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_ARBITER_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"Final report generation failed: {e}"


# =========================
# Chat continuation
# =========================
def chat_reply(user_text: str):
    context_pack = to_jsonable({
        "case": st.session_state.case_id,
        "clinical": st.session_state.clinical,
        "final_report": st.session_state.final_report,
        "report_history": st.session_state.report_history,
        "confidence_label": st.session_state.confidence_label,
    })
    system = """
You are RetinaGPT (educational). Continue discussion for the SAME case.
- Use the latest final report as the baseline.
- Use the structured clinical information as part of the case context.
- If the user asks what to upload next, suggest the single most useful modality and what to look for.
- Keep outputs concise and structured.
- Use retina subspecialty terminology.
"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "CASE CONTEXT:\n" + json.dumps(context_pack, ensure_ascii=False, indent=2)},
    ]
    for m in st.session_state.chat_history[-20:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_text})
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_ARBITER_MODEL, messages=messages, temperature=0,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"Chat reply failed: {e}"


# =========================
# ── MAIN ANALYSIS PIPELINE ──
# =========================
def run_analysis():
    images = st.session_state.images
    clin   = st.session_state.clinical

    st.session_state.last_error_type = None
    st.session_state.last_error_raw  = ""

    # ── Round 1: Independent ──────────────────────────────────────────────
    with st.spinner("🔬 Round 1 — Independent analysis..."):
        gem_raw, gem_r1, gem_err = call_gemini_vision(clin, images)
        oa_raw,  oa_r1           = call_openai_vision(clin, images)

    if not gem_r1:
        st.session_state.last_error_type = gem_err or "generic"
        st.session_state.last_error_raw  = gem_raw or ""
        st.session_state.analysis_done   = False
        return

    if not oa_r1:
        oa_r1 = {"top_diagnosis": "UNCERTAIN", "top_differentials": [],
                 "key_evidence": [], "confidence": "LOW", "requested_additional_info": []}

    # Save arm A and B (solo results)
    st.session_state.arm_a_result = oa_r1
    st.session_state.arm_b_result = gem_r1

    # Arm C: parallel no-debate agreement
    agree_c = overlap_top2(gem_r1, oa_r1)
    st.session_state.arm_c_result = {
        "openai_r1": oa_r1, "gemini_r1": gem_r1, "agreement": agree_c
    }

    # ── Round 2: Cross-critique ───────────────────────────────────────────
    with st.spinner("💬 Round 2 — Cross-critique..."):
        oa_crit_raw, oa_crit = call_openai_critique(clin, images, gem_r1, oa_r1=oa_r1)
        gem_crit_raw, gem_crit, _ = call_gemini_critique(clin, images, oa_r1, gem_r1=gem_r1)

    if not oa_crit:
        oa_crit = {"agree": True, "critique": "No critique available.",
                   "revised_diagnosis": None, "revised_confidence": oa_r1.get("confidence","LOW"),
                   "revision_type": "maintained"}
    if not gem_crit:
        gem_crit = {"agree": True, "critique": "No critique available.",
                    "revised_diagnosis": None, "revised_confidence": gem_r1.get("confidence","LOW"),
                    "revision_type": "maintained"}

    # ── Round 3: Revision ─────────────────────────────────────────────────
    with st.spinner("🔄 Round 3 — Revision..."):
        oa_rev_raw, oa_r3  = call_openai_revision(clin, images, oa_r1, gem_crit)
        gem_rev_raw, gem_r3, _ = call_gemini_revision(clin, images, gem_r1, oa_crit)

    if not oa_r3:  oa_r3  = oa_r1
    if not gem_r3: gem_r3 = gem_r1

    # ── Final agreement after debate ──────────────────────────────────────
    agree_d = overlap_top2(gem_r3, oa_r3)
    confidence_label, confidence_icon = derive_confidence_badge(gem_r3, oa_r3, agree_d)

    # ── Build debate transcript for arbiter ───────────────────────────────
    debate_transcript = {
        "round_1": {"openai": oa_r1,   "gemini": gem_r1},
        "round_2": {"openai_critique_of_gemini": oa_crit,
                    "gemini_critique_of_openai": gem_crit},
        "round_3": {"openai_final": oa_r3, "gemini_final": gem_r3},
        "convergence": {
            "agreed_r1": overlap_top2(gem_r1, oa_r1),
            "agreed_r3": agree_d,
            "debate_changed_outcome": overlap_top2(gem_r1, oa_r1) != agree_d,
        },
    }

    # ── Synthesis ─────────────────────────────────────────────────────────
    with st.spinner("📝 Synthesizing final report..."):
        is_update = len(st.session_state.report_history) > 0
        report = build_final_report(clin, debate_transcript, is_update)

    # ── Store arm D ───────────────────────────────────────────────────────
    st.session_state.arm_d_result = {
        "debate_transcript": debate_transcript,
        "final_report": report,
        "agreement": agree_d,
        "confidence_label": confidence_label,
    }

    # ── Update session state ──────────────────────────────────────────────
    label = get_current_assessment_label()
    st.session_state.debate_log       = debate_transcript
    st.session_state.final_report     = report
    st.session_state.agreement        = agree_d
    st.session_state.confidence_label = confidence_label
    st.session_state.confidence_icon  = confidence_icon
    st.session_state.analysis_done    = True
    st.session_state.report_history.append({
        "label": label,
        "report": report,
        "confidence_label": confidence_label,
        "confidence_icon": confidence_icon,
        "debate_log": debate_transcript,
    })


# =========================
# ── UI HELPERS ──
# =========================
def render_debate_expander(debate_log: dict):
    """Renders the debate transcript in a readable expander."""
    if not debate_log:
        return
    with st.expander("🧠 Debate transcript", expanded=False):
        r1 = debate_log.get("round_1", {})
        r2 = debate_log.get("round_2", {})
        r3 = debate_log.get("round_3", {})
        cv = debate_log.get("convergence", {})

        # Convergence badge
        agreed_r1 = cv.get("agreed_r1", False)
        agreed_r3 = cv.get("agreed_r3", False)
        changed   = cv.get("debate_changed_outcome", False)

        col1, col2, col3 = st.columns(3)
        with col1:
            badge1 = "agree-yes" if agreed_r1 else "agree-no"
            label1 = "✅ Agreed R1" if agreed_r1 else "❌ Disagreed R1"
            st.markdown(f'<span class="agree-badge {badge1}">{label1}</span>', unsafe_allow_html=True)
        with col2:
            badge3 = "agree-yes" if agreed_r3 else "agree-no"
            label3 = "✅ Agreed R3" if agreed_r3 else "❌ Disagreed R3"
            st.markdown(f'<span class="agree-badge {badge3}">{label3}</span>', unsafe_allow_html=True)
        with col3:
            if changed:
                st.markdown('<span class="agree-badge" style="background:#fef3c7;color:#92400e;">🔄 Debate changed outcome</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="agree-badge" style="background:#f3f4f6;color:#374151;">= No change from debate</span>', unsafe_allow_html=True)

        st.markdown("---")

        # Round 1
        st.markdown("**Round 1 — Independent analysis**")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Model A (GPT-4o)")
            oa = r1.get("openai", {})
            st.markdown(f"🔹 **{oa.get('top_diagnosis','—')}** ({oa.get('confidence','—')})")
            for ev in (oa.get("key_evidence") or []):
                st.markdown(f"- {ev}")
        with c2:
            st.caption("Model B (Gemini)")
            gm = r1.get("gemini", {})
            st.markdown(f"🔹 **{gm.get('top_diagnosis','—')}** ({gm.get('confidence','—')})")
            for ev in (gm.get("key_evidence") or []):
                st.markdown(f"- {ev}")

        st.markdown("---")

        # Round 2
        st.markdown("**Round 2 — Cross-critique**")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Model A critiques Model B")
            oa_c = r2.get("openai_critique_of_gemini", {})
            agree_icon = "✅" if oa_c.get("agree") else "❌"
            st.markdown(f"{agree_icon} **Agrees:** {oa_c.get('agree','—')}")
            st.markdown(f"_{oa_c.get('critique','—')}_")
            if oa_c.get("revised_diagnosis"):
                st.markdown(f"🔄 Revised to: **{oa_c['revised_diagnosis']}**")
            rtype = oa_c.get("revision_type","—")
            rcolor = {"evidence_based":"🟢","sycophantic":"🔴","maintained":"🔵"}.get(rtype,"⚪")
            st.markdown(f"{rcolor} {rtype}")
        with c2:
            st.caption("Model B critiques Model A")
            gm_c = r2.get("gemini_critique_of_openai", {})
            agree_icon = "✅" if gm_c.get("agree") else "❌"
            st.markdown(f"{agree_icon} **Agrees:** {gm_c.get('agree','—')}")
            st.markdown(f"_{gm_c.get('critique','—')}_")
            if gm_c.get("revised_diagnosis"):
                st.markdown(f"🔄 Revised to: **{gm_c['revised_diagnosis']}**")
            rtype = gm_c.get("revision_type","—")
            rcolor = {"evidence_based":"🟢","sycophantic":"🔴","maintained":"🔵"}.get(rtype,"⚪")
            st.markdown(f"{rcolor} {rtype}")

        st.markdown("---")

        # Round 3
        st.markdown("**Round 3 — Final positions**")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Model A final")
            oa3 = r3.get("openai_final", {})
            st.markdown(f"🔹 **{oa3.get('top_diagnosis','—')}** ({oa3.get('confidence','—')})")
        with c2:
            st.caption("Model B final")
            gm3 = r3.get("gemini_final", {})
            st.markdown(f"🔹 **{gm3.get('top_diagnosis','—')}** ({gm3.get('confidence','—')})")


# =========================
# ── UI: Clinical input ──
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Clinical information")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age *", min_value=0, max_value=120,
                          value=st.session_state.age or 0, step=1)
with col2:
    sex_opts = ["Select", "Male", "Female", "Other"]
    sex = st.selectbox("Sex *", sex_opts,
                       index=sex_opts.index(st.session_state.sex if st.session_state.sex in sex_opts else "Select"))
with col3:
    lat_opts = ["Select", "OD (Right)", "OS (Left)", "OU (Both)"]
    laterality_map = {"Select":"Select","OD (Right)":"OD (Right)","OS (Left)":"OS (Left)","OU (Both)":"OU (Both)","Unilateral":"OD (Right)","Bilateral":"OU (Both)"}
    cur_lat = laterality_map.get(st.session_state.laterality, "Select")
    laterality = st.selectbox("Laterality *", lat_opts,
                               index=lat_opts.index(cur_lat if cur_lat in lat_opts else "Select"))

col4, col5 = st.columns(2)
with col4:
    visual_acuity = st.text_input("Visual acuity (optional)", value=st.session_state.visual_acuity,
                                   placeholder="e.g. 20/40, 0.3, CF at 2 m")
with col5:
    primary_symptom = st.text_input("Primary symptom (optional)", value=st.session_state.primary_symptom,
                                     placeholder="e.g. blurred vision, metamorphopsia, asymptomatic")

col6, col7 = st.columns(2)
with col6:
    duration = st.text_input("Duration (optional)", value=st.session_state.duration,
                              placeholder="e.g. acute, 3 days, chronic, 2 months")
with col7:
    relevant_history = st.text_input("Relevant history (optional)", value=st.session_state.relevant_history,
                                      placeholder="e.g. diabetes, high myopia, recent viral illness")

additional_notes = st.text_area("Additional notes (optional)", value=st.session_state.additional_notes,
                                  placeholder="Any extra clinical context...", height=90)

# Update session state
st.session_state.age              = int(age)
st.session_state.sex              = sex
st.session_state.laterality       = laterality
st.session_state.visual_acuity    = visual_acuity or ""
st.session_state.primary_symptom  = primary_symptom or ""
st.session_state.duration         = duration or ""
st.session_state.relevant_history = relevant_history or ""
st.session_state.additional_notes = additional_notes or ""

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("Retinal images")

uploads = st.file_uploader(
    "Upload retinal images (OCT, fundus, FAF, FA, OCTA, ICGA, wide-field)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
)

if uploads:
    cols = st.columns(2)
    for i, f in enumerate(uploads[:2]):
        with cols[i % 2]:
            st.image(f, caption=f.name, use_container_width=True)
    if len(uploads) > 2:
        st.caption(f"+ {len(uploads) - 2} more image(s) selected.")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<span class="small-note">* Required fields: Age, Sex, Laterality</span>', unsafe_allow_html=True)

analyze = st.button("🔎 Analyze (3-Round Debate)", type="primary", use_container_width=True)

if analyze:
    missing = []
    if sex == "Select":       missing.append("Sex")
    if laterality == "Select": missing.append("Laterality")
    if missing:
        st.error("Please complete required fields: " + ", ".join(missing))
    elif not uploads:
        st.error("Please upload at least one image.")
    else:
        st.session_state.clinical = build_clinical_summary()
        st.session_state.images   = uploads_to_images(uploads)
        run_analysis()
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)


# =========================
# ── UI: Error states ──
# =========================
if st.session_state.last_error_type:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    msg = "System is busy at the moment. Please retry in a few seconds." \
          if st.session_state.last_error_type == "busy" \
          else "Analysis could not be completed. Please retry."
    st.warning(msg)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔁 Retry analysis", use_container_width=True):
            if st.session_state.images:
                run_analysis(); st.rerun()
    with c2:
        if st.button("🆕 Start new case", use_container_width=True):
            reset_case(); st.rerun()
    with st.expander("Technical details"):
        st.code(st.session_state.last_error_raw or "No technical details available.")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# ── UI: Report output ──
# =========================
if st.session_state.report_history:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    latest_badge = f"{st.session_state.confidence_icon} Diagnostic confidence: {st.session_state.confidence_label}"
    st.markdown(
        f'<span class="pill">{latest_badge}</span>'
        f' <span class="pill muted">Case #{st.session_state.case_id}</span>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    for idx, item in enumerate(st.session_state.report_history):
        item_badge = f"{item.get('confidence_icon','🟡')} Diagnostic confidence: {item.get('confidence_label','Moderate')}"
        st.markdown(f"### {item['label']}")
        st.markdown(f'<span class="pill muted">{item_badge}</span>', unsafe_allow_html=True)

        # Debate transcript expander
        if item.get("debate_log"):
            render_debate_expander(item["debate_log"])

        st.markdown('<div class="subtle-box">', unsafe_allow_html=True)
        st.markdown(item["report"])
        st.markdown("</div>", unsafe_allow_html=True)

        if idx < len(st.session_state.report_history) - 1:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Research data export
    if st.session_state.debate_log:
        export_data = {
            "case_id": st.session_state.case_id,
            "clinical": st.session_state.clinical,
            "arm_a_gpt4o_solo": to_jsonable(st.session_state.arm_a_result),
            "arm_b_gemini_solo": to_jsonable(st.session_state.arm_b_result),
            "arm_c_parallel_no_debate": to_jsonable(st.session_state.arm_c_result),
            "arm_d_debate_full": to_jsonable(st.session_state.arm_d_result),
        }
        st.download_button(
            label="📥 Export research data (JSON)",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name=f"retinagpt_case_{st.session_state.case_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    if st.button("🆕 New patient", use_container_width=True):
        reset_case(); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# ── UI: Add more images ──
# =========================
if st.session_state.report_history:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("➕ Add additional imaging")
    st.caption("Upload additional multimodal imaging for the same case and append an updated assessment below.")

    modality_options = ["OCT","Fundus photo","FAF","FA","OCTA","ICGA","Wide-field imaging","Other"]
    cur_mod = st.session_state.additional_modality
    extra_modality = st.selectbox("Imaging modality", modality_options,
                                   index=modality_options.index(cur_mod if cur_mod in modality_options else "Other"),
                                   key="extra_modality_box")
    st.session_state.additional_modality = extra_modality

    extra_note = st.text_area("Additional clinical note (optional)",
                               placeholder="e.g. OCT shows type 1 MNV, FA demonstrates leakage...",
                               value=st.session_state.additional_clinical_note,
                               key="additional_note_box", height=90)
    st.session_state.additional_clinical_note = extra_note or ""

    extra_uploads = st.file_uploader("Upload additional retinal images",
                                      type=["jpg","jpeg","png","webp"],
                                      accept_multiple_files=True,
                                      key=f"add_uploader_{st.session_state.add_uploader_key}")

    if extra_uploads:
        cols = st.columns(2)
        for i, f in enumerate(extra_uploads[:2]):
            with cols[i % 2]:
                st.image(f, caption=f.name, use_container_width=True)
        if len(extra_uploads) > 2:
            st.caption(f"+ {len(extra_uploads) - 2} more additional image(s).")

    if st.button("🔄 Update diagnosis", use_container_width=True):
        if not extra_uploads and not st.session_state.additional_clinical_note.strip():
            st.warning("Please upload at least one additional image or add a clinical note.")
        else:
            if extra_uploads:
                st.session_state.images = merge_images(
                    st.session_state.images, uploads_to_images(extra_uploads)
                )
            update_note = f"Newly added imaging modality: {st.session_state.additional_modality}"
            if st.session_state.additional_clinical_note.strip():
                update_note += "\n" + st.session_state.additional_clinical_note.strip()
            if st.session_state.clinical.strip():
                st.session_state.clinical = st.session_state.clinical.strip() + "\n\nAdditional update:\n" + update_note
            else:
                st.session_state.clinical = update_note

            st.session_state.additional_clinical_note = ""
            st.session_state.additional_modality      = "OCT"
            st.session_state.add_uploader_key        += 1
            run_analysis()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# ── UI: Follow-up chat ──
# =========================
if st.session_state.report_history:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 Follow-up chat")

    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_msg = st.chat_input("Ask a follow-up question…")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ans = chat_reply(user_msg)
                st.write(ans)
        st.session_state.chat_history.append({"role": "assistant", "content": ans})

    st.markdown("</div>", unsafe_allow_html=True)
