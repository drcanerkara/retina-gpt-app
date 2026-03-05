import os
import json
import base64
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI
from google import genai  # pip install google-genai

st.set_page_config(page_title="RetinaGPT Dual (OpenAI+Gemini)", page_icon="👁️", layout="wide")
st.title("👁️ RetinaGPT — Dual Opinion (OpenAI + Gemini) + Consensus")
st.caption("Educational only. De-identified retinal images (not faces).")

# -----------------------------
# Keys
# -----------------------------
def get_secret(name: str) -> Optional[str]:
    try:
        v = st.secrets.get(name)
        if v:
            return v
    except Exception:
        pass
    return os.getenv(name)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing (Streamlit secrets or env var).")
    st.stop()

if not GEMINI_API_KEY:
    st.warning("Gemini key yok. Yalnız OpenAI ile çalışır. (GEMINI_API_KEY / GOOGLE_API_KEY ekle)")
    # we won't stop; we'll run single-provider mode

openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# -----------------------------
# Prompts
# -----------------------------
JSON_SCHEMA_PROMPT = """
You are a retina subspecialist. Analyze DE-IDENTIFIED ophthalmic images (retina/OCT/FAF/FA/OCTA).
These are NOT faces/people. Do NOT identify any person.
Educational differential only. No treatment instructions.

Return STRICT JSON ONLY (no markdown, no extra text) with schema:
{
  "modalities_detected": ["Fundus","OCT","FAF","FA","ICGA","OCTA"],
  "image_quality": "GOOD|FAIR|POOR",
  "key_findings": ["8-15 short bullets, morphology + location + distribution"],
  "pattern_labels": [{"name":"string","confidence":0.0}],
  "most_likely": {"diagnosis":"string","confidence":"LOW|MODERATE|HIGH"},
  "differential": [{"diagnosis":"string","weight":"HIGH|MEDIUM|LOW","for":["..."],"against":["..."]}],
  "uncertainties": ["..."],
  "next_best_data": [{"item":"string","why":"string"}]
}
"""

CONSENSUS_PROMPT = """
You are a consensus synthesizer (educational). You will be given:
- clinical metadata
- OpenAI JSON opinion
- Gemini JSON opinion (may be null)

Goal:
- If they agree: produce a provisional consensus impression.
- If they disagree: explain why (pattern-level) and propose the MINIMUM next-best-data to resolve.
- Output STRICT JSON ONLY:
{
  "agreement_level": "HIGH|PARTIAL|LOW",
  "consensus": {"diagnosis":"string","confidence":"LOW|MODERATE|HIGH"},
  "where_they_agree": ["..."],
  "where_they_disagree": ["..."],
  "next_best_requests": [{"request":"string","why":"string"}],
  "final_ranked_differential": [{"diagnosis":"string","probability":0.0,"for":["..."],"against":["..."]}],
  "note": "string"
}
"""

# -----------------------------
# Helpers
# -----------------------------
def to_data_url(upload) -> str:
    b = upload.getvalue()
    mime = upload.type or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}"

def safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        # try to salvage first {...}
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                return None
        return None

def call_openai_json(clinical: str, data_urls: List[str], model: str = "gpt-4o") -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type":"text","text": f"{JSON_SCHEMA_PROMPT}\n\nCLINICAL:\n{clinical}"}]
    for u in data_urls:
        content.append({"type":"image_url","image_url":{"url": u}})
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content": content}],
        temperature=0.0,
        max_tokens=1200,
    )
    txt = resp.choices[0].message.content or ""
    js = safe_json_loads(txt)
    if not js:
        return {"error":"openai_json_parse_failed", "raw": txt[:2000]}
    return js

def call_gemini_json(clinical: str, uploads, model: str) -> Dict[str, Any]:
    if not gemini_client:
        return {"error":"gemini_not_configured"}

    # python-genai supports multimodal parts
    parts = [f"{JSON_SCHEMA_PROMPT}\n\nCLINICAL:\n{clinical}"]
    for up in uploads:
        parts.append({"mime_type": up.type or "image/jpeg", "data": up.getvalue()})

    resp = gemini_client.models.generate_content(
        model=model,
        contents=parts
    )
    txt = (resp.text or "").strip()
    js = safe_json_loads(txt)
    if not js:
        return {"error":"gemini_json_parse_failed", "raw": txt[:2000]}
    return js

def consensus(openai_js: Dict[str, Any], gemini_js: Optional[Dict[str, Any]], clinical: str) -> Dict[str, Any]:
    payload = {
        "clinical": clinical,
        "openai": openai_js,
        "gemini": gemini_js
    }
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content": CONSENSUS_PROMPT},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
        ],
        temperature=0.0,
        max_tokens=900
    )
    txt = resp.choices[0].message.content or ""
    js = safe_json_loads(txt)
    if not js:
        return {"error":"consensus_parse_failed", "raw": txt[:2000]}
    return js

# -----------------------------
# UI
# -----------------------------
col1, col2 = st.columns([1,1], gap="large")
with col1:
    clinical = st.text_area("Clinical metadata", height=140, placeholder="Age/Sex/Symptoms/Duration/Laterality/History")
    uploads = st.file_uploader("Upload images (jpg/png/webp)", type=["jpg","jpeg","png","webp"], accept_multiple_files=True)

with col2:
    st.subheader("Models")
    openai_model = st.selectbox("OpenAI vision model", ["gpt-4o", "gpt-4o-mini"], index=0)
    gemini_model = st.text_input("Gemini model name", value="gemini-2.5-pro" if GEMINI_API_KEY else "")
    run = st.button("Run dual analysis", type="primary")

if run:
    if not uploads:
        st.error("At least 1 image required.")
        st.stop()

    data_urls = [to_data_url(u) for u in uploads]

    with st.spinner("OpenAI (GPT-4o) analyzing..."):
        openai_js = call_openai_json(clinical, data_urls, model=openai_model)
    st.subheader("OpenAI opinion (JSON)")
    st.json(openai_js)

    gemini_js = None
    if gemini_client and gemini_model.strip():
        with st.spinner("Gemini analyzing..."):
            gemini_js = call_gemini_json(clinical, uploads, model=gemini_model.strip())
        st.subheader("Gemini opinion (JSON)")
        st.json(gemini_js)
    else:
        st.info("Gemini step skipped (no key or model name).")

    with st.spinner("Building consensus..."):
        cons = consensus(openai_js, gemini_js, clinical)
    st.subheader("Consensus (JSON)")
    st.json(cons)

    st.markdown("### What to do next (quick view)")
    reqs = cons.get("next_best_requests") or []
    if reqs:
        for r in reqs[:6]:
            st.write(f"- **{r.get('request','')}** — {r.get('why','')}")
