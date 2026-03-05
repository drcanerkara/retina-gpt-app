import os
import json
import base64
from typing import Any, Dict, Optional

import streamlit as st
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
APP_TITLE = "RetinaGPT (Basic Core)"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

st.set_page_config(page_title=APP_TITLE, page_icon="👁️", layout="wide")
st.title("👁️ RetinaGPT — Basic Core")
st.caption("Vision-first (morphology → differential). Educational only. No treatment advice.")


# -----------------------------
# API KEY
# -----------------------------
def get_api_key() -> Optional[str]:
    try:
        k = st.secrets.get("OPENAI_API_KEY")
        if k:
            return k
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")

api_key = get_api_key()
if not api_key:
    st.error("OPENAI_API_KEY missing. Add to Streamlit secrets or env.")
    st.stop()

client = OpenAI(api_key=api_key)


# -----------------------------
# PROMPTS
# -----------------------------
MORPHOLOGY_SYSTEM = """
You are a retina imaging morphology extractor for DE-IDENTIFIED ophthalmic images (NOT faces/people).
Do NOT diagnose. Do NOT suggest management.
Return STRICT JSON only.

Goal: describe what is VISIBLE, focusing on global + macular + vascular + pigmentation patterns.

JSON schema:
{
  "modality_guess": ["Fundus"|"OCT"|"FAF"|"FA"|"ICGA"|"OCTA"|"UNCERTAIN"],
  "image_quality": "GOOD|FAIR|POOR",
  "global_pigmentation": "NORMAL|HYPOPIGMENTED|HYPERPIGMENTED|UNCERTAIN",
  "choroidal_vessels_visibility": "PROMINENT|NORMAL|NOT_SEEN|UNCERTAIN",
  "foveal_reflex_or_definition": "NORMAL|REDUCED|ABSENT|UNCERTAIN",
  "macular_abnormality": ["NONE","RPE_CHANGE","ATROPHY","PIGMENT_MOTTLE","GRAYING","ELEVATED_LESION","UNCERTAIN"],
  "vascular_findings": ["NONE","TORTUOSITY","ATTENUATION","SHEATHING","TELANGIECTASIA","AV_NICKING","UNCERTAIN"],
  "hemorrhage_exudates": "PRESENT|ABSENT|UNCERTAIN",
  "notes": ["short bullet notes, max 8"]
}
"""

DIAG_SYSTEM = """
You are RetinaGPT (educational). You receive:
- clinical text
- morphology JSON extracted from an image

You MUST:
1) Use morphology as primary evidence.
2) Provide an EDUCATIONAL ranked differential (no patient-specific treatment).
3) Return TWO blocks:
(A) STRICT JSON inside ```json ... ```
(B) Human report with headings.

Hard trigger rules:
- If morphology shows HYPOPIGMENTED + PROMINENT choroidal vessels AND clinical mentions nystagmus or lifelong low vision,
  include "Albinism spectrum / Ocular albinism" in TOP differential.
- If morphology notes a solitary torpedo/teardrop-shaped hypopigmented lesion temporal to fovea (if mentioned in notes),
  include "Torpedo maculopathy" in differential.

JSON schema for block (A):
{
  "case_summary": {"age": null, "sex": null, "symptoms": null, "duration": null, "laterality": null, "history": null},
  "modalities_detected": [],
  "missing_modalities_suggested": [],
  "patterns": [{"name":"", "confidence":0.0}],
  "feature_checklist": {
    "subretinal_fluid":"PRESENT|ABSENT|UNCERTAIN",
    "intraretinal_fluid":"PRESENT|ABSENT|UNCERTAIN",
    "hemorrhage_exudation":"PRESENT|ABSENT|UNCERTAIN",
    "inner_retinal_ischemia":"PRESENT|ABSENT|UNCERTAIN",
    "ez_disruption":"PRESENT|ABSENT|UNCERTAIN",
    "rpe_atrophy_hypertransmission":"PRESENT|ABSENT|UNCERTAIN",
    "vitelliform_material":"PRESENT|ABSENT|UNCERTAIN",
    "inflammatory_signs":"PRESENT|ABSENT|UNCERTAIN",
    "m_nv_suspected":"PRESENT|ABSENT|UNCERTAIN",
    "true_mass_lesion_suspected":"PRESENT|ABSENT|UNCERTAIN"
  },
  "most_likely": {"diagnosis":"", "probability":0.0},
  "differential": [{"diagnosis":"", "probability":0.0, "for":[], "against":[]}],
  "confidence_level":"LOW|MODERATE|HIGH",
  "next_best_requests":[{"request":"", "why":""}],
  "urgency":"CRITICAL|URGENT|ROUTINE"
}

Human report headings:
1) Clinical Triage
2) Morphology Summary (from extractor)
3) Detected Modalities + Missing Modalities
4) Detected Pattern(s)
5) Integrated Discussion
6) Most Likely + Differential (ranked)
7) Confidence + Next tests (educational)
8) Educational limitations statement
"""


# -----------------------------
# RESPONSES API HELPERS
# -----------------------------
def to_data_url(file_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def responses_text(resp) -> str:
    # SDK convenience: output_text exists on responses
    try:
        return resp.output_text or ""
    except Exception:
        return ""

def safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None

def extract_json_fence(text: str) -> Optional[Dict[str, Any]]:
    if "```json" not in text:
        return None
    try:
        s = text.index("```json") + len("```json")
        e = text.index("```", s)
        block = text[s:e].strip()
        return json.loads(block)
    except Exception:
        return None


# -----------------------------
# UI
# -----------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    clinical = st.text_area(
        "Clinical text (age/sex/symptoms/duration/laterality/history)",
        height=160,
        placeholder="Example:\nAge: 37\nSex: Female\nSymptoms: lifelong low vision + nystagmus\nDuration: chronic\nLaterality: bilateral\nHistory: congenital"
    )
    img = st.file_uploader("Upload ONE image (fundus/OCT/FAF)", type=["jpg","jpeg","png","webp"])

    if img:
        st.image(img, caption="Uploaded image", use_container_width=True)

    run = st.button("🔎 Analyze", type="primary")

with col2:
    st.subheader("Debug")
    show_debug = st.checkbox("Show morphology JSON (debug)", value=True)


# -----------------------------
# RUN
# -----------------------------
if run:
    if not img:
        st.error("Please upload an image.")
        st.stop()

    file_bytes = img.read()
    mime = img.type or "image/jpeg"
    data_url = to_data_url(file_bytes, mime)

    # 1) Morphology extraction (STRICT JSON)
    with st.spinner("Step 1/2: Extracting morphology..."):
        resp1 = client.responses.create(
            model=MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": MORPHOLOGY_SYSTEM},
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                ]
            }],
        )
        morph_text = responses_text(resp1).strip()
        morph = safe_json_load(morph_text)

    if morph is None:
        st.error("Morphology extractor did not return valid JSON. Try again with a clearer image.")
        st.text(morph_text[:2000])
        st.stop()

    if show_debug:
        st.subheader("Morphology JSON (debug)")
        st.json(morph)

    # 2) Diagnosis & report (JSON fenced + report)
    with st.spinner("Step 2/2: Generating educational differential..."):
        user_payload = {
            "clinical_text": clinical.strip() if clinical.strip() else "(not provided)",
            "morphology_json": morph
        }

        resp2 = client.responses.create(
            model=MODEL,
            input=[{
                "role": "system",
                "content": [{"type": "input_text", "text": DIAG_SYSTEM}]
            },{
                "role": "user",
                "content": [{"type": "input_text", "text": json.dumps(user_payload, ensure_ascii=False)}]
            }],
        )

        out = responses_text(resp2)

    st.subheader("Assistant output")
    st.markdown(out)

    parsed = extract_json_fence(out)
    if parsed:
        st.subheader("Structured output (debug)")
        st.json(parsed)
    else:
        st.warning("No parsable ```json block found in the output.")
