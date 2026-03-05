import streamlit as st
import base64
import json
import os
import re
from openai import OpenAI
from google import genai
from google.genai import types

st.set_page_config(page_title="RetinaGPT v4", page_icon="👁️", layout="wide")
st.title("RetinaGPT v4")

# ----------------------------
# API KEYS
# ----------------------------
openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

if not openai_key:
    st.error("OPENAI_API_KEY not found (Streamlit secrets/environment).")
    st.stop()

openai_client = OpenAI(api_key=openai_key)

gemini_client = None
if gemini_key:
    try:
        gemini_client = genai.Client(api_key=gemini_key)
    except Exception as e:
        st.warning(f"Gemini client init failed: {str(e)}")
        gemini_client = None

# ----------------------------
# SIDEBAR SETTINGS
# ----------------------------
with st.sidebar:
    st.header("OpenAI")
    openai_model = st.text_input("OpenAI vision model", value="gpt-4o")
    openai_reason_model = st.text_input("OpenAI reasoning model", value="gpt-4o-mini")

    st.header("Gemini")
    gemini_model = st.text_input("Gemini model", value="gemini-1.5-flash")
    use_gemini = st.checkbox("Enable Gemini", value=True)

    if not gemini_key:
        st.info("GEMINI_API_KEY not set → Gemini disabled automatically.")

# ----------------------------
# INPUTS
# ----------------------------
clinical = st.text_area(
    "Clinical Information",
    placeholder="Age, sex, symptoms, duration, laterality, history..."
)

uploads = st.file_uploader(
    "Upload retinal images (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# run değişkenini baştan tanımlıyoruz → NameError önlemek için
run = False
run = st.button("Analyze", type="primary")

# ----------------------------
# HELPERS
# ----------------------------
def encode_image_openai(file):
    b64 = base64.b64encode(file.getvalue()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{file.type};base64,{b64}"}
    }

def parse_json_loose(text: str):
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    import re
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

# ----------------------------
# OPENAI VISION
# ----------------------------
def call_openai_vision():
    prompt = """
You are a retina specialist.
Task:
1) Describe morphology (what you actually see) first.
2) Output STRICT JSON ONLY (no markdown, no extra text).
Return JSON keys:
- modalities_detected: list[str]
- image_quality: one of ["POOR","FAIR","GOOD","EXCELLENT"]
- key_findings: list[str] (short bullets)
- top_diagnoses: list[{"name": str, "confidence": float, "for": list[str], "against": list[str]}]
"""
    content = [{"type": "text", "text": prompt}]
    for f in uploads:
        content.append(encode_image_openai(f))

    resp = openai_client.chat.completions.create(
        model=openai_model,
        messages=[{"role": "user", "content": content}],
        temperature=0
    )
    raw = resp.choices[0].message.content
    js = parse_json_loose(raw)
    return raw, js

# ----------------------------
# GEMINI VISION (2025-2026 google-genai uyumlu)
# ----------------------------
def gemini_part_text(text: str):
    return types.Part(text=text)

def gemini_part_bytes(data: bytes, mime_type: str):
    return types.Part(
        inline_data=types.Blob(
            mime_type=mime_type,
            data=data
        )
    )

def call_gemini_vision():
    if not (use_gemini and gemini_client):
        return None, None

    prompt = """
You are a retina specialist.
Task:
1) Describe morphology first.
2) Output STRICT JSON ONLY (no markdown, no extra text).
Return JSON keys:
- modalities_detected: list[str]
- image_quality: one of ["POOR","FAIR","GOOD","EXCELLENT"]
- key_findings: list[str]
- top_diagnoses: list[{"name": str, "confidence": float, "for": list[str], "against": list[str]}]
"""

    parts = [gemini_part_text(prompt)]

    for f in uploads:
        parts.append(gemini_part_bytes(f.getvalue(), f.type))

    contents = [types.Content(role="user", parts=parts)]

    generation_config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json"
    )

    try:
        resp = gemini_client.models.generate_content(
            model=gemini_model,
            contents=contents,
            generation_config=generation_config
        )

        if hasattr(resp, "text") and resp.text:
            raw = resp.text
        else:
            try:
                raw = resp.candidates[0].content.parts[0].text
            except Exception:
                raw = str(resp)

        js = parse_json_loose(raw)
        return raw, js

    except Exception as e:
        error_msg = f"Gemini call failed: {str(e)}"
        return json.dumps({"error": error_msg})[:2000], None

# ----------------------------
# FINAL REPORT (OpenAI reasoning)
# ----------------------------
def build_final_report(openai_json, gemini_json):
    payload = {
        "clinical": clinical,
        "openai_vision": openai_json,
        "gemini_vision": gemini_json
    }

    system = """
You are RetinaGPT, a retina subspecialty educational discussion and decision-support system.
PURPOSE
Provide structured academic discussion of retinal imaging findings and differential diagnostic reasoning for educational purposes only. Not medical advice.
STYLE
- Formal medical English, objective, concise.
- Describe morphology first.
- Then ranked differential with discriminators.
- If OpenAI vs Gemini disagree, explicitly state the disagreement and propose what additional imaging/clinical data would resolve it.
OUTPUT
Write a clean report with:
1) Imaging summary (morphology)
2) Most likely diagnosis + why
3) Differential (ranked)
4) What to obtain next (OCT/FAF/OCTA/FA etc.)
5) Triage: routine / urgent / emergent (educational)
"""

    resp = openai_client.chat.completions.create(
        model=openai_reason_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, indent=2)}
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content

# ----------------------------
# UI RUN
# ----------------------------
if run:
    if not uploads:
        st.error("Please upload at least one image.")
    else:
        with st.spinner("Running OpenAI vision..."):
            openai_raw, openai_json = call_openai_vision()

        with st.spinner("Running Gemini vision..."):
            gemini_raw, gemini_json = call_gemini_vision()

        with st.spinner("Generating final report..."):
            final_report = build_final_report(openai_json, gemini_json)

        st.subheader("Final Report")
        st.markdown(final_report)

        st.subheader("Debug / Raw Outputs")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### OpenAI raw response")
            st.code(openai_raw or "No response", language="json")
            st.markdown("### OpenAI parsed JSON")
            st.json(openai_json)

        with col2:
            st.markdown("### Gemini raw response")
            st.code(gemini_raw or "No response", language="json")
            st.markdown("### Gemini parsed JSON")
            st.json(gemini_json)
