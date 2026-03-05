import os
import json
import base64
import streamlit as st
from openai import OpenAI

# Gemini (google-genai)
from google import genai
from google.genai import types

st.set_page_config(page_title="RetinaGPT (Dual Opinion)", page_icon="👁️", layout="wide")
st.title("👁️ RetinaGPT — Dual Opinion (OpenAI + Gemini)")

st.caption("Educational only. Not medical advice.")

# ---------------------------
# Keys
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing (env or Streamlit secrets).")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)

gemini_client = None
if GEMINI_API_KEY:
    # google-genai client
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------
# Sidebar settings
# ---------------------------
with st.sidebar:
    st.header("Models")

    openai_vision_model = st.text_input("OpenAI vision model", value="gpt-4o")
    openai_arbiter_model = st.text_input("OpenAI arbiter model", value="gpt-4o-mini")

    st.divider()
    use_gemini = st.checkbox("Enable Gemini", value=True)

    # IMPORTANT: gemini-1.5-flash may be gone in your API/version -> default to a newer one
    # You can click "List Gemini models" to find the exact ones available to your key.
    gemini_model = st.text_input("Gemini model", value="gemini-3-flash-preview")

    if st.button("List Gemini models (debug)") and gemini_client:
        try:
            models = gemini_client.models.list()
            names = []
            for m in models:
                # Different SDK versions expose fields differently; keep it defensive
                n = getattr(m, "name", None) or str(m)
                names.append(n)
            st.write("Available Gemini models for this key:")
            st.code("\n".join(names[:200]))
        except Exception as e:
            st.error(f"ListModels failed: {e}")

    st.divider()
    st.caption("Tip: If Gemini returns 404 NOT_FOUND, your model name is not available for your API/key.")

# ---------------------------
# Inputs
# ---------------------------
clinical = st.text_area(
    "Clinical info (age, sex, symptoms, duration, laterality, history)",
    height=140,
    placeholder="Example: 58M, sudden painless vision loss OD 2 days, HTN, no DM..."
)

uploads = st.file_uploader(
    "Upload retinal images (jpg/png)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

colA, colB = st.columns([1, 1])
with colA:
    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)
with colB:
    st.write("")

# ---------------------------
# Helpers
# ---------------------------
def openai_image_block(file) -> dict:
    b64 = base64.b64encode(file.getvalue()).decode("utf-8")
    return {
        "type": "input_image",
        "image_url": f"data:{file.type};base64,{b64}",
    }

def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

# ---------------------------
# OpenAI opinion (free-form text OR JSON)
# ---------------------------
def call_openai_opinion(clinical_text: str, files):
    prompt = f"""
You are a retina specialist. Analyze the retinal image(s) and clinical info.
Return an EDUCATIONAL assessment (not medical advice).

Format:
1) Key imaging findings (morphology-first, modality-specific if possible)
2) Most likely diagnosis (ranked top 3) with brief justification
3) Important differentials to exclude + what would help (OCT/FAF/OCTA/FA etc.)
4) Red flags / urgency (educational)
5) If uncertain, say why.
Clinical info:
{clinical_text if clinical_text.strip() else "(none provided)"}
"""

    content = [{"type": "input_text", "text": prompt}]
    for f in files:
        content.append(openai_image_block(f))

    # Responses API (recommended)
    resp = openai_client.responses.create(
        model=openai_vision_model,
        input=[{"role": "user", "content": content}],
        temperature=0,
    )
    return resp.output_text

# ---------------------------
# Gemini opinion
# ---------------------------
def call_gemini_opinion(clinical_text: str, files):
    if not gemini_client:
        return None

    prompt = f"""
You are a retina specialist. Analyze the retinal image(s) and clinical info.
Return an EDUCATIONAL assessment (not medical advice).

Format:
1) Key imaging findings (morphology-first)
2) Most likely diagnosis (ranked top 3) + brief justification
3) Differentials + what additional imaging/clinical info would help
4) Red flags / urgency (educational)
Clinical info:
{clinical_text if clinical_text.strip() else "(none provided)"}
"""

    # Newer google-genai patterns: pass text as STRING + images as Part.from_bytes
    contents = [prompt]

    for f in files:
        contents.append(
            types.Part.from_bytes(
                data=f.getvalue(),
                mime_type=f.type
            )
        )

    resp = gemini_client.models.generate_content(
        model=gemini_model.strip(),
        contents=contents,
        # keep it deterministic
        config=types.GenerateContentConfig(temperature=0),
    )
    # Depending on SDK, text may be resp.text
    return getattr(resp, "text", None) or str(resp)

# ---------------------------
# Arbiter / Consensus
# ---------------------------
def build_consensus_report(clinical_text: str, openai_opinion: str, gemini_opinion: str):
    payload = {
        "clinical_info": clinical_text,
        "openai_opinion": openai_opinion,
        "gemini_opinion": gemini_opinion,
    }

    arbiter_prompt = """
You are a senior retina specialist acting as an arbiter.
You will be given:
- Clinical info
- OpenAI opinion
- Gemini opinion (may be missing)

Tasks:
A) Summarize shared findings (agreement)
B) Summarize disagreements (what differs) and WHY disagreement may occur
C) Provide a cautious consensus diagnosis ranking (top 3) with confidence (low/med/high)
D) Recommend the single most useful next test/imaging to resolve uncertainty
E) Keep it educational, morphology-first, avoid over-commitment.
"""

    resp = openai_client.responses.create(
        model=openai_arbiter_model,
        input=[
            {"role": "system", "content": arbiter_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        temperature=0,
    )
    return resp.output_text

# ---------------------------
# Run
# ---------------------------
if analyze_btn:
    if not uploads:
        st.error("Please upload at least one image.")
        st.stop()

    with st.spinner("Running OpenAI vision..."):
        openai_text = call_openai_opinion(clinical, uploads)

    gemini_text = None
    gemini_error = None

    if use_gemini:
        if not gemini_client:
            gemini_error = "GEMINI_API_KEY missing, Gemini disabled."
        else:
            with st.spinner("Running Gemini vision..."):
                try:
                    gemini_text = call_gemini_opinion(clinical, uploads)
                except Exception as e:
                    gemini_error = str(e)

    with st.spinner("Building consensus report..."):
        consensus = build_consensus_report(clinical, openai_text, gemini_text)

    st.subheader("✅ Consensus Report (Arbiter)")
    st.write(consensus)

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("OpenAI opinion")
        st.write(openai_text)

    with c2:
        st.subheader("Gemini opinion")
        if gemini_text:
            st.write(gemini_text)
        else:
            st.warning("Gemini opinion not available.")
            if gemini_error:
                st.code(gemini_error)

    st.divider()
    with st.expander("Debug (raw)"):
        st.write("OpenAI raw:")
        st.code(openai_text)

        st.write("Gemini raw:")
        st.code(gemini_text if gemini_text else "(none)")
