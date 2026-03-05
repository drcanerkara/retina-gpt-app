import os
import base64
import streamlit as st
from openai import OpenAI

# Gemini (new SDK)
from google import genai

st.set_page_config(page_title="RetinaGPT - Dual Opinion", page_icon="👁️", layout="wide")
st.title("👁️ Dual-LLM Retina Opinions (ChatGPT + Gemini)")

# -----------------------------
# Keys
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing (env or Streamlit secrets).")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)

gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options={"api_version": "v1"}
    )

# -----------------------------
# Sidebar settings
# -----------------------------
with st.sidebar:
    st.header("Models")

    openai_vision_model = st.text_input("OpenAI vision model", value="gpt-4o")
    openai_consensus_model = st.text_input("OpenAI consensus model", value="gpt-4o-mini")

    gemini_model = st.text_input("Gemini model", value="gemini-1.5-flash")
    use_gemini = st.checkbox("Enable Gemini", value=True)

    st.divider()
    st.caption("This tool is educational only. Not medical advice.")

# -----------------------------
# Inputs
# -----------------------------
clinical = st.text_area(
    "Clinical info (age, sex, symptoms, duration, laterality, history)",
    height=140,
    placeholder="Example: 58M, sudden painless vision loss OD 2 days, HTN, no DM..."
)

uploads = st.file_uploader(
    "Upload retinal images (Fundus / OCT / FAF / FA / OCTA)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

colA, colB = st.columns([1, 1])

with colA:
    run = st.button("Analyze", type="primary")

with colB:
    st.button("Reset (refresh page)")

# -----------------------------
# Helpers
# -----------------------------
def openai_data_url(file) -> str:
    b64 = base64.b64encode(file.getvalue()).decode("utf-8")
    return f"data:{file.type};base64,{b64}"

def call_openai_opinion(clinical_text: str, files):
    """
    Returns free-text 'expert opinion' from OpenAI vision model (no JSON).
    """
    content = [
        {
            "type": "input_text",
            "text": f"""
You are a retina subspecialist. Provide an EDUCATIONAL imaging-based opinion.

Rules:
- Describe morphology first, then give a ranked differential.
- State confidence and key discriminators.
- If information is insufficient, say what additional imaging/clinical detail would change your ranking.
- Do NOT output JSON. Output clear structured text with headings.

Clinical info:
{clinical_text if clinical_text else "(none provided)"}
"""
        }
    ]

    for f in files:
        content.append(
            {"type": "input_image", "image_url": openai_data_url(f), "detail": "high"}
        )

    # Use Responses API style to match current best practice
    resp = openai_client.responses.create(
        model=openai_vision_model,
        input=[{"role": "user", "content": content}],
        temperature=0
    )
    return resp.output_text

def call_gemini_opinion(clinical_text: str, files):
    """
    Returns free-text 'expert opinion' from Gemini (no JSON).
    Uses a simple contents list (text + inline image bytes) to avoid types.Part churn.
    """
    if not gemini_client:
        return None

    # Build "contents" as a list of parts (string + images).
    # google-genai accepts bytes image parts via dict with inline_data.
    contents = []

    prompt = f"""
You are a retina subspecialist. Provide an EDUCATIONAL imaging-based opinion.

Rules:
- Describe morphology first, then give a ranked differential.
- State confidence and key discriminators.
- If information is insufficient, say what additional imaging/clinical detail would change your ranking.
- Do NOT output JSON. Output clear structured text with headings.

Clinical info:
{clinical_text if clinical_text else "(none provided)"}
"""
    contents.append(prompt)

    for f in files:
        contents.append({
            "inline_data": {
                "mime_type": f.type,
                "data": f.getvalue()
            }
        })

    resp = gemini_client.models.generate_content(
        model=gemini_model,
        contents=contents
    )

    # In many environments resp.text is the final rendered text
    return getattr(resp, "text", None) or str(resp)

def call_consensus(openai_text: str, gemini_text: str, clinical_text: str):
    """
    Uses OpenAI to generate a consensus / arbitration report from both opinions.
    """
    user_payload = f"""
Clinical info:
{clinical_text if clinical_text else "(none provided)"}

--- OPENAI (ChatGPT) OPINION ---
{openai_text}

--- GEMINI OPINION ---
{gemini_text if gemini_text else "(Gemini not available or failed)"}
"""

    system = """
You are a retina subspecialist acting as an independent arbiter.

Task:
1) Summarize shared findings (agreement).
2) Summarize disagreements (what differs).
3) Provide a reasoned consensus: ranked differential with confidence.
4) If disagreement persists, list the TOP 3 additional inputs (imaging or clinical) most likely to resolve it.
5) Add a short safety note: educational only, clinical correlation needed.

Style:
- Formal medical English.
- Retina subspecialty terminology.
- No JSON.
"""

    resp = openai_client.responses.create(
        model=openai_consensus_model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_payload}
        ],
        temperature=0
    )
    return resp.output_text

# -----------------------------
# UI: Preview uploads
# -----------------------------
if uploads:
    st.subheader("Preview")
    prev_cols = st.columns(min(3, len(uploads)))
    for i, f in enumerate(uploads[:3]):
        with prev_cols[i]:
            st.image(f, caption=f.name, use_container_width=True)

# -----------------------------
# Run
# -----------------------------
if run:
    if not uploads:
        st.error("Please upload at least one image.")
        st.stop()

    with st.spinner("Getting ChatGPT opinion..."):
        openai_op = call_openai_opinion(clinical, uploads)

    gemini_op = None
    if use_gemini:
        if not gemini_client:
            st.warning("Gemini key not found. Gemini is disabled.")
        else:
            with st.spinner("Getting Gemini opinion..."):
                try:
                    gemini_op = call_gemini_opinion(clinical, uploads)
                except Exception as e:
                    gemini_op = None
                    st.error(f"Gemini call failed: {e}")

    with st.spinner("Building consensus report..."):
        consensus = call_consensus(openai_op, gemini_op, clinical)

    st.subheader("✅ Consensus Report (Arbiter)")
    st.write(consensus)

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("ChatGPT opinion (raw)")
        st.write(openai_op)

    with c2:
        st.subheader("Gemini opinion (raw)")
        if gemini_op:
            st.write(gemini_op)
        else:
            st.info("Gemini opinion not available.")
