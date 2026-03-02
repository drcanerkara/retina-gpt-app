import streamlit as st
import base64
from openai import OpenAI

# ---------------- Page config ----------------
st.set_page_config(page_title="RetinaGPT", page_icon="👁️")

# ---------------- System prompt ----------------
SYSTEM_PROMPT = """
You are a retina subspecialty educational discussion system.
Your purpose is to provide structured academic discussion of retinal imaging findings and differential diagnostic reasoning for educational purposes.
All outputs must be structured, objective, concise, and written in formal medical English.

TERMINOLOGY STANDARD: Use formal ophthalmic subspecialty terminology (e.g., cotton-wool spot, cherry-red spot).

STRUCTURED ANALYSIS:
1) Imaging Quality
2) Structural Findings
3) Vascular Findings
4) Peripheral Assessment
5) Pattern Discussion
6) Pathophysiologic Considerations

DIFFERENTIAL: Provide up to three diagnostic considerations.
Use: 'Additional clinical or imaging data that may help clarify the pattern include:'.

LIMITATIONS: Educational purposes only. Not medical advice.
"""

# ---------------- API key (hidden) ----------------
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.error("OPENAI_API_KEY not found. Please add it in the Streamlit Cloud Secrets panel.")
    st.stop()

# ---------------- State init ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "case_done" not in st.session_state:
    st.session_state.case_done = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def reset_case():
    st.session_state.messages = []
    st.session_state.case_done = False
    st.session_state.uploader_key += 1  # resets uploader
    st.rerun()


def file_to_data_url(file) -> str:
    """Convert an uploaded file to a data URL (safe for multiple files)."""
    b = file.getvalue()  # do NOT use read() repeatedly
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{file.type};base64,{b64}"


# ---------------- Header (centered) ----------------
st.markdown(
    """
    <div style="text-align: center;">
        <h1>👁️ RetinaGPT</h1>
        <p style="font-size:16px; margin-top:-10px;">
            Prepared by Mehmet ÇITIRIK & Caner KARA
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------- Clinical details (ABOVE image upload) ----------------
clinical_text = st.text_area(
    "Please provide clinical details",
    placeholder="Age, Sex, Symptoms, Duration, Laterality, History",
    height=100,
    disabled=st.session_state.case_done,
)

# ---------------- Upload (multi-file) ----------------
uploaded_files = st.file_uploader(
    "Please upload retinal imaging (Fundus / OCT / FAF / FA) — jpg/png/webp",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
    disabled=st.session_state.case_done,
)

# ---------------- Preview (optional but helpful) ----------------
if uploaded_files and len(uploaded_files) > 0:
    st.subheader("Image Preview")
    for i, f in enumerate(uploaded_files, start=1):
        st.image(f, caption=f"Image {i}: {f.name}")

# ---------------- Chat history ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------- New Patient button after analysis ----------------
if st.session_state.case_done:
    st.divider()
    if st.button("🧼 Ask New Patient", use_container_width=True):
        reset_case()
    st.caption("Clears the current case (messages + images) and starts a fresh patient.")

# ---------------- Analyze button ----------------
analyze = st.button("🔍 Analyze", use_container_width=True, disabled=st.session_state.case_done)

if analyze:
    if not uploaded_files or len(uploaded_files) == 0:
        st.error("Please upload at least one image (required for analysis).")
        st.stop()

    user_payload = (clinical_text or "").strip()
    if not user_payload:
        user_payload = "No clinical details provided."

    # Store user message for display/history
    st.session_state.messages.append({"role": "user", "content": user_payload})
    with st.chat_message("user"):
        st.markdown(user_payload)

    # Build multimodal content blocks (multiple images in one case)
    content_blocks = [
        {"type": "input_text", "text": user_payload},
        {"type": "input_text", "text": "Multiple images uploaded. Interpret them as a single case and integrate findings across modalities."},
    ]

    for i, f in enumerate(uploaded_files, start=1):
        content_blocks.append({"type": "input_text", "text": f"Image {i} filename: {f.name}"})
        content_blocks.append({"type": "input_image", "image_url": file_to_data_url(f)})

    client = OpenAI(api_key=api_key)

    with st.chat_message("assistant"):
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *[
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ],
                {"role": "user", "content": content_blocks},
            ],
        )

        full_response = response.output_text
        st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # finish case -> show New Patient button
    st.session_state.case_done = True
    st.rerun()
