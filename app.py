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
TERMINOLOGY STANDARD: Use formal ophthalmic subspecialty terminology. (e.g., cotton-wool spot, cherry-red spot).
STRUCTURED ANALYSIS: 1) Imaging Quality, 2) Structural Findings, 3) Vascular Findings, 4) Peripheral Assessment, 5) Pattern Discussion, 6) Pathophysiologic Considerations.
DIFFERENTIAL: Provide up to three diagnostic considerations. Use: 'Additional clinical or imaging data that may help clarify the pattern include:'.
LIMITATIONS: Educational purposes only. Not medical advice.
"""

# ---------------- API key (hidden) ----------------
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.error("OPENAI_API_KEY bulunamadı. Lütfen .streamlit/secrets.toml içine ekleyin.")
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

# ---------------- Header ----------------
st.title("👁️ RetinaGPT")
st.caption("Prepared by Mehmet ÇITIRIK & Caner KARA")
st.markdown("---")

# ---------------- Upload ----------------
uploaded = st.file_uploader(
    "Fundus/OCT/FAF/FFA görüntüsü yükleyin (jpg/png/webp)",
    type=["jpg", "jpeg", "png", "webp"],
    key=f"uploader_{st.session_state.uploader_key}",
    disabled=st.session_state.case_done
)

# ---------------- Chat history ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------- New Patient button after analysis ----------------
if st.session_state.case_done:
    st.divider()
    if st.button("🧼 Ask New Patient", use_container_width=True):
        reset_case()
    st.caption("Clears the current case (messages + image) and starts a fresh patient.")

# ---------------- Chat input ----------------
placeholder_text = (
    "Bulguları veya vaka detaylarını buraya yazın... "
    "(Age, Sex, Primary symptom, Symptom duration: acute/subacute/chronic, "
    "Laterality: unilateral/bilateral, Relevant history, Imaging modality)"
)

prompt = st.chat_input(placeholder_text, disabled=st.session_state.case_done)

# ---------------- On submit ----------------
if prompt:
    if uploaded is None:
        st.error("Lütfen bir görüntü yükleyin. (Analiz için görüntü gerekli.)")
        st.stop()

    # show/store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # image -> data URL
    img_bytes = uploaded.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    mime = uploaded.type
    data_url = f"data:{mime};base64,{b64}"

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
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                },
            ],
        )

        full_response = response.output_text
        st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # finish case -> show New Patient button
    st.session_state.case_done = True
    st.rerun()
