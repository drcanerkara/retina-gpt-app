import streamlit as st
import base64
from openai import OpenAI

st.set_page_config(page_title="Retina Academic Discussion", page_icon="👁️")

with st.sidebar:
    st.title("⚙️ Ayarlar")
    api_key = st.text_input("OpenAI API Key Giriniz:", type="password")
    st.info("Bu sistem akademik retina görüntüleme analizi için tasarlanmıştır.")

st.title("👁️ Retina Subspecialty Educational System")
st.markdown("---")

SYSTEM_PROMPT = """
You are a retina subspecialty educational discussion system.
Your purpose is to provide structured academic discussion of retinal imaging findings and differential diagnostic reasoning for educational purposes.
All outputs must be structured, objective, concise, and written in formal medical English.
TERMINOLOGY STANDARD: Use formal ophthalmic subspecialty terminology. (e.g., cotton-wool spot, cherry-red spot).
STRUCTURED ANALYSIS: 1) Imaging Quality, 2) Structural Findings, 3) Vascular Findings, 4) Peripheral Assessment, 5) Pattern Discussion, 6) Pathophysiologic Considerations.
DIFFERENTIAL: Provide up to three diagnostic considerations. Use: 'Additional clinical or imaging data that may help clarify the pattern include:'.
LIMITATIONS: Educational purposes only. Not medical advice.
"""

# ---- Image uploader (kritik parça) ----
uploaded = st.file_uploader(
    "Fundus/OCT/FAF/FFA görüntüsü yükleyin (jpg/png)",
    type=["jpg", "jpeg", "png", "webp"]
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Bulguları veya vaka detaylarını buraya yazın...")

if prompt:
    if not api_key:
        st.error("Lütfen sol tarafa API Key giriniz!")
        st.stop()

    if uploaded is None:
        st.error("Lütfen bir görüntü yükleyin. (Şu an modele görüntü gitmiyor.)")
        st.stop()

    # kullanıcı mesajı ekrana ve history'e
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # image -> base64 data URL
    img_bytes = uploaded.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    mime = uploaded.type  # e.g. image/png
    data_url = f"data:{mime};base64,{b64}"

    client = OpenAI(api_key=api_key)

    with st.chat_message("assistant"):
        # Responses API (multimodal için en düzgün yol)
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                # önce geçmişi text olarak aktar (isterseniz kısaltabilirsiniz)
                *[
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ],
                # bu turda multimodal içerik
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
