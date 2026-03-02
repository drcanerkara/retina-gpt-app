import streamlit as st
from openai import OpenAI
import base64
from PIL import Image
import io

# Sayfa ayarları
st.set_page_config(page_title="Retina Academic Discussion", page_icon="👁️")

# Sol panel
with st.sidebar:
    st.title("⚙️ Ayarlar")
    api_key = st.text_input("OpenAI API Key Giriniz:", type="password")
    st.info("Bu sistem akademik retina görüntüleme analizi için tasarlanmıştır.")

st.title("👁️ Retina Subspecialty Educational System")
st.markdown("---")

# Sistem prompt (aynı kalabilir)
SYSTEM_PROMPT = """
You are a retina subspecialty educational discussion system.
Your purpose is to provide structured academic discussion of retinal imaging findings and differential diagnostic reasoning for educational purposes.
All outputs must be structured, objective, concise, and written in formal medical English.
TERMINOLOGY STANDARD: Use formal ophthalmic subspecialty terminology. (e.g., cotton-wool spot, cherry-red spot).
STRUCTURED ANALYSIS: 1) Imaging Quality, 2) Structural Findings, 3) Vascular Findings, 4) Peripheral Assessment, 5) Pattern Discussion, 6) Pathophysiologic Considerations.
DIFFERENTIAL: Provide up to three diagnostic considerations. Use: 'Additional clinical or imaging data that may help clarify the pattern include:'.
LIMITATIONS: Educational purposes only. Not medical advice.
"""

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları göster (resim varsa göster)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user" and "image" in message:
            st.image(message["image"], caption="Yüklenen fundus görüntüsü", use_column_width=True)
        st.markdown(message["content"])

# Resim yükleme alanı (chat altına veya sidebar'a koyabilirsin)
uploaded_file = st.file_uploader("Retina/fundus görüntüsü yükleyin (jpg/png)", type=["jpg", "jpeg", "png"])

# Chat input
if prompt := st.chat_input("Bulguları, vaka detaylarını veya sorunuzu yazın..."):

    if not api_key:
        st.error("Lütfen sol taraftaki API Key alanını doldurun!")
    else:
        # Kullanıcı mesajını hazırla
        user_content = [{"type": "text", "text": prompt}]

        image_data = None
        if uploaded_file is not None:
            # Resmi base64'e çevir
            bytes_data = uploaded_file.getvalue()
            base64_image = base64.b64encode(bytes_data).decode('utf-8')
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
            # Resmi session'a da kaydet (gösterim için)
            image_data = Image.open(io.BytesIO(bytes_data))

        full_user_message = {"role": "user", "content": prompt}
        if image_data:
            full_user_message["image"] = image_data  # gösterim için PIL Image

        st.session_state.messages.append(full_user_message)

        with st.chat_message("user"):
            if image_data:
                st.image(image_data, caption="Yüklenen görüntü", use_column_width=True)
            st.markdown(prompt)

        # OpenAI çağrısı
        client = OpenAI(api_key=api_key)

        with st.chat_message("assistant"):
            with st.spinner("RetinaGPT analiz ediyor..."):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *[
                            # Eski mesajları multimodal uyumlu hale getir (sadece text varsa text, yoksa content list)
                            msg if isinstance(msg.get("content"), str) else
                            {"role": msg["role"], "content": [c for c in msg.get("content", []) if isinstance(c, dict)]}
                            for msg in st.session_state.messages
                        ]
                    ],
                    max_tokens=1500,
                    temperature=0.3  # daha tutarlı tıbbi cevap için düşük
                )

                full_response = response.choices[0].message.content
                st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
