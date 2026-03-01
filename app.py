import streamlit as st
from openai import OpenAI
import base64

# Sayfa Yapılandırması
st.set_page_config(page_title="RetinaGPT v3", page_icon="👁️")

def encode_image(uploaded_file):
    """Görüntüyü OpenAI API'nin okuyabileceği formata çevirir."""
    return base64.b64encode(uploaded_file.read()).decode('utf-8')

st.title("RetinaGPT v3 – Academic Retinal Case Analysis")

api_key = st.text_input("Enter OpenAI API Key", type="password")

st.subheader("Clinical Information")

col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age")
    sex = st.selectbox("Sex", ["Male", "Female"])
    laterality = st.selectbox("Laterality", ["Unilateral", "Bilateral"])
with col2:
    symptom = st.text_input("Primary Symptom")
    duration = st.selectbox("Symptom Duration", ["Acute", "Subacute", "Chronic"])

uploaded_files = st.file_uploader(
    "Upload retinal images (Fundus, OCT, FFA, OCTA)",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"]
)

if st.button("Analyze Case"):
    if not api_key:
        st.warning("Please enter API key.")
    elif not uploaded_files:
        st.warning("Please upload at least one image for analysis.")
    else:
        client = OpenAI(api_key=api_key)

        # 1. Klinik Bağlamı Hazırla
        clinical_context = f"Age: {age}, Sex: {sex}, Symptom: {symptom}, Duration: {duration}, Laterality: {laterality}"

        # 2. Mesaj Yapısını Kur (System Prompt senin orijinal talimatın olmalı)
        messages = [
            {
                "role": "system",
                "content": """You are a retina subspecialty educational discussion system. 
                Analyze the provided retinal imaging patterns and provide a structured academic discussion.
                STRUCTURE: 1) Imaging Quality, 2) Structural Findings, 3) Vascular Findings, 4) Peripheral Assessment, 5) Pattern Discussion, 6) Pathophysiologic Considerations.
                DIFFERENTIAL: Provide up to three diagnostic considerations with supporting/conflicting features.
                Formal medical English only."""
            }
        ]

        # 3. Görüntüleri Mesaja Ekle
        user_content = [{"type": "text", "text": f"Clinical Data: {clinical_context}\n\nPlease analyze the attached images."}]
        
        for uploaded_file in uploaded_files:
            base64_image = encode_image(uploaded_file)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        messages.append({"role": "user", "content": user_content})

        # 4. API'ye Gönder
        with st.spinner("Analyzing images and clinical data..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2000
                )

                st.subheader("RetinaGPT Analysis")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"An error occurred: {e}")
