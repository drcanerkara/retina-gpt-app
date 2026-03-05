import streamlit as st
import google.generativeai as genai
from openai import OpenAI
from PIL import Image
import base64
import io
import json

# --- API YAPILANDIRMASI ---
# Not: API anahtarlarınızı Streamlit Secrets veya .env üzerinden yönetin
genai.configure(api_key="GEMINI_API_KEY")
client = OpenAI(api_key="OPENAI_API_KEY")

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_consensous_analysis(image):
    # 1. Gemini Analizi
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = """Bu bir fundus (göz dibi) fotoğrafıdır. 
    Lütfen şu formatta JSON yanıt ver: {"tani": "Hastalık Adı", "bulgu": "Kısa açıklama"}"""
    
    gemini_resp = model.generate_content([prompt, image])
    # JSON temizleme (bazı modeller ```json ... ``` içinde verir)
    gemini_data = json.loads(gemini_resp.text.replace('```json', '').replace('```', ''))

    # 2. OpenAI GPT-4o Analizi
    base64_image = encode_image_to_base64(image)
    gpt_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }],
        response_format={ "type": "json_object" }
    )
    gpt_data = json.loads(gpt_resp.choices[0].message.content)

    return gemini_data, gpt_data

# --- STREAMLIT ARAYÜZÜ ---
st.title("👨‍⚕️ Hibrit Göz Tanı Asistanı")
uploaded_file = st.file_uploader("Fundus fotoğrafı yükleyin...", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='Yüklenen Görüntü', width=400)
    
    if st.button("Konsültasyon Başlat"):
        with st.spinner('Yapay zeka modelleri tartışıyor...'):
            gem_res, gpt_res = get_consensous_analysis(img)
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Gemini:** {gem_res['tani']}")
            with col2:
                st.info(f"**ChatGPT:** {gpt_res['tani']}")

            # ORTAK KARAR MANTIĞI
            # Basit bir anahtar kelime karşılaştırması
            gem_tani = gem_res['tani'].lower()
            gpt_tani = gpt_res['tani'].lower()

            if gem_tani in gpt_tani or gpt_tani in gem_tani:
                st.success(f"✅ **ORTAK KARAR:** Modeller {gem_res['tani']} tanısında hemfikir.")
            else:
                st.warning("⚠️ **ÇELİŞKİ:** Modeller farklı tanılar koydu. Manuel inceleme önerilir.")
                
            st.write("**Ayrıntılı Bulgular:**")
            st.write(f"- Gemini: {gem_res['bulgu']}")
            st.write(f"- ChatGPT: {gpt_res['bulgu']}")
