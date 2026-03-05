import streamlit as st
import google.generativeai as genai
from openai import OpenAI

# API Ayarları
genai.configure(api_key="GEMINI_API_KEY")
client = OpenAI(api_key="OPENAI_API_KEY")

st.title("Hibrit Göz Analiz Sistemi")

uploaded_file = st.file_uploader("Fundus Görüntüsü Yükleyin", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    st.image(uploaded_file, caption='Yüklenen Görüntü', use_column_width=True)
    
    if st.button("Analiz Et"):
        with st.spinner('Her iki model de inceliyor...'):
            
            # 1. Gemini Analizi
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            # (Görüntü işleme adımları buraya gelecek)
            gemini_res = gemini_model.generate_content(["Bu fundus görüntüsünü analiz et.", image])
            
            # 2. ChatGPT Analizi
            # (Görüntü base64'e çevrilerek gönderilecek)
            gpt_res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Bu fundus görüntüsünü analiz et."}]
            )

            # 3. Ortak Karar Mantığı
            st.subheader("Analiz Sonuçları")
            col1, col2 = st.columns(2)
            col1.write(f"**Gemini:** {gemini_res.text}")
            col2.write(f"**ChatGPT:** {gpt_res.choices[0].message.content}")

            # Basit bir "eğer ikisinde de X geçiyorsa" kontrolü
            if "ven" in gemini_res.text.lower() and "ven" in gpt_res.choices[0].message.content.lower():
                st.success("ORTAK KARAR: İki model de Ven Tıkanıklığı konusunda hemfikir.")
            else:
                st.warning("DİKKAT: Modeller farklı bulgular saptadı, lütfen uzman doktora danışın.")
