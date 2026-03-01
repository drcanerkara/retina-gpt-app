import streamlit as st
from openai import OpenAI
import base64

st.set_page_config(page_title="RetinaGPT v3", page_icon="👁️")

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode('utf-8')

st.title("RetinaGPT v3 – Academic Retinal Case Analysis")

api_key = st.text_input("Enter OpenAI API Key", type="password")

# Klinik Veri Girişi
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age")
    sex = st.selectbox("Sex", ["Male", "Female"])
with col2:
    symptom = st.text_input("Primary Symptom")
    duration = st.selectbox("Symptom Duration", ["Acute", "Subacute", "Chronic"])

uploaded_files = st.file_uploader("Upload retinal images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if st.button("Analyze Case"):
    if not api_key:
        st.error("Please enter your OpenAI API Key.")
    elif not uploaded_files:
        st.error("Please upload at least one image.")
    else:
        try:
            client = OpenAI(api_key=api_key)
            
            # Görüntüleri API formatına hazırlıyoruz
            content_list = []
            
            # Talimat ve Klinik Veri
            instruction_text = f"""You are a senior retina specialist. Analyze the ATTACHED IMAGES.
            CLINICAL CONTEXT: Age {age}, Sex {sex}, Symptom: {symptom}, Duration: {duration}.
            
            TASK: Look at the images and describe the specific findings. 
            Do NOT say you cannot see the images. Use the following structure:
            1) Imaging Quality 2) Structural Findings 3) Vascular Findings 4) Peripheral Assessment 5) Pattern Discussion 6) Pathophysiologic Considerations.
            DIFFERENTIAL: Up to three considerations."""
            
            content_list.append({"type": "text", "text": instruction_text})

            # Resimleri ekle
            for uploaded_file in uploaded_files:
                base64_img = encode_image(uploaded_file)
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}",
                        "detail": "high" # Yüksek çözünürlükte analiz etmesini söyler
                    }
                })

            with st.spinner("GPT is looking at the images..."):
                response = client.chat.completions.create(
                    model="gpt-4o", # Mutlaka gpt-4o olmalı
                    messages=[{"role": "user", "content": content_list}],
                    max_tokens=1500,
                    temperature=0.1
                )
                
                st.markdown("---")
                st.subheader("Analysis Result")
                st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"Error: {e}")
