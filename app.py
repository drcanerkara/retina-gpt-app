import streamlit as st
from openai import OpenAI
import base64

st.set_page_config(page_title="RetinaGPT v3", page_icon="👁️")

# Görüntüyü hazırlayan fonksiyon
def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode('utf-8')

st.title("RetinaGPT v3 – Academic Retinal Case Analysis")

api_key = st.text_input("Enter OpenAI API Key", type="password")

# Klinik Veri Girişi
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age")
    sex = st.selectbox("Sex", ["Male", "Female"])
    laterality = st.selectbox("Laterality", ["Unilateral", "Bilateral"])
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
        client = OpenAI(api_key=api_key)
        
        # Talimatları ve klinik veriyi birleştiriyoruz
        instruction_text = f"""You are a retina subspecialty educational discussion system. 
        Analyze the ATTACHED IMAGES based on this clinical context:
        Age: {age}, Sex: {sex}, Symptom: {symptom}, Duration: {duration}, Laterality: {laterality}.
        
        PROVIDE A DETAILED ANALYSIS IN THIS STRUCTURE:
        1) Imaging Quality
        2) Structural Findings
        3) Vascular Findings
        4) Peripheral Assessment
        5) Pattern Discussion (Analyze what you see in the image)
        6) Pathophysiologic Considerations
        
        DIFFERENTIAL: Provide up to three diagnostic considerations.
        Always refer to the specific findings visible in the uploaded images."""

        # OpenAI Vision formatına uygun mesaj hazırlığı
        content = [{"type": "text", "text": instruction_text}]
        
        for uploaded_file in uploaded_files:
            base64_img = encode_image(uploaded_file)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })

        messages = [{"role": "user", "content": content}]

        with st.spinner("Analyzing your images..."):
            try:
                # GPT-4o Vision modelini çağırıyoruz
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=1500,
                    temperature=0.1
                )
                
                st.markdown("---")
                st.subheader("RetinaGPT Analysis Result")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"API Error: {str(e)}")
