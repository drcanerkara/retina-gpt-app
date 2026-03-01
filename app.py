import streamlit as st
from openai import OpenAI
import base64

st.set_page_config(page_title="RetinaGPT Pro", page_icon="👁️")

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode('utf-8')

st.title("RetinaGPT v3 – Professional Imaging Analysis")

api_key = st.text_input("Enter OpenAI API Key", type="password")

# Klinik Parametreler
col1, col2 = st.columns(2)
with col1:
    age = st.text_input("Age (e.g., 65)")
    symptom = st.text_input("Symptoms (e.g., Vision Loss)")
with col2:
    duration = st.text_input("Duration (e.g., 2 days)")
    laterality = st.selectbox("Laterality", ["Unilateral", "Bilateral"])

uploaded_files = st.file_uploader("Upload Retinal Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if st.button("RUN MULTIMODAL ANALYSIS"):
    if not api_key:
        st.error("Please enter your API Key.")
    elif not uploaded_files:
        st.error("Please upload images.")
    else:
        try:
            client = OpenAI(api_key=api_key)
            
            # Görüntüleri hazırlıyoruz
            content_list = []
            
            # GPT'Yİ GÖRMEYE ZORLAYAN PROFESYONEL TALİMAT
            instruction = f"""You are a specialized Retina Imaging AI assistant. 
            The user is a medical professional using this for research/education.
            
            YOUR TASK: Analyze the specific visual findings in the attached retinal images.
            CONTEXT: Age {age}, {symptom}, {duration}, {laterality}.
            
            DO NOT output a generic guide. Describe the EXACT lesions, hemorrhages, or fluid visible in THESE images.
            
            STRUCTURE:
            1) Imaging Quality
            2) Structural Morphology (Describe macula, fovea, and nerve head)
            3) Vascular Findings (Describe vessels)
            4) Peripheral Assessment
            5) Pattern Correlation (Identify specific patterns like drusen, exudates, etc.)
            6) Theoretical Pathophysiology
            
            DIFFERENTIAL: List 3 possible correlations based on VISUAL DATA."""
            
            content_list.append({"type": "text", "text": instruction})

            for uploaded_file in uploaded_files:
                base64_img = encode_image(uploaded_file)
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}",
                        "detail": "high"
                    }
                })

            with st.spinner("AI is scanning the pixels..."):
                response = client.chat.completions.create(
                    model="gpt-4o", 
                    messages=[{"role": "user", "content": content_list}],
                    max_tokens=2000,
                    temperature=0.1
                )
                
                st.markdown("---")
                st.markdown("### 🔬 RetinaGPT Multimodal Analysis")
                st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"Error: {e}")
