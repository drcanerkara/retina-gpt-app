import streamlit as st
from openai import OpenAI

st.title("RetinaGPT v3 – Academic Retinal Case Analysis")

api_key = st.text_input("Enter OpenAI API Key", type="password")

st.subheader("Clinical Information")

age = st.text_input("Age")
sex = st.selectbox("Sex", ["Male", "Female"])
symptom = st.text_input("Primary Symptom")
duration = st.selectbox("Symptom Duration", ["Acute", "Subacute", "Chronic"])
laterality = st.selectbox("Laterality", ["Unilateral", "Bilateral"])

uploaded_files = st.file_uploader(
    "Upload retinal images (Fundus, OCT, FFA, OCTA)",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"]
)

if st.button("Analyze Case"):
    if not api_key:
        st.warning("Please enter API key.")
    else:
        client = OpenAI(api_key=api_key)

        clinical_context = f"""
        Age: {age}
        Sex: {sex}
        Primary Symptom: {symptom}
        Duration: {duration}
        Laterality: {laterality}
        """

        messages = [
            {
                "role": "system",
                "content": "You are a retina subspecialty educational analysis assistant. Provide structured retinal assessment."
            },
            {
                "role": "user",
                "content": clinical_context
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2
        )

        st.subheader("RetinaGPT Analysis")
        st.write(response.choices[0].message.content)
