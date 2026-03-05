import streamlit as st
from openai import OpenAI
import base64

st.set_page_config(page_title="RetinaGPT", page_icon="👁️")

st.title("👁️ RetinaGPT")
st.write("Upload retinal image → ChatGPT analyzes it")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

uploaded = st.file_uploader("Upload retinal image", type=["jpg","jpeg","png"])

if uploaded:

    st.image(uploaded)

    if st.button("Analyze"):

        image_bytes = uploaded.read()
        base64_image = base64.b64encode(image_bytes).decode()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role":"system",
                    "content":"You are a retina specialist. Analyze retinal images carefully and provide differential diagnosis."
                },
                {
                    "role":"user",
                    "content":[
                        {"type":"text","text":"Analyze this retinal image"},
                        {
                            "type":"image_url",
                            "image_url":{
                                "url":f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1200
        )

        result = response.choices[0].message.content

        st.write("### Analysis")
        st.write(result)
