import streamlit as st
import base64
import json
import os

from openai import OpenAI
from google import genai
from google.genai import types

st.set_page_config(page_title="RetinaGPT v4", page_icon="👁️", layout="wide")

st.title("RetinaGPT v4")

# API KEYS
openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

if not openai_key:
    st.error("OPENAI_API_KEY not found")
    st.stop()

openai_client = OpenAI(api_key=openai_key)

gemini_client = None
if gemini_key:
    gemini_client = genai.Client(api_key=gemini_key)

# SETTINGS
with st.sidebar:

    st.header("OpenAI")

    openai_model = st.text_input(
        "OpenAI vision model",
        value="gpt-4o"
    )

    openai_reason_model = st.text_input(
        "OpenAI reasoning model",
        value="gpt-4o-mini"
    )

    st.header("Gemini")

    gemini_model = st.text_input(
        "Gemini model",
        value="gemini-1.5-flash"
    )

    use_gemini = st.checkbox("Enable Gemini", value=True)

# INPUT
clinical = st.text_area(
    "Clinical Information",
    placeholder="Age, symptoms, duration, laterality..."
)

uploads = st.file_uploader(
    "Upload retinal images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

run = st.button("Analyze")

# HELPER
def encode_image(file):

    return {
        "type": "image_url",
        "image_url": {
            "url": "data:" + file.type + ";base64," +
            base64.b64encode(file.getvalue()).decode()
        }
    }


def parse_json(text):

    try:
        return json.loads(text)
    except:

        import re

        m = re.search(r"\{.*\}", text, re.S)

        if m:
            try:
                return json.loads(m.group())
            except:
                return None

    return None


# OPENAI CALL
def call_openai():

    content = [{
        "type": "text",
        "text": f"""
You are a retina specialist.

Analyze the retinal image.

Return STRICT JSON with:

modalities_detected
image_quality
key_findings
top_diagnoses
"""
    }]

    for f in uploads:

        content.append(encode_image(f))

    resp = openai_client.chat.completions.create(

        model=openai_model,

        messages=[{
            "role": "user",
            "content": content
        }],

        temperature=0
    )

    raw = resp.choices[0].message.content

    return raw, parse_json(raw)


# GEMINI CALL
def call_gemini():

    prompt = f"""
You are a retina specialist.

Analyze retinal images.

Return STRICT JSON with:

modalities_detected
image_quality
key_findings
top_diagnoses
"""

    parts = [types.Part.from_text(prompt)]

    for f in uploads:

        parts.append(
            types.Part.from_bytes(
                data=f.getvalue(),
                mime_type=f.type
            )
        )

    resp = gemini_client.models.generate_content(

        model=gemini_model,

        contents=[
            types.Content(
                role="user",
                parts=parts
            )
        ],

        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json"
        )
    )

    raw = resp.text

    return raw, parse_json(raw)


# FINAL REPORT
def build_report(openai_json, gemini_json):

    payload = {

        "clinical": clinical,
        "openai": openai_json,
        "gemini": gemini_json
    }

    resp = openai_client.chat.completions.create(

        model=openai_reason_model,

        messages=[

            {
                "role": "system",
                "content": """
You are a retina specialist.

Create an educational imaging report.

Describe morphology first.
Then differential diagnosis.
"""
            },

            {
                "role": "user",
                "content": json.dumps(payload, indent=2)
            }
        ]
    )

    return resp.choices[0].message.content


# RUN
if run:

    if not uploads:

        st.error("Upload image")
        st.stop()

    st.write("Running OpenAI vision...")

    openai_raw, openai_json = call_openai()

    gemini_raw = None
    gemini_json = None

    if use_gemini and gemini_client:

        st.write("Running Gemini vision...")

        gemini_raw, gemini_json = call_gemini()

    st.write("Generating report...")

    report = build_report(openai_json, gemini_json)

    st.subheader("Final Report")

    st.write(report)

    st.subheader("Debug")

    col1, col2 = st.columns(2)

    with col1:

        st.write("OpenAI raw")

        st.code(openai_raw)

        st.write("OpenAI JSON")

        st.json(openai_json)

    with col2:

        st.write("Gemini raw")

        st.code(gemini_raw)

        st.write("Gemini JSON")

        st.json(gemini_json)
