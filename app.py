import os
import json
import base64
import re
import streamlit as st
from openai import OpenAI
from google import genai


# =========================
# Page config
# =========================
st.set_page_config(page_title="RetinaGPT", page_icon="👁️", layout="wide")

st.markdown(
    """
    <style>

    .block-container{
        padding-top:4.5rem;
        padding-bottom:1.5rem;
        max-width:1100px;
    }

    .big-title{
        font-size:2rem;
        font-weight:800;
        margin-bottom:0.5rem;
        line-height:1.2;
    }

    .card{
        border:1px solid rgba(0,0,0,0.08);
        border-radius:14px;
        padding:16px;
        background:white;
    }

    .divider{
        height:1px;
        background:rgba(0,0,0,0.08);
        margin:12px 0;
    }

    .pill{
        display:inline-block;
        padding:4px 10px;
        border-radius:999px;
        border:1px solid rgba(0,0,0,0.10);
        font-size:0.85rem;
    }

    .muted{
        color:#6b7280;
    }

    .history-item{
        padding:10px 12px;
        border:1px solid rgba(0,0,0,0.08);
        border-radius:12px;
        margin-bottom:8px;
        background:#fff;
    }

    @media (max-width:768px){
        .block-container{
            padding-top:5.5rem;
            max-width:100%;
        }
        .big-title{
            font-size:1.6rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# API Keys
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY")
    st.stop()

if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

OPENAI_VISION_MODEL = "gpt-4o"
OPENAI_ARBITER_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-1.5-flash"


# =========================
# Session State
# =========================
def init_state():

    if "case_id" not in st.session_state:
        st.session_state.case_id = 1

    if "clinical" not in st.session_state:
        st.session_state.clinical = ""

    if "images" not in st.session_state:
        st.session_state.images = []

    if "report" not in st.session_state:
        st.session_state.report = ""

    if "agreement" not in st.session_state:
        st.session_state.agreement = None

    if "history" not in st.session_state:
        st.session_state.history = []

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 1

init_state()


# =========================
# Helpers
# =========================
def image_to_data_url(mime, data):
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def normalize(text):
    return re.sub(r"[^a-z0-9]+"," ",text.lower()).strip()


def overlap(dx1,dx2):

    if not dx1 or not dx2:
        return False

    return normalize(dx1)==normalize(dx2)


def uploads_to_images(files):

    imgs=[]

    for f in files:
        imgs.append({
            "mime":f.type,
            "data":f.getvalue()
        })

    return imgs


# =========================
# Vision Calls
# =========================
def openai_vision(clinical,images):

    content=[{
        "type":"text",
        "text":f"""
Analyze retinal images and clinical info.

Return JSON:
{{
"top_diagnosis":"",
"differentials":[],
"evidence":[]
}}

Clinical info:
{clinical}
"""
    }]

    for img in images:

        content.append({
            "type":"image_url",
            "image_url":{
                "url":image_to_data_url(img["mime"],img["data"])
            }
        })

    resp=openai_client.chat.completions.create(
        model=OPENAI_VISION_MODEL,
        messages=[{"role":"user","content":content}],
        temperature=0
    )

    txt=resp.choices[0].message.content

    try:
        return json.loads(txt)
    except:
        return None


def gemini_vision(clinical,images):

    parts=[{
        "text":f"""
Analyze retinal images and clinical info.

Return JSON:
{{
"top_diagnosis":"",
"differentials":[],
"evidence":[]
}}

Clinical info:
{clinical}
"""
    }]

    for img in images:

        parts.append({
            "inline_data":{
                "mime_type":img["mime"],
                "data":base64.b64encode(img["data"]).decode()
            }
        })

    resp=gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[{"role":"user","parts":parts}]
    )

    txt=resp.text

    try:
        return json.loads(txt)
    except:
        return None


# =========================
# Final Report
# =========================
def build_report(clinical,g,o,agree):

    payload={
        "clinical":clinical,
        "gemini":g,
        "openai":o,
        "agreement":agree
    }

    system="""
You are a retina specialist.

Write report in this format:

1. Most likely diagnosis
(bold)

2. Key imaging findings

3. Differential diagnosis

4. Management considerations

5. Suggested additional imaging

Keep concise and clinical.
"""

    resp=openai_client.chat.completions.create(
        model=OPENAI_ARBITER_MODEL,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":json.dumps(payload)}
        ],
        temperature=0
    )

    return resp.choices[0].message.content


# =========================
# Reset case
# =========================
def new_case():

    if st.session_state.report:

        st.session_state.history.insert(0,{
            "case":st.session_state.case_id,
            "report":st.session_state.report
        })

        st.session_state.history=st.session_state.history[:5]

    st.session_state.case_id+=1
    st.session_state.report=""
    st.session_state.clinical=""
    st.session_state.images=[]
    st.session_state.uploader_key+=1


# =========================
# Layout
# =========================
left,right=st.columns([1,3])

# =========================
# Sidebar history
# =========================
with left:

    st.markdown("### Recent cases")

    if not st.session_state.history:
        st.caption("No previous cases yet.")

    for item in st.session_state.history:

        st.markdown(
            f"""
<div class='history-item'>
Case #{item["case"]}
</div>
""",
            unsafe_allow_html=True
        )


# =========================
# Main UI
# =========================
with right:

    st.markdown("<div class='big-title'>👁️ RetinaGPT</div>",unsafe_allow_html=True)

    st.markdown("<div class='card'>",unsafe_allow_html=True)

    clinical=st.text_area(
        "Clinical info (optional)",
        placeholder="Age, symptoms, duration, laterality...",
        value=st.session_state.clinical
    )

    st.session_state.clinical=clinical

    uploads=st.file_uploader(
        "Upload retinal images",
        type=["jpg","jpeg","png","webp"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploads:

        cols=st.columns(2)

        for i,f in enumerate(uploads[:2]):

            with cols[i%2]:
                st.image(f,use_container_width=True)

    st.markdown("<div class='divider'></div>",unsafe_allow_html=True)

    analyze=st.button("🔎 Analyze",use_container_width=True,type="primary")

    if analyze:

        if not uploads:

            st.error("Please upload image")

        else:

            images=uploads_to_images(uploads)

            with st.spinner("Analyzing..."):

                g=gemini_vision(clinical,images)
                o=openai_vision(clinical,images)

                dx1=g.get("top_diagnosis") if g else ""
                dx2=o.get("top_diagnosis") if o else ""

                agree=overlap(dx1,dx2)

                report=build_report(clinical,g,o,agree)

                st.session_state.report=report
                st.session_state.agreement=agree

    st.markdown("</div>",unsafe_allow_html=True)


# =========================
# Output
# =========================
if st.session_state.report:

    st.markdown("<div class='card'>",unsafe_allow_html=True)

    tag="Agreement" if st.session_state.agreement else "Different opinions"

    st.markdown(
        f"<span class='pill'>{tag}</span> <span class='pill muted'>Case #{st.session_state.case_id}</span>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='divider'></div>",unsafe_allow_html=True)

    st.markdown(st.session_state.report)

    st.markdown("<div class='divider'></div>",unsafe_allow_html=True)

    if st.button("🆕 Ask new patient",use_container_width=True):

        new_case()
        st.rerun()

    st.markdown("</div>",unsafe_allow_html=True)
