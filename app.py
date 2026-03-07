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
GEMINI_MODEL = "gemini-1.5-flash-latest"
# Eğer bu da çalışmazsa bunu deneyebilirsin:
# GEMINI_MODEL = "gemini-1.5-pro-latest"


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
    if "current_top_dx" not in st.session_state:
        st.session_state.current_top_dx = ""

init_state()


# =========================
# Helpers
# =========================
def image_to_data_url(mime, data):
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def overlap(dx1, dx2):
    if not dx1 or not dx2:
        return False
    return normalize(dx1) == normalize(dx2)


def uploads_to_images(files):
    imgs = []
    for f in files or []:
        imgs.append({
            "name": f.name,
            "mime": f.type or "image/jpeg",
            "data": f.getvalue()
        })
    return imgs


def safe_json_extract(text):
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    return None


def to_jsonable(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes:{len(obj)}>"
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    if hasattr(obj, "model_dump"):
        try:
            return to_jsonable(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return to_jsonable(obj.dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return to_jsonable(vars(obj))
        except Exception:
            pass
    return str(obj)


# =========================
# Vision Calls
# =========================
def openai_vision(clinical, images):
    content = [{
        "type": "text",
        "text": f"""
Analyze retinal images and clinical info.

Return STRICT JSON:
{{
  "top_diagnosis": "",
  "differentials": [],
  "evidence": []
}}

Clinical info:
{clinical}
"""
    }]

    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(img["mime"], img["data"])
            }
        })

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0
        )
        txt = resp.choices[0].message.content or ""
        return safe_json_extract(txt)
    except Exception as e:
        st.warning(f"OpenAI vision error: {e}")
        return None


def gemini_vision(clinical, images):
    parts = [{
        "text": f"""
Analyze retinal images and clinical info.

Return STRICT JSON:
{{
  "top_diagnosis": "",
  "differentials": [],
  "evidence": []
}}

Clinical info:
{clinical}
"""
    }]

    for img in images:
        parts.append({
            "inline_data": {
                "mime_type": img["mime"],
                "data": base64.b64encode(img["data"]).decode()
            }
        })

    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[{"role": "user", "parts": parts}],
            config={"temperature": 0, "response_mime_type": "application/json"}
        )
        txt = getattr(resp, "text", None) or ""
        return safe_json_extract(txt)
    except Exception as e:
        st.warning(f"Gemini error: {e}")
        return None


# =========================
# Final Report
# =========================
def build_report(clinical, g, o, agree):
    payload = to_jsonable({
        "clinical": clinical,
        "gemini": g,
        "openai": o,
        "agreement": agree
    })

    system = """
You are a retina specialist.

Write the report in exactly this format:

1. Most likely diagnosis
- diagnosis in **bold**
- add confidence in parentheses: High / Moderate / Low

2. Key imaging findings
- concise bullet points

3. Differential diagnosis
- up to 3 items
- short explanation only

4. Management considerations
- brief and practical
- mention interventions only if clearly relevant

5. Suggested additional imaging
- only if useful
- otherwise write: None

End with one short line:
Educational interpretation only; correlate clinically.
"""

    resp = openai_client.chat.completions.create(
        model=OPENAI_ARBITER_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ],
        temperature=0
    )

    return resp.choices[0].message.content or ""


# =========================
# History
# =========================
def save_current_case_to_history():
    if st.session_state.report:
        item = {
            "case": st.session_state.case_id,
            "diagnosis": st.session_state.current_top_dx or "Unknown diagnosis",
            "clinical": st.session_state.clinical,
            "report": st.session_state.report
        }
        if not st.session_state.history or st.session_state.history[0]["report"] != item["report"]:
            st.session_state.history.insert(0, item)
            st.session_state.history = st.session_state.history[:5]


def load_case_from_history(index):
    item = st.session_state.history[index]
    st.session_state.report = item["report"]
    st.session_state.clinical = item["clinical"]
    st.session_state.current_top_dx = item["diagnosis"]


def new_case():
    save_current_case_to_history()
    st.session_state.case_id += 1
    st.session_state.report = ""
    st.session_state.clinical = ""
    st.session_state.images = []
    st.session_state.uploader_key += 1
    st.session_state.current_top_dx = ""


# =========================
# Layout
# =========================
left, right = st.columns([1, 3], gap="large")

with left:
    st.markdown("### Recent cases")

    if not st.session_state.history:
        st.caption("No previous cases yet.")
    else:
        for i, item in enumerate(st.session_state.history):
            st.markdown(
                f"""
                <div class='history-item'>
                    <div><strong>Case #{item["case"]}</strong></div>
                    <div class='muted'>{item["diagnosis"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button(f"Open Case #{item['case']}", key=f"open_case_{i}", use_container_width=True):
                load_case_from_history(i)
                st.rerun()

with right:
    st.markdown("<div class='big-title'>👁️ RetinaGPT</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    clinical = st.text_area(
        "Clinical info (optional)",
        placeholder="Age, sex, symptoms, duration, laterality, relevant history...",
        value=st.session_state.clinical,
        height=120
    )
    st.session_state.clinical = clinical

    uploads = st.file_uploader(
        "Upload retinal images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploads:
        cols = st.columns(2)
        for i, f in enumerate(uploads[:2]):
            with cols[i % 2]:
                st.image(f, use_container_width=True, caption=f.name)
        if len(uploads) > 2:
            st.caption(f"+ {len(uploads) - 2} more image(s) selected.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    analyze = st.button("🔎 Analyze", use_container_width=True, type="primary")

    if analyze:
        if not uploads:
            st.error("Please upload at least one image.")
        else:
            images = uploads_to_images(uploads)

            with st.spinner("Analyzing..."):
                g = gemini_vision(clinical, images)
                o = openai_vision(clinical, images)

                if not g and not o:
                    st.error("Both Gemini and OpenAI failed.")
                    st.stop()

                if not g:
                    g = {
                        "top_diagnosis": "UNCERTAIN",
                        "differentials": [],
                        "evidence": []
                    }

                if not o:
                    o = {
                        "top_diagnosis": "UNCERTAIN",
                        "differentials": [],
                        "evidence": []
                    }

                dx1 = (g or {}).get("top_diagnosis", "")
                dx2 = (o or {}).get("top_diagnosis", "")
                agree = overlap(dx1, dx2)

                report = build_report(clinical, g, o, agree)

                st.session_state.report = report
                st.session_state.agreement = agree
                st.session_state.current_top_dx = dx1 or dx2 or ""
                st.session_state.images = images

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Output
# =========================
if st.session_state.report:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    tag = "Agreement" if st.session_state.agreement else "Different opinions"

    st.markdown(
        f"<span class='pill'>{tag}</span> <span class='pill muted'>Case #{st.session_state.case_id}</span>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(st.session_state.report)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if st.button("🆕 Ask new patient", use_container_width=True):
        new_case()
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
