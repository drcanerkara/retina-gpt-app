import os
import json
import base64
import re
import streamlit as st

from openai import OpenAI
from google import genai


# =========================
# Page
# =========================
st.set_page_config(page_title="RetinaGPT", page_icon="👁️", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1.5rem; max-width: 1100px;}
      .big-title {font-size: 2rem; font-weight: 800; margin-bottom: 0.3rem;}
      .card {
          border: 1px solid rgba(0,0,0,0.08);
          border-radius: 14px;
          padding: 16px;
          background: white;
      }
      .divider {
          height: 1px;
          background: rgba(0,0,0,0.08);
          margin: 12px 0;
      }
      .pill {
          display:inline-block;
          padding: 4px 10px;
          border-radius: 999px;
          border: 1px solid rgba(0,0,0,0.10);
          font-size: 0.85rem;
      }
      .muted {color:#6b7280;}
      .history-item {
          padding: 10px 12px;
          border: 1px solid rgba(0,0,0,0.08);
          border-radius: 12px;
          margin-bottom: 8px;
          background: #fff;
      }
      @media (max-width: 768px) {
          .block-container {padding-top: 1rem; max-width: 100%;}
          .big-title {font-size: 1.7rem;}
      }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Keys
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY.")
    st.stop()

if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY.")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

OPENAI_VISION_MODEL = "gpt-4o"
OPENAI_ARBITER_MODEL = "gpt-4o-mini"
GEMINI_VISION_MODEL = "gemini-3-flash-preview"  # replace if needed


# =========================
# Session state
# =========================
def ss_init():
    if "case_id" not in st.session_state:
        st.session_state.case_id = 1
    if "clinical" not in st.session_state:
        st.session_state.clinical = ""
    if "images" not in st.session_state:
        st.session_state.images = []
    if "final_report" not in st.session_state:
        st.session_state.final_report = ""
    if "agreement" not in st.session_state:
        st.session_state.agreement = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 1
    if "case_history" not in st.session_state:
        st.session_state.case_history = []
    if "current_top_dx" not in st.session_state:
        st.session_state.current_top_dx = ""

ss_init()


def save_current_case_to_history():
    if st.session_state.final_report:
        item = {
            "case_id": st.session_state.case_id,
            "diagnosis": st.session_state.current_top_dx or "Unknown diagnosis",
            "clinical": st.session_state.clinical,
            "report": st.session_state.final_report
        }
        # avoid duplicate save of same case/report
        if not st.session_state.case_history or st.session_state.case_history[0]["report"] != item["report"]:
            st.session_state.case_history.insert(0, item)
            st.session_state.case_history = st.session_state.case_history[:5]


def reset_case():
    save_current_case_to_history()
    st.session_state.case_id += 1
    st.session_state.clinical = ""
    st.session_state.images = []
    st.session_state.final_report = ""
    st.session_state.agreement = None
    st.session_state.chat_history = []
    st.session_state.analysis_done = False
    st.session_state.uploader_key += 1
    st.session_state.current_top_dx = ""


def load_case_from_history(index: int):
    item = st.session_state.case_history[index]
    st.session_state.final_report = item["report"]
    st.session_state.clinical = item["clinical"]
    st.session_state.current_top_dx = item["diagnosis"]
    st.session_state.analysis_done = True
    st.session_state.chat_history = []


# =========================
# Helpers
# =========================
def b64_data_url(mime: str, data: bytes) -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def safe_json_extract(text: str):
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
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


def normalize_dx(dx: str) -> str:
    if not dx:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", dx.lower()).strip()


def overlap_top2(a: dict, b: dict) -> bool:
    a1 = normalize_dx((a or {}).get("top_diagnosis", ""))
    b1 = normalize_dx((b or {}).get("top_diagnosis", ""))
    if a1 and b1 and a1 == b1:
        return True
    a2 = [normalize_dx(x) for x in ((a or {}).get("top_differentials") or [])[:2]]
    b2 = [normalize_dx(x) for x in ((b or {}).get("top_differentials") or [])[:2]]
    return (a1 in b2) or (b1 in a2) or (set(a2) & set(b2))


VISION_JSON_SCHEMA = {
    "top_diagnosis": "string",
    "top_differentials": ["string", "string", "string"],
    "key_evidence": ["string", "string", "string"],
    "confidence": "LOW|MODERATE|HIGH",
    "requested_additional_info": ["string", "string", "string"]
}


def uploads_to_images(files):
    out = []
    for f in files or []:
        out.append({
            "name": f.name,
            "mime": f.type or "image/jpeg",
            "data": f.getvalue()
        })
    return out


# =========================
# Vision calls
# =========================
def call_openai_vision(clinical_text: str, images):
    content = [{
        "type": "text",
        "text": (
            "You are a retina specialist. Analyze the provided retinal images + clinical info.\n"
            "Return STRICT JSON ONLY, matching this key structure:\n"
            f"{json.dumps(VISION_JSON_SCHEMA, ensure_ascii=False)}\n\n"
            f"Clinical info:\n{clinical_text if clinical_text.strip() else '(none provided)'}"
        )
    }]

    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": b64_data_url(img["mime"], img["data"])}
        })

    resp = openai_client.chat.completions.create(
        model=OPENAI_VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0
    )
    raw = resp.choices[0].message.content or ""
    return raw, safe_json_extract(raw)


def call_gemini_vision(clinical_text: str, images):
    parts = [{
        "text": (
            "You are a retina specialist. Analyze the provided retinal images + clinical info.\n"
            "Return STRICT JSON ONLY, matching this key structure:\n"
            f"{json.dumps(VISION_JSON_SCHEMA, ensure_ascii=False)}\n\n"
            f"Clinical info:\n{clinical_text if clinical_text.strip() else '(none provided)'}"
        )
    }]

    for img in images:
        parts.append({
            "inline_data": {
                "mime_type": img["mime"],
                "data": base64.b64encode(img["data"]).decode("utf-8")
            }
        })

    resp = gemini_client.models.generate_content(
        model=GEMINI_VISION_MODEL,
        contents=[{"role": "user", "parts": parts}],
        config={"temperature": 0, "response_mime_type": "application/json"}
    )
    raw = getattr(resp, "text", None) or ""
    return raw, safe_json_extract(raw)


# =========================
# Final report
# =========================
def build_final_report(clinical_text: str, gemini_js: dict, openai_js: dict, agreement: bool):
    payload = to_jsonable({
        "clinical": clinical_text,
        "gemini_opinion": gemini_js,
        "openai_opinion": openai_js,
        "agreement": agreement
    })

    system = """
You are RetinaGPT Arbiter. Produce ONE final educational report.

Use exactly this format:

1) Most likely diagnosis:
- Write the diagnosis in **bold**
- Add confidence in parentheses: High / Moderate / Low

2) Key imaging findings:
- 3 to 6 concise bullet points
- Morphology-first
- Use retina terminology

3) Differential diagnosis:
- Up to 3 diagnoses
- Short explanation only

4) Management considerations:
- Keep this brief and practical
- Mention interventions only if clearly relevant and supported by the case pattern
- Avoid generic internal medicine advice unless directly relevant

5) Suggested additional imaging:
- Only include if genuinely useful
- If not needed, write: None

End with one short line:
Educational interpretation only; correlate clinically.
"""

    resp = openai_client.chat.completions.create(
        model=OPENAI_ARBITER_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}
        ],
        temperature=0
    )
    return resp.choices[0].message.content or ""


# =========================
# Chat
# =========================
def chat_reply(user_text: str):
    context_pack = to_jsonable({
        "case": st.session_state.case_id,
        "clinical": st.session_state.clinical,
        "final_report": st.session_state.final_report
    })

    system = """
You are RetinaGPT (educational). Continue discussion for the SAME case.
- Use the existing final report as the baseline.
- Keep outputs concise and retina-focused.
- If user asks what to upload next, recommend the single most useful modality and what to look for.
"""

    messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": "CASE CONTEXT:\n" + json.dumps(context_pack, ensure_ascii=False, indent=2)})

    for m in st.session_state.chat_history[-20:]:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_text})

    resp = openai_client.chat.completions.create(
        model=OPENAI_ARBITER_MODEL,
        messages=messages,
        temperature=0
    )
    return resp.choices[0].message.content or ""


# =========================
# Analysis pipeline
# =========================
def run_analysis():
    images = st.session_state.images
    clin = st.session_state.clinical

    with st.spinner("Analyzing..."):
        _, gem_js = call_gemini_vision(clin, images)
        _, oa_js = call_openai_vision(clin, images)

    if not gem_js:
        st.error("Gemini did not return valid JSON. Try again.")
        return

    if not oa_js:
        oa_js = {
            "top_diagnosis": "UNCERTAIN",
            "top_differentials": [],
            "key_evidence": [],
            "confidence": "LOW",
            "requested_additional_info": []
        }

    st.session_state.current_top_dx = gem_js.get("top_diagnosis", "") or oa_js.get("top_diagnosis", "")
    agree = overlap_top2(gem_js, oa_js)
    report = build_final_report(clin, gem_js, oa_js, agree)

    st.session_state.final_report = report
    st.session_state.agreement = agree
    st.session_state.analysis_done = True


# =========================
# Layout
# =========================
left_col, right_col = st.columns([1, 3], gap="large")

with left_col:
    st.markdown("### Recent cases")
    if st.session_state.case_history:
        for i, item in enumerate(st.session_state.case_history):
            st.markdown(
                f"""
                <div class="history-item">
                    <div><strong>Case #{item['case_id']}</strong></div>
                    <div class="muted">{item['diagnosis']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button(f"Open Case #{item['case_id']}", key=f"open_case_{i}", use_container_width=True):
                load_case_from_history(i)
                st.rerun()
    else:
        st.caption("No previous cases yet.")

with right_col:
    st.markdown('<div class="big-title">👁️ RetinaGPT</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    clinical = st.text_area(
        "Clinical info (optional)",
        placeholder="Age/sex, symptoms, duration, laterality, relevant history...",
        height=120,
        value=st.session_state.clinical
    )
    st.session_state.clinical = clinical or ""

    uploads = st.file_uploader(
        "Upload retinal images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploads:
        cols = st.columns(2)
        show = uploads[:2]
        for i, f in enumerate(show):
            with cols[i % 2]:
                st.image(f, caption=f.name, use_container_width=True)
        if len(uploads) > 2:
            st.caption(f"+ {len(uploads) - 2} more image(s) selected.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    analyze = st.button("🔎 Analyze", type="primary", use_container_width=True)

    if analyze:
        if not uploads:
            st.error("Please upload at least one image.")
        else:
            st.session_state.images = uploads_to_images(uploads)
            run_analysis()

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.final_report:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        tag = "✅ Agreement" if st.session_state.agreement else "⚠️ Disagreement"
        st.markdown(
            f'<span class="pill">{tag}</span> <span class="pill muted">Case #{st.session_state.case_id}</span>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown(st.session_state.final_report)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if st.button("🆕 Ask new patient", use_container_width=True):
            reset_case()
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.final_report:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💬 Follow-up chat")

        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        user_msg = st.chat_input("Ask a follow-up question…")

        if user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.write(user_msg)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    ans = chat_reply(user_msg)
                    st.write(ans)

            st.session_state.chat_history.append({"role": "assistant", "content": ans})

        st.markdown("</div>", unsafe_allow_html=True)
