import os
import json
import base64
import re
import streamlit as st

from openai import OpenAI
from google import genai

# =========================
# Sayfa Konfigürasyonu (Geniş Mod)
# =========================
st.set_page_config(page_title="RetinaGPT", page_icon="👁️", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .big-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem;}
      .card {border: 1px solid rgba(0,0,0,0.08); border-radius: 14px; padding: 20px; background: white; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);}
      .divider {height: 1px; background: rgba(0,0,0,0.08); margin: 15px 0;}
      .pill {display:inline-block; padding: 4px 12px; border-radius: 999px; border: 1px solid rgba(0,0,0,0.10); font-size: 0.85rem; font-weight: 600;}
      .muted {color:#6b7280;}
      /* Bullet pointleri sağa yanaştırmak için stil */
      .report-text ul { padding-left: 40px; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# API Keys & Clients
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)

if not (OPENAI_API_KEY and GEMINI_API_KEY):
    st.error("API Anahtarları eksik. Lütfen ortam değişkenlerini kontrol edin.")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

OPENAI_VISION_MODEL = "gpt-4o"
OPENAI_ARBITER_MODEL = "gpt-4o-mini"
GEMINI_VISION_MODEL = "gemini-3-flash-preview"

# =========================
# Session State & Sidebar (History)
# =========================
if "history_list" not in st.session_state:
    st.session_state.history_list = []

def ss_init():
    if "case_id" not in st.session_state: st.session_state.case_id = 1
    if "clinical" not in st.session_state: st.session_state.clinical = ""
    if "images" not in st.session_state: st.session_state.images = []
    if "final_report" not in st.session_state: st.session_state.final_report = ""
    if "agreement" not in st.session_state: st.session_state.agreement = None
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "uploader_key" not in st.session_state: st.session_state.uploader_key = 1

ss_init()

with st.sidebar:
    st.markdown("### 📂 Case History")
    if not st.session_state.history_list:
        st.caption("No history yet.")
    for h in st.session_state.history_list:
        with st.expander(f"Case #{h['id']} - {h['dx'][:20]}..."):
            st.write(h['report'])
    
    if st.button("🗑️ Clear History"):
        st.session_state.history_list = []
        st.rerun()

def reset_case():
    st.session_state.case_id += 1
    st.session_state.clinical = ""
    st.session_state.images = []
    st.session_state.final_report = ""
    st.session_state.chat_history = []
    st.session_state.uploader_key += 1

# =========================
# Helpers
# =========================
def b64_data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"

def safe_json_extract(text: str):
    try: return json.loads(text)
    except:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try: return json.loads(m.group(0))
            except: return None
    return None

def normalize_dx(dx: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (dx or "").lower()).strip()

def overlap_top2(a: dict, b: dict) -> bool:
    a1, b1 = normalize_dx((a or {}).get("top_diagnosis")), normalize_dx((b or {}).get("top_diagnosis"))
    if a1 and b1 and a1 == b1: return True
    a2 = [normalize_dx(x) for x in ((a or {}).get("top_differentials") or [])[:2]]
    b2 = [normalize_dx(x) for x in ((b or {}).get("top_differentials") or [])[:2]]
    return (a1 in b2) or (b1 in a2) or (set(a2) & set(b2))

def uploads_to_images(files):
    return [{"name": f.name, "mime": f.type or "image/jpeg", "data": f.getvalue()} for f in files or []]

# =========================
# Vision & Arbiter Calls
# =========================
VISION_JSON_SCHEMA = {
    "top_diagnosis": "string", "top_differentials": ["string"], "key_evidence": ["string"], "confidence": "LOW|MODERATE|HIGH"
}

def call_openai_vision(clinical_text: str, images):
    content = [{"type": "text", "text": f"Retina specialist analysis. JSON ONLY. Schema: {json.dumps(VISION_JSON_SCHEMA)}. Clinical: {clinical_text}"}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": b64_data_url(img['mime'], img['data'])}})
    resp = openai_client.chat.completions.create(model=OPENAI_VISION_MODEL, messages=[{"role": "user", "content": content}], temperature=0)
    return safe_json_extract(resp.choices[0].message.content)

def call_gemini_vision(clinical_text: str, images):
    parts = [{"text": f"Retina specialist analysis. JSON ONLY. Schema: {json.dumps(VISION_JSON_SCHEMA)}. Clinical: {clinical_text}"}]
    for img in images:
        parts.append({"inline_data": {"mime_type": img["mime"], "data": base64.b64encode(img["data"]).decode("utf-8")}})
    resp = gemini_client.models.generate_content(model=GEMINI_VISION_MODEL, contents=[{"role": "user", "parts": parts}], config={"temperature": 0, "response_mime_type": "application/json"})
    return safe_json_extract(resp.text)

def build_final_report(clinical_text: str, gemini_js: dict, openai_js: dict, agreement: bool):
    system = """
    You are RetinaGPT Arbiter. Produce a medical report with these STICKT rules:
    - Use headers like **1)**, **2)**, **3)**, **4)**, **5)**. (Bold numbers)
    - All bullet points MUST be indented with spaces to appear shifted to the right.
    - Section 1 must start with: "1) Most likely diagnosis: **[DIAGNOSIS NAME]**"
    - Morphology-first reasoning. Use retina subspecialty terms.
    """
    payload = {"clinical": clinical_text, "gemini": gemini_js, "openai": openai_js, "agreement": agreement}
    resp = openai_client.chat.completions.create(model=OPENAI_ARBITER_MODEL, messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload)}], temperature=0)
    return resp.choices[0].message.content

# =========================
# Main UI
# =========================
col_main, col_spacer = st.columns([3, 1])

with col_main:
    st.markdown('<div class="big-title">👁️ RetinaGPT</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        clinical = st.text_area("Clinical info", placeholder="Age, symptoms, history...", height=100, value=st.session_state.clinical)
        st.session_state.clinical = clinical
        
        uploads = st.file_uploader("Upload retinal images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key=f"up_{st.session_state.uploader_key}")
        
        if uploads:
            cols = st.columns(4)
            for i, f in enumerate(uploads[:4]):
                cols[i].image(f, use_container_width=True)
        
        if st.button("🔎 Analyze Case", type="primary", use_container_width=True):
            if not uploads:
                st.error("Please upload images.")
            else:
                st.session_state.images = uploads_to_images(uploads)
                with st.spinner("Consulting AI Specialists..."):
                    g_js = call_gemini_vision(clinical, st.session_state.images)
                    o_js = call_openai_vision(clinical, st.session_state.images)
                    agree = overlap_top2(g_js, o_js)
                    report = build_final_report(clinical, g_js, o_js, agree)
                    
                    st.session_state.final_report = report
                    st.session_state.agreement = agree
                    # Add to history
                    st.session_state.history_list.append({"id": st.session_state.case_id, "dx": g_js.get("top_diagnosis", "Unknown"), "report": report})
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Output & Chat
# =========================
if st.session_state.final_report:
    st.markdown('<div class="card report-text">', unsafe_allow_html=True)
    tag = "✅ Agreement" if st.session_state.agreement else "⚠️ Disagreement"
    st.markdown(f'<span class="pill">{tag}</span> <span class="pill muted">Case #{st.session_state.case_id}</span>', unsafe_allow_html=True)
    st.markdown(st.session_state.final_report)
    
    if st.button("🆕 New Patient", use_container_width=True):
        reset_case()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 Clinical Chat & Follow-up")
    
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]): st.write(m["content"])

    # Yeni Görüntü Ekleme Butonu (+)
    with st.expander("➕ Add Additional Imaging (e.g. OCT) to this Chat"):
        extra_file = st.file_uploader("Upload follow-up scan", type=["jpg", "png"], key="extra_up")

    user_msg = st.chat_input("Ask a question about the case or the new scan...")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"): st.write(user_msg)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Burada normalde chat_reply fonksiyonun çağrılır. 
                # Eğer extra_file varsa, arka planda vision modeline tekrar gönderilebilir.
                ans = "I have received your message. (Logic: Here the AI will analyze the report and any new images uploaded in the '+' section.)"
                st.write(ans)
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
    st.markdown('</div>', unsafe_allow_html=True)
