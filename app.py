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
st.set_page_config(page_title="RetinaGPT — Consensus", page_icon="👁️", layout="wide")
st.title("👁️ RetinaGPT — Consensus")
st.caption("Educational only. Not medical advice.")


# =========================
# Keys
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY (env or Streamlit secrets).")
    st.stop()

if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY (env or Streamlit secrets).")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# =========================
# Models (hardcoded; user doesn't need to see)
# =========================
OPENAI_VISION_MODEL = "gpt-4o"
OPENAI_ARBITER_MODEL = "gpt-4o-mini"

# Put the Gemini model name that works in your account
GEMINI_VISION_MODEL = "gemini-3-flash-preview"


# =========================
# Session state
# =========================
def ss_init():
    if "case_counter" not in st.session_state:
        st.session_state.case_counter = 1
    if "clinical" not in st.session_state:
        st.session_state.clinical = ""
    if "images" not in st.session_state:
        # each: {name, mime, data(bytes)}
        st.session_state.images = []
    if "last_report" not in st.session_state:
        st.session_state.last_report = ""
    if "last_needed" not in st.session_state:
        st.session_state.last_needed = []
    if "chat_history" not in st.session_state:
        # each: {"role": "user"|"assistant", "content": "..."}
        st.session_state.chat_history = []

ss_init()


def reset_case():
    st.session_state.case_counter += 1
    st.session_state.clinical = ""
    st.session_state.images = []
    st.session_state.last_report = ""
    st.session_state.last_needed = []
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


def uploader_to_images(files):
    out = []
    for f in files or []:
        out.append({
            "name": f.name,
            "mime": f.type or "image/jpeg",
            "data": f.getvalue()
        })
    return out


# =========================
# Model calls (Vision opinions)
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
# Arbiter (single final report)
# =========================
def build_final_report(clinical_text: str, gemini_js: dict, openai_js: dict, agreement: bool):
    payload = to_jsonable({
        "clinical": clinical_text,
        "gemini_opinion": gemini_js,
        "openai_opinion": openai_js,
        "agreement": agreement
    })

    system = """
You are RetinaGPT Arbiter.

Goal: produce ONE final educational report for a retina case.
- Morphology-first, then diagnosis.
- Use retina terminology.
- If agreement=true: present ONE most likely diagnosis with appropriate confidence.
- If disagreement: provide short ranked differential and 'Needed to confirm' (max 4).
- No JSON; write readable clinical English.

Structure:
1) Most likely diagnosis + Confidence
2) Key evidence (3-6 bullets)
3) Top differentials (up to 3)
4) Needed to confirm (up to 4)
5) Educational red flags (up to 3)
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
# Chat continuation (keeps context)
# =========================
def chat_reply(user_text: str):
    # Build a compact context pack
    context_pack = to_jsonable({
        "case": st.session_state.case_counter,
        "clinical": st.session_state.clinical,
        "images_count": len(st.session_state.images),
        "last_report": st.session_state.last_report,
        "suggested_needed": st.session_state.last_needed
    })

    # Use arbiter model for stable dialogue
    system = """
You are RetinaGPT (educational). Continue the conversation about the SAME case.

Rules:
- Use the existing case context (clinical info, uploaded images count, last report).
- If user says they have new imaging (e.g., OCT) they can upload it; suggest uploading and then re-run.
- If user asks to refine the diagnosis: update reasoning but remain cautious.
- Keep outputs concise and clinically structured.
"""

    messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": "CASE CONTEXT:\n" + json.dumps(context_pack, ensure_ascii=False, indent=2)})

    # Add prior conversation
    for m in st.session_state.chat_history[-20:]:
        messages.append({"role": m["role"], "content": m["content"]})

    # Add new user message
    messages.append({"role": "user", "content": user_text})

    resp = openai_client.chat.completions.create(
        model=OPENAI_ARBITER_MODEL,
        messages=messages,
        temperature=0
    )
    return resp.choices[0].message.content or ""


# =========================
# Pipeline
# =========================
def run_analysis_pipeline():
    images = st.session_state.images
    clin = st.session_state.clinical

    with st.spinner("Running Gemini + OpenAI..."):
        gem_raw, gem_js = call_gemini_vision(clin, images)
        oa_raw, oa_js = call_openai_vision(clin, images)

    if not gem_js:
        st.error("Gemini did not return valid JSON (parsing failed). Try re-run.")
        st.stop()

    if not oa_js:
        oa_js = {
            "top_diagnosis": "UNCERTAIN",
            "top_differentials": [],
            "key_evidence": [],
            "confidence": "LOW",
            "requested_additional_info": ["Re-run OpenAI or provide additional imaging."]
        }

    agree = overlap_top2(gem_js, oa_js)
    report = build_final_report(clin, gem_js, oa_js, agree)

    needed = list(dict.fromkeys(
        (gem_js.get("requested_additional_info") or []) +
        (oa_js.get("requested_additional_info") or [])
    ))

    st.session_state.last_report = report
    st.session_state.last_needed = needed


# =========================
# UI Layout (Tabs)
# =========================
tab1, tab2 = st.tabs(["🩺 Case", "💬 Chat"])


with tab1:
    left, right = st.columns([1, 1])

    with left:
        if st.button("🆕 New patient / New case"):
            reset_case()
            st.rerun()

        st.subheader(f"Case #{st.session_state.case_counter}")

        clinical = st.text_area(
            "Clinical info (age/sex/symptoms/duration/laterality/history)",
            height=140,
            value=st.session_state.clinical,
            placeholder="Example: 58M, sudden painless vision loss OD 2 days, HTN..."
        )

        new_uploads = st.file_uploader(
            "Upload retinal images (fundus/OCT/FA/FAF/OCTA) — JPG/PNG/WEBP",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True
        )

        btn1, btn2, btn3 = st.columns([1, 1, 1])
        with btn1:
            analyze = st.button("🔎 Analyze", type="primary", use_container_width=True)
        with btn2:
            rerun = st.button("🔄 Re-run analysis", use_container_width=True)
        with btn3:
            add_imgs = st.button("➕ Add images to case", use_container_width=True)

        # Save clinical always
        st.session_state.clinical = clinical or ""

        if add_imgs:
            if not new_uploads:
                st.warning("No new images selected.")
            else:
                st.session_state.images.extend(uploader_to_images(new_uploads))
                st.success(f"Added {len(new_uploads)} image(s) to this case.")

        if analyze:
            if not new_uploads and not st.session_state.images:
                st.error("Please upload at least 1 image.")
                st.stop()

            # On first analyze, if user selected new_uploads, use those as initial set
            if new_uploads:
                st.session_state.images = uploader_to_images(new_uploads)

            run_analysis_pipeline()
            st.success("Analysis completed.")

        if rerun:
            if new_uploads:
                # If user uploads, append by default (more natural for OCT add)
                st.session_state.images.extend(uploader_to_images(new_uploads))

            if not st.session_state.images:
                st.error("No images in case. Upload first.")
                st.stop()

            run_analysis_pipeline()
            st.success("Re-run completed.")

    with right:
        st.subheader("Final Report")
        if st.session_state.last_report:
            st.write(st.session_state.last_report)

            st.subheader("Suggested additional imaging / info (if needed)")
            if st.session_state.last_needed:
                for x in st.session_state.last_needed[:8]:
                    st.write(f"• {x}")
            else:
                st.write("None.")
        else:
            st.info("Run **Analyze** to generate the first report.")

        st.subheader("Case images (current)")
        if st.session_state.images:
            # Show thumbnails
            thumbs = st.columns(3)
            for i, img in enumerate(st.session_state.images[:12]):
                with thumbs[i % 3]:
                    st.image(img["data"], caption=img["name"], use_container_width=True)
            if len(st.session_state.images) > 12:
                st.caption(f"+ {len(st.session_state.images) - 12} more image(s) in this case.")
        else:
            st.write("No images yet.")


with tab2:
    st.subheader("Chat (continue this case)")

    # Render conversation
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_msg = st.chat_input("Type a message… (e.g., 'I have OCT; what should I look for?' )")

    if user_msg:
        # Store user message
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ans = chat_reply(user_msg)
                st.write(ans)

        st.session_state.chat_history.append({"role": "assistant", "content": ans})
