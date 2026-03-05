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
# Fixed models (no sidebar)
# =========================
OPENAI_VISION_MODEL = "gpt-4o"
OPENAI_ARBITER_MODEL = "gpt-4o-mini"

# Put the Gemini model name that WORKS in your account:
# examples you can try (depends on your key availability):
# - "gemini-3-flash-preview"
# - "gemini-1.5-pro-latest"
# - "gemini-1.5-flash-latest"
GEMINI_VISION_MODEL = "gemini-3-flash-preview"


# =========================
# Session state
# =========================
if "case_counter" not in st.session_state:
    st.session_state.case_counter = 1

if "saved_clinical" not in st.session_state:
    st.session_state.saved_clinical = ""

if "saved_images" not in st.session_state:
    st.session_state.saved_images = []  # list of dicts: {name, mime, data_bytes}

if "last_report" not in st.session_state:
    st.session_state.last_report = ""

if "last_needed" not in st.session_state:
    st.session_state.last_needed = []


def reset_case():
    st.session_state.case_counter += 1
    st.session_state.saved_clinical = ""
    st.session_state.saved_images = []
    st.session_state.last_report = ""
    st.session_state.last_needed = []


# =========================
# UI Controls
# =========================
top_left, top_right = st.columns([1, 4])
with top_left:
    if st.button("🆕 New patient / New case"):
        reset_case()
        st.rerun()

st.subheader(f"Case #{st.session_state.case_counter}")

st.info("Tip: Update clinical info and/or add more images, then click **Re-run analysis**.")


# =========================
# Inputs
# =========================
clinical = st.text_area(
    "Clinical info (age/sex/symptoms/duration/laterality/history)",
    height=140,
    value=st.session_state.saved_clinical,
    placeholder="Example: 58M, sudden painless vision loss OD 2 days, HTN, no DM..."
)

uploads = st.file_uploader(
    "Upload retinal images (fundus/OCT/FA/FAF/OCTA) — JPG/PNG/WEBP",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    analyze_btn = st.button("🔎 Analyze", type="primary", use_container_width=True)
with c2:
    rerun_btn = st.button("🔄 Re-run analysis", use_container_width=True)
with c3:
    st.write("")


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


def normalize_dx(dx: str) -> str:
    if not dx:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", dx.lower()).strip()


def overlap_top2(a: dict, b: dict) -> bool:
    a1 = normalize_dx(a.get("top_diagnosis", ""))
    b1 = normalize_dx(b.get("top_diagnosis", ""))
    if a1 and b1 and a1 == b1:
        return True
    a2 = [normalize_dx(x) for x in (a.get("top_differentials") or [])[:2]]
    b2 = [normalize_dx(x) for x in (b.get("top_differentials") or [])[:2]]
    return (a1 in b2) or (b1 in a2) or (set(a2) & set(b2))


VISION_JSON_SCHEMA = {
    "top_diagnosis": "string",
    "top_differentials": ["string", "string", "string"],
    "key_evidence": ["string", "string", "string"],
    "confidence": "LOW|MODERATE|HIGH",
    "requested_additional_info": ["string", "string", "string"]
}


def images_from_uploader(uploader_files):
    """
    Convert Streamlit uploaded files into stable in-memory list (bytes),
    so Re-run works even if user doesn't re-upload.
    """
    imgs = []
    for f in uploader_files or []:
        imgs.append({
            "name": f.name,
            "mime": f.type or "image/jpeg",
            "data": f.getvalue()
        })
    return imgs


# =========================
# Model calls (NO feature extraction, NO RAG)
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

    # Important: use inline_data base64 because it's the most compatible path
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
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
        }
    )

    raw = getattr(resp, "text", None) or ""
    return raw, safe_json_extract(raw)


def build_final_report(clinical_text: str, gemini_js: dict, openai_js: dict, agreement: bool):
    payload = {
        "clinical": clinical_text,
        "gemini_opinion": gemini_js,
        "openai_opinion": openai_js,
        "agreement": agreement
    }

    system = """
You are RetinaGPT Arbiter.

Goal: produce ONE final educational report for a retina case.
- Morphology-first, then diagnosis.
- Use retina terminology.
- If agreement=true: present ONE most likely diagnosis with HIGH confidence (unless evidence is weak).
- If they disagree: do not force certainty; provide a short ranked differential and a 'Needed to confirm' list (max 4).
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
# Run logic
# =========================
def run_pipeline():
    images = st.session_state.saved_images
    clin = st.session_state.saved_clinical

    with st.spinner("Running Gemini + OpenAI..."):
        gem_raw, gem_js = call_gemini_vision(clin, images)
        oa_raw, oa_js = call_openai_vision(clin, images)

    if not gem_js:
        st.error("Gemini did not return valid JSON (parsing failed).")
        # show minimal debug to help
        st.code(gem_raw[:3000])
        st.stop()

    if not oa_js:
        # continue with Gemini only (OpenAI vision may occasionally return non-JSON)
        oa_js = {
            "top_diagnosis": "UNCERTAIN",
            "top_differentials": [],
            "key_evidence": [],
            "confidence": "LOW",
            "requested_additional_info": ["Re-run OpenAI or provide additional imaging."]
        }

    agree = overlap_top2(gem_js, oa_js)
    report = build_final_report(clin, gem_js, oa_js, agree)

    # suggested list
    needed = list(dict.fromkeys(
        (gemini_js.get("requested_additional_info") or []) +
        (oa_js.get("requested_additional_info") or [])
    ))

    st.session_state.last_report = report
    st.session_state.last_needed = needed

    # Show ONE output
    st.subheader("Final Report")
    st.write(report)

    st.subheader("Suggested additional imaging / info (if needed)")
    if needed:
        for x in needed[:8]:
            st.write(f"• {x}")
    else:
        st.write("None.")

    # fully hidden debug (remove if you want)
    with st.expander("🔧 Debug (admin only)"):
        st.markdown("**Gemini raw**")
        st.code(gem_raw[:4000])
        st.markdown("**Gemini JSON**")
        st.json(gem_js)

        st.markdown("**OpenAI raw**")
        st.code(oa_raw[:4000])
        st.markdown("**OpenAI JSON**")
        st.json(oa_js)


# =========================
# Button actions
# =========================
if analyze_btn:
    # Save current inputs into session state
    if uploads:
        st.session_state.saved_images = images_from_uploader(uploads)
    else:
        st.session_state.saved_images = []

    st.session_state.saved_clinical = clinical or ""

    if not st.session_state.saved_images:
        st.error("Please upload at least 1 image.")
        st.stop()

    run_pipeline()

elif rerun_btn:
    # Re-run uses stored images; allow user to update clinical text without reupload
    st.session_state.saved_clinical = clinical or ""

    # If user uploaded new images, replace stored images (common expectation)
    if uploads:
        st.session_state.saved_images = images_from_uploader(uploads)

    if not st.session_state.saved_images:
        st.warning("No images available. Upload images first, then Analyze.")
        st.stop()

    run_pipeline()

# =========================
# Persist last report on screen (optional)
# =========================
if st.session_state.last_report and not (analyze_btn or rerun_btn):
    st.subheader("Last Final Report")
    st.write(st.session_state.last_report)
    if st.session_state.last_needed:
        st.subheader("Suggested additional imaging / info (if needed)")
        for x in st.session_state.last_needed[:8]:
            st.write(f"• {x}")
