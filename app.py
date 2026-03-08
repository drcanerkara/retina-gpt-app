import os
import json
import base64
import re
import time
import streamlit as st

from openai import OpenAI
from google import genai


# =========================
# Page
# =========================
st.set_page_config(page_title="RetinaGPT", page_icon="👁️", layout="centered")

st.markdown(
    """
    <style>
      .block-container {
          padding-top: 2rem;
          padding-bottom: 2rem;
          max-width: 920px;
      }
      .big-title {
          font-size: 2.0rem;
          font-weight: 800;
          margin-bottom: 0.25rem;
      }
      .card {
          border: 1px solid rgba(0,0,0,0.08);
          border-radius: 14px;
          padding: 16px;
          background: white;
      }
      .divider {
          height: 1px;
          background: rgba(0,0,0,0.08);
          margin: 14px 0;
      }
      .pill {
          display: inline-block;
          padding: 4px 10px;
          border-radius: 999px;
          border: 1px solid rgba(0,0,0,0.10);
          font-size: 0.85rem;
      }
      .muted {
          color: #6b7280;
      }
      .subtle-box {
          border-left: 3px solid rgba(0,0,0,0.08);
          padding-left: 12px;
          margin-top: 8px;
      }
      .small-note {
          font-size: 0.9rem;
          color: #6b7280;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">👁️ RetinaGPT</div>', unsafe_allow_html=True)
st.caption("Structured clinical input improves multimodal retinal diagnostic reasoning.")


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


# =========================
# Models
# =========================
OPENAI_VISION_MODEL = "gpt-4o"
OPENAI_ARBITER_MODEL = "gpt-4o-mini"
GEMINI_VISION_MODEL = "gemini-3-flash-preview"
# Daha stabil denemek istersen:
# GEMINI_VISION_MODEL = "gemini-1.5-pro"


# =========================
# Session state
# =========================
def ss_init():
    defaults = {
        "case_id": 1,
        "images": [],
        "final_report": "",
        "report_history": [],
        "agreement": None,
        "confidence_label": "",
        "confidence_icon": "",
        "chat_history": [],
        "analysis_done": False,
        "uploader_key": 1,
        "last_error_type": None,
        "last_error_raw": "",
        "additional_clinical_note": "",
        "add_uploader_key": 1,
        "additional_modality": "OCT",
        # structured clinical fields
        "age": None,
        "sex": "Select",
        "laterality": "Select",
        "visual_acuity": "",
        "primary_symptom": "",
        "duration": "",
        "relevant_history": "",
        "additional_notes": "",
        # frozen text sent to models
        "clinical": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


ss_init()


def reset_case():
    st.session_state.case_id += 1
    st.session_state.images = []
    st.session_state.final_report = ""
    st.session_state.report_history = []
    st.session_state.agreement = None
    st.session_state.confidence_label = ""
    st.session_state.confidence_icon = ""
    st.session_state.chat_history = []
    st.session_state.analysis_done = False
    st.session_state.uploader_key += 1
    st.session_state.last_error_type = None
    st.session_state.last_error_raw = ""
    st.session_state.additional_clinical_note = ""
    st.session_state.add_uploader_key += 1
    st.session_state.additional_modality = "OCT"

    st.session_state.age = None
    st.session_state.sex = "Select"
    st.session_state.laterality = "Select"
    st.session_state.visual_acuity = ""
    st.session_state.primary_symptom = ""
    st.session_state.duration = ""
    st.session_state.relevant_history = ""
    st.session_state.additional_notes = ""
    st.session_state.clinical = ""


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
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
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


def normalize_confidence(value: str) -> str:
    if not value:
        return "LOW"
    v = str(value).strip().upper()
    if v in {"HIGH", "MODERATE", "LOW"}:
        return v
    return "LOW"


def overlap_top2(a: dict, b: dict) -> bool:
    a1 = normalize_dx((a or {}).get("top_diagnosis", ""))
    b1 = normalize_dx((b or {}).get("top_diagnosis", ""))

    if a1 and b1 and a1 == b1:
        return True

    a2 = [normalize_dx(x) for x in ((a or {}).get("top_differentials") or [])[:2]]
    b2 = [normalize_dx(x) for x in ((b or {}).get("top_differentials") or [])[:2]]

    return (a1 in b2) or (b1 in a2) or bool(set(a2) & set(b2))


def derive_confidence_badge(gemini_js: dict, openai_js: dict, agreement: bool):
    gem_conf = normalize_confidence((gemini_js or {}).get("confidence", "LOW"))
    oa_conf = normalize_confidence((openai_js or {}).get("confidence", "LOW"))

    if agreement:
        if gem_conf == "HIGH" and oa_conf == "HIGH":
            return "High", "🟢"
        return "Moderate", "🟡"

    return "Low", "🔴"


VISION_JSON_SCHEMA = {
    "top_diagnosis": "string",
    "top_differentials": ["string", "string", "string"],
    "key_evidence": ["string", "string", "string"],
    "confidence": "LOW|MODERATE|HIGH",
    "requested_additional_info": ["string", "string", "string"],
}


def uploads_to_images(files):
    out = []
    for f in files or []:
        out.append(
            {
                "name": f.name,
                "mime": f.type or "image/jpeg",
                "data": f.getvalue(),
            }
        )
    return out


def merge_images(old_images, new_images):
    merged = list(old_images or [])
    merged.extend(new_images or [])
    return merged


def get_current_assessment_label():
    if len(st.session_state.report_history) == 0:
        return "Initial assessment"
    return f"Updated assessment {len(st.session_state.report_history)}"


def build_clinical_summary():
    lines = []

    if st.session_state.age is not None:
        lines.append(f"Age: {st.session_state.age}")
    if st.session_state.sex and st.session_state.sex != "Select":
        lines.append(f"Sex: {st.session_state.sex}")
    if st.session_state.laterality and st.session_state.laterality != "Select":
        lines.append(f"Laterality: {st.session_state.laterality}")
    if st.session_state.visual_acuity.strip():
        lines.append(f"Visual acuity: {st.session_state.visual_acuity.strip()}")
    if st.session_state.primary_symptom.strip():
        lines.append(f"Primary symptom: {st.session_state.primary_symptom.strip()}")
    if st.session_state.duration.strip():
        lines.append(f"Duration: {st.session_state.duration.strip()}")
    if st.session_state.relevant_history.strip():
        lines.append(f"Relevant history: {st.session_state.relevant_history.strip()}")
    if st.session_state.additional_notes.strip():
        lines.append(f"Additional notes: {st.session_state.additional_notes.strip()}")

    return "\n".join(lines).strip()


# =========================
# Vision calls
# =========================
def call_openai_vision(clinical_text: str, images):
    content = [
        {
            "type": "text",
            "text": (
                "You are a retina specialist. Analyze the provided retinal images + clinical info.\n"
                "Return STRICT JSON ONLY, matching this key structure:\n"
                f"{json.dumps(VISION_JSON_SCHEMA, ensure_ascii=False)}\n\n"
                f"Clinical info:\n{clinical_text if clinical_text.strip() else '(none provided)'}"
            ),
        }
    ]

    for img in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": b64_data_url(img["mime"], img["data"])},
            }
        )

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        return raw, safe_json_extract(raw)
    except Exception as e:
        return f"OpenAI vision error: {str(e)}", None


def call_gemini_vision(clinical_text: str, images, max_retries: int = 3):
    parts = [
        {
            "text": (
                "You are a retina specialist. Analyze the provided retinal images + clinical info.\n"
                "Return STRICT JSON ONLY, matching this key structure:\n"
                f"{json.dumps(VISION_JSON_SCHEMA, ensure_ascii=False)}\n\n"
                f"Clinical info:\n{clinical_text if clinical_text.strip() else '(none provided)'}"
            )
        }
    ]

    for img in images:
        parts.append(
            {
                "inline_data": {
                    "mime_type": img["mime"],
                    "data": base64.b64encode(img["data"]).decode("utf-8"),
                }
            }
        )

    last_error = ""

    for attempt in range(max_retries):
        try:
            resp = gemini_client.models.generate_content(
                model=GEMINI_VISION_MODEL,
                contents=[{"role": "user", "parts": parts}],
                config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                },
            )
            raw = getattr(resp, "text", None) or ""
            parsed = safe_json_extract(raw)

            if parsed:
                return raw, parsed, None

            last_error = raw or "Gemini returned empty or invalid JSON."

        except Exception as e:
            last_error = f"Gemini vision error: {str(e)}"

            if (
                "503" in last_error
                or "UNAVAILABLE" in last_error.upper()
                or "high demand" in last_error.lower()
                or "busy" in last_error.lower()
            ):
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                return last_error, None, "busy"

            return last_error, None, "generic"

    return last_error, None, "busy"


# =========================
# Final report
# =========================
def build_final_report(clinical_text: str, gemini_js: dict, openai_js: dict, agreement: bool, is_update: bool):
    payload = to_jsonable(
        {
            "clinical": clinical_text,
            "gemini_opinion": gemini_js,
            "openai_opinion": openai_js,
            "agreement": agreement,
            "is_update": is_update,
        }
    )

    system = """
You are RetinaGPT Arbiter, a retina subspecialty clinical reasoning assistant.

Generate ONE final educational report based on the two model opinions.

FORMAT (Markdown)

**Most likely diagnosis**
- State the most probable diagnosis in **bold**
- Add confidence level (High / Moderate / Low)
- Brief justification

**Key imaging findings**
- Bullet list of the most relevant retinal imaging features
- Focus on morphology (RPE changes, ellipsoid zone status, SRF, IRF, PED, hemorrhage, vascular changes)

**Differential diagnosis**
- 2-4 most relevant alternatives
- Provide one short discriminating clue for each

**Management considerations**
- Evidence-based retina management considerations
- Mention observation vs treatment when relevant
- Include follow-up strategy if appropriate

**Suggested additional imaging**
- Recommend the most useful additional imaging modalities
- Examples: OCT, OCTA, FA, ICGA, FAF, wide-field imaging

STYLE
- Use retina subspecialty terminology
- Keep the report concise but clinically meaningful
- Prefer morphology-first reasoning
- Avoid unnecessary explanations

DECISION RULES
- Use the provided clinical information actively in the reasoning.
- If both models agree, present the diagnosis with higher confidence.
- If there is disagreement, still provide the most likely diagnosis but reflect uncertainty appropriately.
- Do not mention the internal model names.
- If this is an updated assessment after additional imaging, explicitly mention how the newly added imaging supports, refines, or changes the working diagnosis when relevant.
- In "Suggested additional imaging", do not recommend modalities that are already clearly available in the current case data.
- If newly added imaging already includes OCT, FAF, FA, OCTA, ICGA, or wide-field imaging, suggest only still-missing modalities that would provide additional diagnostic value.
"""

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_ARBITER_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"Final report generation failed: {str(e)}"


# =========================
# Chat continuation
# =========================
def chat_reply(user_text: str):
    context_pack = to_jsonable(
        {
            "case": st.session_state.case_id,
            "clinical": st.session_state.clinical,
            "final_report": st.session_state.final_report,
            "report_history": st.session_state.report_history,
            "confidence_label": st.session_state.confidence_label,
        }
    )

    system = """
You are RetinaGPT (educational). Continue discussion for the SAME case.
- Use the latest final report as the baseline.
- Use the structured clinical information as part of the case context.
- If the user asks what to upload next, suggest the single most useful modality and what to look for.
- Keep outputs concise and structured.
- Use retina subspecialty terminology.
"""

    messages = [{"role": "system", "content": system}]
    messages.append(
        {
            "role": "user",
            "content": "CASE CONTEXT:\n" + json.dumps(context_pack, ensure_ascii=False, indent=2),
        }
    )

    for m in st.session_state.chat_history[-20:]:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_text})

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_ARBITER_MODEL,
            messages=messages,
            temperature=0,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"Chat reply failed: {str(e)}"


# =========================
# Analysis pipeline
# =========================
def run_analysis():
    images = st.session_state.images
    clin = st.session_state.clinical

    st.session_state.last_error_type = None
    st.session_state.last_error_raw = ""

    with st.spinner("Analyzing..."):
        gem_raw, gem_js, gem_error_type = call_gemini_vision(clin, images)
        _, oa_js = call_openai_vision(clin, images)

    if not gem_js:
        st.session_state.last_error_type = gem_error_type or "generic"
        st.session_state.last_error_raw = gem_raw or ""
        st.session_state.final_report = ""
        st.session_state.agreement = None
        st.session_state.confidence_label = ""
        st.session_state.confidence_icon = ""
        st.session_state.analysis_done = False
        return

    if not oa_js:
        oa_js = {
            "top_diagnosis": "UNCERTAIN",
            "top_differentials": [],
            "key_evidence": [],
            "confidence": "LOW",
            "requested_additional_info": [],
        }

    agree = overlap_top2(gem_js, oa_js)
    confidence_label, confidence_icon = derive_confidence_badge(gem_js, oa_js, agree)

    label = get_current_assessment_label()
    is_update = len(st.session_state.report_history) > 0

    report = build_final_report(clin, gem_js, oa_js, agree, is_update)

    st.session_state.final_report = report
    st.session_state.report_history.append(
        {
            "label": label,
            "report": report,
            "confidence_label": confidence_label,
            "confidence_icon": confidence_icon,
        }
    )
    st.session_state.agreement = agree
    st.session_state.confidence_label = confidence_label
    st.session_state.confidence_icon = confidence_icon
    st.session_state.analysis_done = True


# =========================
# UI - structured clinical input
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Clinical information")

col1, col2, col3 = st.columns(3)

with col1:
    age_default = st.session_state.age if st.session_state.age is not None else 0
    age = st.number_input(
        "Age *",
        min_value=0,
        max_value=120,
        value=age_default,
        step=1,
    )

with col2:
    sex = st.selectbox(
        "Sex *",
        ["Select", "Male", "Female", "Other"],
        index=["Select", "Male", "Female", "Other"].index(
            st.session_state.sex if st.session_state.sex in ["Select", "Male", "Female", "Other"] else "Select"
        ),
    )

with col3:
    laterality = st.selectbox(
        "Laterality *",
        ["Select", "Unilateral", "Bilateral"],
        index=["Select", "Unilateral", "Bilateral"].index(
            st.session_state.laterality if st.session_state.laterality in ["Select", "Unilateral", "Bilateral"] else "Select"
        ),
    )

col4, col5 = st.columns(2)

with col4:
    visual_acuity = st.text_input(
        "Visual acuity (optional)",
        value=st.session_state.visual_acuity,
        placeholder="e.g. 20/40, 0.3, CF at 2 m",
    )

with col5:
    primary_symptom = st.text_input(
        "Primary symptom (optional)",
        value=st.session_state.primary_symptom,
        placeholder="e.g. blurred vision, metamorphopsia, asymptomatic",
    )

col6, col7 = st.columns(2)

with col6:
    duration = st.text_input(
        "Duration (optional)",
        value=st.session_state.duration,
        placeholder="e.g. acute, 3 days, chronic, 2 months",
    )

with col7:
    relevant_history = st.text_input(
        "Relevant history (optional)",
        value=st.session_state.relevant_history,
        placeholder="e.g. diabetes, high myopia, recent viral illness",
    )

additional_notes = st.text_area(
    "Additional notes (optional)",
    value=st.session_state.additional_notes,
    placeholder="Any extra clinical context...",
    height=90,
)

st.session_state.age = int(age)
st.session_state.sex = sex
st.session_state.laterality = laterality
st.session_state.visual_acuity = visual_acuity or ""
st.session_state.primary_symptom = primary_symptom or ""
st.session_state.duration = duration or ""
st.session_state.relevant_history = relevant_history or ""
st.session_state.additional_notes = additional_notes or ""

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.subheader("Retinal images")

uploads = st.file_uploader(
    "Upload retinal images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
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
st.markdown('<span class="small-note">* Required fields: Age, Sex, Laterality</span>', unsafe_allow_html=True)

analyze = st.button("🔎 Analyze", type="primary", use_container_width=True)

if analyze:
    missing = []

    if st.session_state.sex == "Select":
        missing.append("Sex")
    if st.session_state.laterality == "Select":
        missing.append("Laterality")

    if missing:
        st.error("Please complete the required clinical fields: " + ", ".join(missing))
    elif not uploads:
        st.error("Please upload at least one image.")
    else:
        st.session_state.clinical = build_clinical_summary()
        st.session_state.images = uploads_to_images(uploads)
        run_analysis()
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Friendly Gemini error UI
# =========================
if st.session_state.last_error_type == "busy":
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.warning("System is busy at the moment. Please retry in a few seconds.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔁 Retry analysis", use_container_width=True):
            if st.session_state.images:
                run_analysis()
                st.rerun()

    with col2:
        if st.button("🆕 Start new case", use_container_width=True):
            reset_case()
            st.rerun()

    with st.expander("Technical details"):
        st.code(st.session_state.last_error_raw or "No technical details available.")

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.last_error_type:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.warning("Analysis could not be completed. Please retry.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔁 Retry analysis", use_container_width=True):
            if st.session_state.images:
                run_analysis()
                st.rerun()

    with col2:
        if st.button("🆕 Start new case", use_container_width=True):
            reset_case()
            st.rerun()

    with st.expander("Technical details"):
        st.code(st.session_state.last_error_raw or "No technical details available.")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Output history
# =========================
if st.session_state.report_history:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    latest_badge = f"{st.session_state.confidence_icon} Diagnostic confidence: {st.session_state.confidence_label}"
    st.markdown(
        f'<span class="pill">{latest_badge}</span> <span class="pill muted">Case #{st.session_state.case_id}</span>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    for idx, item in enumerate(st.session_state.report_history):
        item_badge = f"{item.get('confidence_icon', '🟡')} Diagnostic confidence: {item.get('confidence_label', 'Moderate')}"

        st.markdown(f"### {item['label']}")
        st.markdown(f'<span class="pill muted">{item_badge}</span>', unsafe_allow_html=True)
        st.markdown('<div class="subtle-box">', unsafe_allow_html=True)
        st.markdown(item["report"])
        st.markdown("</div>", unsafe_allow_html=True)

        if idx < len(st.session_state.report_history) - 1:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.button("🆕 Ask new patient", use_container_width=True):
        reset_case()
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Add more images / update diagnosis
# =========================
if st.session_state.report_history:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("➕ Add additional imaging")

    st.caption("Upload additional multimodal imaging for the same case and append an updated assessment below.")

    modality_options = [
        "OCT",
        "Fundus photo",
        "FAF",
        "FA",
        "OCTA",
        "ICGA",
        "Wide-field imaging",
        "Other",
    ]

    current_modality = st.session_state.additional_modality
    if current_modality not in modality_options:
        current_modality = "Other"

    extra_modality = st.selectbox(
        "Imaging modality",
        modality_options,
        index=modality_options.index(current_modality),
        key="extra_modality_box",
    )
    st.session_state.additional_modality = extra_modality

    extra_note = st.text_area(
        "Additional clinical note (optional)",
        placeholder="Example: Macular edema is better delineated on OCT, FA shows leakage, OCTA suggests type 1 MNV...",
        value=st.session_state.additional_clinical_note,
        key="additional_note_box",
        height=90,
    )
    st.session_state.additional_clinical_note = extra_note or ""

    extra_uploads = st.file_uploader(
        "Upload additional retinal images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key=f"add_uploader_{st.session_state.add_uploader_key}",
    )

    if extra_uploads:
        cols = st.columns(2)
        preview_extra = extra_uploads[:2]

        for i, f in enumerate(preview_extra):
            with cols[i % 2]:
                st.image(f, caption=f.name, use_container_width=True)

        if len(extra_uploads) > 2:
            st.caption(f"+ {len(extra_uploads) - 2} more additional image(s) selected.")

    if st.button("🔄 Update diagnosis", use_container_width=True):
        if not extra_uploads and not st.session_state.additional_clinical_note.strip():
            st.warning("Please upload at least one additional image or add a clinical note.")
        else:
            if extra_uploads:
                new_images = uploads_to_images(extra_uploads)
                st.session_state.images = merge_images(st.session_state.images, new_images)

            modality_note = f"Newly added imaging modality: {st.session_state.additional_modality}"

            combined_update_note = modality_note
            if st.session_state.additional_clinical_note.strip():
                combined_update_note += "\n" + st.session_state.additional_clinical_note.strip()

            if st.session_state.clinical.strip():
                st.session_state.clinical = (
                    st.session_state.clinical.strip()
                    + "\n\nAdditional update:\n"
                    + combined_update_note
                )
            else:
                st.session_state.clinical = combined_update_note

            st.session_state.additional_clinical_note = ""
            st.session_state.additional_modality = "OCT"
            st.session_state.add_uploader_key += 1

            run_analysis()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Chat
# =========================
if st.session_state.report_history:
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
