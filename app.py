import os
import json
import base64
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI

# Optional RAG deps
try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    faiss = None
    np = None

# -----------------------------
# CONFIG
# -----------------------------
APP_TITLE = "RetinaGPT (Vision-first)"
APP_SUBTITLE = "Educational retinal imaging discussion + optional RAG (no medical advice)"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # vision-first default
RAG_DIR = os.getenv("RAG_DIR", "data")               # expects data/index.faiss and data/meta.json
MAX_RAG_HITS = int(os.getenv("MAX_RAG_HITS", "6"))

st.set_page_config(page_title=APP_TITLE, page_icon="👁️", layout="wide")


# -----------------------------
# SYSTEM / PROMPTS (short on purpose)
# -----------------------------
SYSTEM_BASE = """You are RetinaGPT, a retina subspecialty imaging tutor.
The user uploads DE-IDENTIFIED ophthalmic images (fundus/OCT/FAF/angiography). These are NOT faces/people.
Task: describe imaging morphology and provide an EDUCATIONAL differential diagnosis.
No patient-specific treatment instructions. No dosing. Educational only."""

VISION_KEYWORDS_PROMPT = """Extract VISUAL FEATURES ONLY (no diagnosis).
Return 12-20 bullet keywords, each short.
Include: modality guess (fundus/OCT/FAF/FA/OCTA), laterality clues if visible,
pigmentation/background (hypopigmented fundus, prominent choroidal vessels),
macular appearance (foveal reflex, foveal hypoplasia suspicion),
optic disc appearance, hemorrhage/exudates, vessel caliber/tortuosity,
lesion shape/location keywords (e.g., torpedo-shaped temporal to fovea)."""

FINAL_TASK_PROMPT = """Use the images + clinical metadata to produce an educational impression.
1) Describe key findings by modality.
2) Provide a ranked differential with brief for/against.
3) Recommend missing modalities/tests (high level).
If RAG cards are provided, treat them as reference knowledge and cite them implicitly (do not invent facts beyond them)."""


# -----------------------------
# UTIL
# -----------------------------
def safe_get_api_key() -> Optional[str]:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def to_data_url(file_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_input_images(uploaded_files) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    if not uploaded_files:
        return parts
    for f in uploaded_files:
        mime = f.type or "image/jpeg"
        url = to_data_url(f.getvalue(), mime)
        parts.append({"type": "input_image", "image_url": url, "detail": "high"})
    return parts


# -----------------------------
# OPTIONAL RAG (FAISS)
# -----------------------------
@st.cache_resource
def load_rag_index() -> Tuple[bool, Optional[Any], Optional[List[Dict[str, Any]]]]:
    index_path = os.path.join(RAG_DIR, "index.faiss")
    meta_path = os.path.join(RAG_DIR, "meta.json")

    if faiss is None or np is None:
        return (False, None, None)

    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return (False, None, None)

    try:
        idx = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, list):
            return (False, None, None)
        return (True, idx, meta)
    except Exception:
        return (False, None, None)


def get_embedding(client: OpenAI, text: str) -> Optional[List[float]]:
    try:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return emb.data[0].embedding
    except Exception:
        return None


def rag_retrieve(client: OpenAI, query: str, k: int) -> Tuple[str, Dict[str, Any]]:
    enabled, idx, meta = load_rag_index()
    status = {"enabled": False, "hits": 0, "index_dim": None, "error": None}

    if not enabled or idx is None or meta is None:
        return ("", status)

    try:
        status["enabled"] = True
        status["index_dim"] = int(idx.d)

        emb = get_embedding(client, query)
        if emb is None:
            status["error"] = "Embedding failed"
            return ("", status)

        x = np.array([emb], dtype="float32")
        D, I = idx.search(x, k)

        hits = []
        for j, row_id in enumerate(I[0].tolist()):
            if row_id < 0 or row_id >= len(meta):
                continue
            chunk = meta[row_id]
            title = chunk.get("title", f"chunk_{row_id}")
            text = chunk.get("text", "")
            hits.append((title, text, float(D[0][j])))

        status["hits"] = len(hits)
        if not hits:
            return ("", status)

        lines = ["REFERENCE CARDS (RAG):"]
        for n, (title, text, dist) in enumerate(hits, start=1):
            lines.append(f"\n--- CARD {n}: {title} ---\n{text}\n")
        return ("\n".join(lines), status)

    except Exception as e:
        status["error"] = str(e)
        return ("", status)


# -----------------------------
# JSON SCHEMA (Structured Outputs)
# -----------------------------
RESPONSE_SCHEMA: Dict[str, Any] = {
    "name": "retina_report",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "case_summary": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "age": {"type": ["string", "null"]},
                    "sex": {"type": ["string", "null"]},
                    "symptoms": {"type": ["string", "null"]},
                    "duration": {"type": ["string", "null"]},
                    "laterality": {"type": ["string", "null"]},
                    "history": {"type": ["string", "null"]},
                },
                "required": ["age", "sex", "symptoms", "duration", "laterality", "history"],
            },
            "modalities_detected": {
                "type": "array",
                "items": {"type": "string"},
            },
            "missing_modalities_suggested": {
                "type": "array",
                "items": {"type": "string"},
            },
            "patterns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["name", "confidence"],
                },
            },
            "feature_checklist": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subretinal_fluid": {"type": "string"},
                    "intraretinal_fluid": {"type": "string"},
                    "hemorrhage_exudation": {"type": "string"},
                    "inner_retinal_ischemia": {"type": "string"},
                    "ez_disruption": {"type": "string"},
                    "rpe_atrophy_hypertransmission": {"type": "string"},
                    "vitelliform_material": {"type": "string"},
                    "inflammatory_signs": {"type": "string"},
                    "m_nv_suspected": {"type": "string"},
                    "true_mass_lesion_suspected": {"type": "string"},
                },
                "required": [
                    "subretinal_fluid", "intraretinal_fluid", "hemorrhage_exudation",
                    "inner_retinal_ischemia", "ez_disruption", "rpe_atrophy_hypertransmission",
                    "vitelliform_material", "inflammatory_signs", "m_nv_suspected",
                    "true_mass_lesion_suspected"
                ],
            },
            "most_likely": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "diagnosis": {"type": "string"},
                    "probability": {"type": "number"},
                },
                "required": ["diagnosis", "probability"],
            },
            "differential": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "diagnosis": {"type": "string"},
                        "probability": {"type": "number"},
                        "for": {"type": "array", "items": {"type": "string"}},
                        "against": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["diagnosis", "probability", "for", "against"],
                },
            },
            "confidence_level": {"type": "string"},
            "next_best_requests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "request": {"type": "string"},
                        "why": {"type": "string"},
                    },
                    "required": ["request", "why"],
                },
            },
            "urgency": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": [
            "case_summary", "modalities_detected", "missing_modalities_suggested",
            "patterns", "feature_checklist", "most_likely", "differential",
            "confidence_level", "next_best_requests", "urgency", "notes"
        ],
    },
}


def render_human_report(j: Dict[str, Any]) -> str:
    cs = j.get("case_summary", {})
    lines = []
    lines.append("## Human Report\n")

    lines.append("### 1) Clinical Triage")
    triage = f"Age: {cs.get('age')} | Sex: {cs.get('sex')} | Symptoms: {cs.get('symptoms')} | Duration: {cs.get('duration')} | Laterality: {cs.get('laterality')} | History: {cs.get('history')}"
    lines.append(triage)

    lines.append("\n### 2) Detected Modalities + Missing Modalities")
    lines.append(f"Detected: {', '.join(j.get('modalities_detected', []) or [])}")
    lines.append(f"Missing/suggested: {', '.join(j.get('missing_modalities_suggested', []) or [])}")

    lines.append("\n### 3) Detected Pattern(s) (with confidence)")
    for p in (j.get("patterns") or [])[:6]:
        lines.append(f"- {p.get('name')} (confidence: {p.get('confidence')})")

    lines.append("\n### 4) Image Feature Checklist")
    fc = j.get("feature_checklist", {})
    for k, v in fc.items():
        lines.append(f"- {k}: {v}")

    lines.append("\n### 5) Most Likely Diagnosis (educational impression)")
    ml = j.get("most_likely", {})
    lines.append(f"{ml.get('diagnosis')} (p≈{ml.get('probability')})")

    lines.append("\n### 6) Differential Diagnosis (ranked)")
    for d in (j.get("differential") or [])[:6]:
        lines.append(f"- **{d.get('diagnosis')}** (p≈{d.get('probability')})")
        if d.get("for"):
            lines.append(f"  - For: {', '.join(d.get('for')[:5])}")
        if d.get("against"):
            lines.append(f"  - Against: {', '.join(d.get('against')[:5])}")

    lines.append("\n### 7) Confidence + Urgency")
    lines.append(f"Confidence: {j.get('confidence_level')} | Urgency: {j.get('urgency')}")

    lines.append("\n### 8) Next best requests")
    for r in (j.get("next_best_requests") or [])[:6]:
        lines.append(f"- {r.get('request')}: {r.get('why')}")

    lines.append("\n### 9) Notes")
    lines.append(j.get("notes", ""))

    lines.append("\n---\n**Educational only. Not medical advice.**")
    return "\n".join(lines)


# -----------------------------
# MODEL CALLS (Responses API)
# -----------------------------
def responses_keywords(
    client: OpenAI,
    model: str,
    clinical_text: str,
    image_parts: List[Dict[str, Any]],
) -> str:
    user_text = (
        "DE-IDENTIFIED OPHTHALMIC IMAGES (retina).\n"
        "Clinical metadata:\n"
        f"{clinical_text.strip() if clinical_text.strip() else '(none)'}\n\n"
        + VISION_KEYWORDS_PROMPT
    )

    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}] + image_parts
        }],
        # keep it deterministic
        temperature=0.0,
        max_output_tokens=300,
    )
    return (resp.output_text or "").strip()


def responses_final(
    client: OpenAI,
    model: str,
    clinical_text: str,
    image_parts: List[Dict[str, Any]],
    rag_context: str,
    keywords: str,
) -> Dict[str, Any]:
    user_text = (
        "DE-IDENTIFIED OPHTHALMIC IMAGES (retina). Educational differential only.\n\n"
        f"CLINICAL METADATA:\n{clinical_text.strip() if clinical_text.strip() else '(not provided)'}\n\n"
        f"IMAGE KEYWORDS (visual-only extractor):\n{keywords if keywords else '(none)'}\n\n"
        f"{FINAL_TASK_PROMPT}\n\n"
    )

    if rag_context:
        user_text += "\nREFERENCE CARDS (RAG):\n" + rag_context + "\n"

    resp = client.responses.create(
        model=model,
        input=[{
            "role": "system",
            "content": [{"type": "input_text", "text": SYSTEM_BASE}]
        }, {
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}] + image_parts
        }],
        text={
            "format": {
                "type": "json_schema",
                "json_schema": RESPONSE_SCHEMA
            }
        },
        temperature=0.2,
        max_output_tokens=1200,
    )

    # With json_schema, output_text should be valid JSON string.
    out = (resp.output_text or "").strip()
    return json.loads(out)


# -----------------------------
# UI
# -----------------------------
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

api_key = safe_get_api_key()
if not api_key:
    st.error("OPENAI_API_KEY bulunamadı. Streamlit Secrets veya env var olarak ekle.")
    st.stop()

client = OpenAI(api_key=api_key)

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", options=["gpt-4o", "gpt-4o-mini"], index=0)
    use_rag = st.checkbox("Enable RAG (FAISS)", value=True)
    rag_k = st.slider("RAG hits (k)", min_value=2, max_value=10, value=MAX_RAG_HITS, step=1)
    st.divider()
    st.caption("Tip: Best vision performance usually with gpt-4o + detail=high, and minimal prompt noise.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Clinical metadata")
    clinical_text = st.text_area(
        "Age, Sex, Symptoms, Duration, Laterality, History",
        height=140,
        placeholder="Example:\nAge: 27\nSex: Male\nSymptoms: acute blurred vision + central scotomas\nDuration: 3 days\nLaterality: bilateral\nHistory: recent URTI 1 week ago"
    )

    st.subheader("Upload retinal imaging")
    files = st.file_uploader(
        "Fundus / OCT / FAF / FA / OCTA (jpg/png/webp)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    run = st.button("🔎 Analyze", type="primary")

with col2:
    st.subheader("Debug")
    enabled, idx, meta = load_rag_index()
    if use_rag and enabled:
        st.success(f"RAG index loaded ✅ | meta_len={len(meta)}")
    elif use_rag:
        st.warning("RAG not available (check faiss-cpu + data/index.faiss + data/meta.json).")

    keywords_box = st.empty()
    rag_box = st.empty()

    with st.expander("Extracted image keywords (debug)", expanded=False):
        st.write("Run Analyze to populate.")

    with st.expander("Retrieved cards (debug)", expanded=False):
        st.write("Run Analyze to populate.")


if run:
    image_parts = build_input_images(files)

    if not image_parts:
        st.warning("En az 1 retinal görüntü yüklemen önerilir.")

    with st.spinner("Step 1/3: Extracting visual keywords (vision-first)..."):
        try:
            keywords = responses_keywords(client, model="gpt-4o", clinical_text=clinical_text, image_parts=image_parts)
        except Exception as e:
            keywords = ""
            st.error(f"Keyword extractor failed: {e}")

    with st.spinner("Step 2/3: RAG retrieval (optional)..."):
        rag_context = ""
        rag_status = {"enabled": False, "hits": 0, "index_dim": None, "error": None}
        if use_rag:
            query = (clinical_text or "").strip()
            if keywords:
                query += "\n\n" + keywords
            query += "\n\nretina fundus OCT FAF angiography differential diagnosis"
            rag_context, rag_status = rag_retrieve(client, query=query, k=rag_k)

    with st.spinner("Step 3/3: Final structured analysis..."):
        try:
            result_json = responses_final(
                client=client,
                model=model,
                clinical_text=clinical_text,
                image_parts=image_parts,
                rag_context=rag_context,
                keywords=keywords,
            )
        except Exception as e:
            st.error(f"Final analysis failed: {e}")
            st.stop()

    st.subheader("Structured JSON")
    st.json(result_json)

    st.subheader("Human Report")
    st.markdown(render_human_report(result_json))

    with st.expander("Extracted image keywords (debug)", expanded=False):
        st.code(keywords or "(none)", language="text")

    with st.expander("Retrieved cards (debug)", expanded=False):
        if rag_context:
            st.code(rag_context, language="text")
        else:
            st.write(f"(none) | rag_status={rag_status}")

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(result_json, ensure_ascii=False, indent=2),
        file_name="retina_report.json",
        mime="application/json",
    )
