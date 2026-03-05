import os
import json
import base64
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI

# Optional RAG deps (FAISS local index)
try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    faiss = None
    np = None


# =============================
# CONFIG
# =============================
APP_TITLE = "RetinaGPT (2-pass) — Vision ➜ RAG ➜ Reasoning"
APP_SUBTITLE = "Educational retinal imaging discussion (DE-IDENTIFIED). Not medical advice."

DEFAULT_VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
DEFAULT_REASON_MODEL = os.getenv("REASON_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

RAG_DIR = os.getenv("RAG_DIR", "data")  # expects data/index.faiss and data/meta.json
MAX_RAG_HITS = int(os.getenv("MAX_RAG_HITS", "6"))

st.set_page_config(page_title=APP_TITLE, page_icon="👁️", layout="wide")


# =============================
# SAFER SYSTEM PROMPT (final)
# =============================
FINAL_SYSTEM_PROMPT = """
You are RetinaGPT, a retina subspecialty educational imaging discussion system.

IMPORTANT CONTEXT
- Inputs are DE-IDENTIFIED ophthalmic images (fundus/OCT/FAF/FA/OCTA).
- Not photos of faces/people. Do NOT identify any person.
- Provide educational discussion only. No individualized treatment plans or dosing.

STYLE
- Formal medical English, objective, concise.
- Describe morphology first, then ranked differential.
- Use retina terminology (EZ, RPE, SHRM, PED, hypertransmission, etc.).
- Do not over-commit when uncertain.

RAG
If REFERENCE CARDS are provided, treat them as primary factual source to refine discriminators, pitfalls, work-up, and high-level management.

GUARDRAILS
- Do NOT call something a “mass/tumor/elevated lesion” unless OCT clearly shows a dome/solid structure.
- If fundus shows pale/hypopigmented background with prominent choroidal vessels, consider conditions with reduced fundus pigment (e.g., albinism) and comment on foveal hypoplasia suspicion (if visible).

OUTPUT
Return ONE JSON object matching the provided schema exactly.
"""


# =============================
# UTIL
# =============================
def safe_get_secrets_api_key() -> Optional[str]:
    # Streamlit Cloud: st.secrets["OPENAI_API_KEY"]
    # Local: env var OPENAI_API_KEY
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def b64_data_url(file_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(file_bytes).decode('utf-8')}"


def build_image_inputs(uploaded_files) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not uploaded_files:
        return items
    for f in uploaded_files:
        mime = f.type or "image/jpeg"
        url = b64_data_url(f.getvalue(), mime)
        items.append({"type": "input_image", "image_url": url, "detail": "high"})
    return items


def _coerce_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""


# =============================
# RAG (FAISS)
# =============================
@st.cache_resource
def load_faiss(rag_dir: str) -> Tuple[bool, Optional[Any], Optional[List[Dict[str, Any]]]]:
    if faiss is None or np is None:
        return (False, None, None)

    index_path = os.path.join(rag_dir, "index.faiss")
    meta_path = os.path.join(rag_dir, "meta.json")
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
        emb = client.embeddings.create(model=EMBED_MODEL, input=text)
        return emb.data[0].embedding
    except Exception:
        return None


def rag_retrieve(client: OpenAI, query: str, k: int) -> Tuple[str, Dict[str, Any]]:
    enabled, idx, meta = load_faiss(RAG_DIR)
    status: Dict[str, Any] = {"enabled": False, "hits": 0, "index_dim": None, "error": None}

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

        hits: List[Tuple[str, str, float]] = []
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

        context_lines = ["REFERENCE CARDS (RAG):"]
        for n, (title, text, _dist) in enumerate(hits, start=1):
            context_lines.append(f"\n--- CARD {n}: {title} ---\n{text}\n")

        return ("\n".join(context_lines), status)

    except Exception as e:
        status["error"] = _coerce_str(e)
        return ("", status)


# =============================
# 1) VISION FEATURE EXTRACTOR (Structured Output)
# =============================
VISION_SCHEMA = {
    "name": "retina_vision_extract",
    "schema": {
        "type": "object",
        "properties": {
            "modality_guess": {"type": "array", "items": {"type": "string"}},
            "fundus_background_pigment": {"type": "string", "description": "LOW|NORMAL|INCREASED|UNCERTAIN"},
            "choroidal_vessel_visibility": {"type": "string", "description": "PROMINENT|NORMAL|REDUCED|UNCERTAIN"},
            "foveal_hypoplasia_suspected": {"type": "string", "description": "YES|NO|UNCERTAIN"},
            "optic_disc_notes": {"type": "array", "items": {"type": "string"}},
            "hemorrhage_exudates": {"type": "string", "description": "PRESENT|ABSENT|UNCERTAIN"},
            "key_morphology_keywords": {"type": "array", "items": {"type": "string"}},
            "lesion_location_keywords": {"type": "array", "items": {"type": "string"}},
            "quality_notes": {"type": "array", "items": {"type": "string"}},
            "one_line_morphology": {"type": "string"},
        },
        "required": ["key_morphology_keywords", "one_line_morphology"],
    },
}


def vision_extract(
    client: OpenAI,
    model: str,
    clinical_text: str,
    image_inputs: List[Dict[str, Any]],
    optional_note: str,
) -> Dict[str, Any]:
    system = (
        "You are a retina imaging morphology extractor for DE-IDENTIFIED ophthalmic images.\n"
        "Task: morphology ONLY. No diagnosis.\n"
        "Be sensitive to: retinal hemorrhage vs pigment, hypopigmented fundus + prominent choroidal vessels, "
        "foveal hypoplasia suspicion, disc anomalies, myopic tessellation.\n"
        "Return structured JSON only."
    )

    user_text = (
        "DE-IDENTIFIED OPHTHALMIC IMAGES (retina), not faces/people.\n\n"
        f"CLINICAL:\n{clinical_text.strip() if clinical_text.strip() else '(none)'}\n\n"
        f"OPTIONAL NOTE:\n{optional_note.strip() if optional_note.strip() else '(none)'}\n"
    )

    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}] + (image_inputs or []),
            },
        ],
        temperature=0.0,
        max_output_tokens=600,
        response_format={
    "type": "json_schema",
    "json_schema": {
        "name": VISION_SCHEMA["name"],
        "schema": VISION_SCHEMA["schema"],
        "strict": True
    }
},
    )

    # Structured output comes back as parsed JSON
    try:
        return resp.output[0].content[0].json  # type: ignore
    except Exception:
        # Fallback (should be rare with schema)
        return {"key_morphology_keywords": [], "one_line_morphology": "extract_failed"}


# =============================
# 2) FINAL REASONER (Structured Output)
# =============================
FINAL_SCHEMA = {
    "name": "retina_final",
    "schema": {
        "type": "object",
        "properties": {
            "structured": {
                "type": "object",
                "properties": {
                    "case_summary": {
                        "type": "object",
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
                    "modalities_detected": {"type": "array", "items": {"type": "string"}},
                    "missing_modalities_suggested": {"type": "array", "items": {"type": "string"}},
                    "patterns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["name", "confidence"],
                        },
                    },
                    "feature_checklist": {
                        "type": "object",
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
                            "subretinal_fluid",
                            "intraretinal_fluid",
                            "hemorrhage_exudation",
                            "inner_retinal_ischemia",
                            "ez_disruption",
                            "rpe_atrophy_hypertransmission",
                            "vitelliform_material",
                            "inflammatory_signs",
                            "m_nv_suspected",
                            "true_mass_lesion_suspected",
                        ],
                    },
                    "most_likely": {
                        "type": "object",
                        "properties": {"diagnosis": {"type": "string"}, "probability": {"type": "number"}},
                        "required": ["diagnosis", "probability"],
                    },
                    "differential": {
                        "type": "array",
                        "items": {
                            "type": "object",
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
                            "properties": {"request": {"type": "string"}, "why": {"type": "string"}},
                            "required": ["request", "why"],
                        },
                    },
                    "urgency": {"type": "string"},
                },
                "required": [
                    "case_summary",
                    "modalities_detected",
                    "missing_modalities_suggested",
                    "patterns",
                    "feature_checklist",
                    "most_likely",
                    "differential",
                    "confidence_level",
                    "next_best_requests",
                    "urgency",
                ],
            },
            "report_markdown": {"type": "string"},
        },
        "required": ["structured", "report_markdown"],
    },
}


def final_reason(
    client: OpenAI,
    model: str,
    clinical_text: str,
    vision_features: Dict[str, Any],
    rag_context: str,
    image_inputs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # We include images again in final pass (helps a LOT vs relying only on keywords)
    user_text = (
        "DE-IDENTIFIED OPHTHALMIC IMAGES (retina).\n\n"
        "CLINICAL METADATA:\n"
        + (clinical_text.strip() if clinical_text.strip() else "(not provided)")
        + "\n\n"
        "VISION EXTRACTED MORPHOLOGY (structured):\n"
        + json.dumps(vision_features, ensure_ascii=False, indent=2)
        + "\n\n"
        "If RAG cards are provided below, use them to refine the differential."
    )

    content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_text}] + (image_inputs or [])
    messages = [
        {"role": "system", "content": [{"type": "input_text", "text": FINAL_SYSTEM_PROMPT}]},
    ]
    if rag_context:
        messages.append({"role": "system", "content": [{"type": "input_text", "text": rag_context}]})
    messages.append({"role": "user", "content": content})

    resp = client.responses.create(
        model=model,
        input=messages,
        temperature=0.2,
        max_output_tokens=1600,
        response_format={"type": "json_schema", "json_schema": FINAL_SCHEMA},
    )

    try:
        return resp.output[0].content[0].json  # type: ignore
    except Exception:
        # Fallback
        return {
            "structured": {
                "case_summary": {"age": None, "sex": None, "symptoms": None, "duration": None, "laterality": None, "history": None},
                "modalities_detected": [],
                "missing_modalities_suggested": [],
                "patterns": [],
                "feature_checklist": {
                    "subretinal_fluid": "UNCERTAIN",
                    "intraretinal_fluid": "UNCERTAIN",
                    "hemorrhage_exudation": "UNCERTAIN",
                    "inner_retinal_ischemia": "UNCERTAIN",
                    "ez_disruption": "UNCERTAIN",
                    "rpe_atrophy_hypertransmission": "UNCERTAIN",
                    "vitelliform_material": "UNCERTAIN",
                    "inflammatory_signs": "UNCERTAIN",
                    "m_nv_suspected": "UNCERTAIN",
                    "true_mass_lesion_suspected": "UNCERTAIN",
                },
                "most_likely": {"diagnosis": "Unknown", "probability": 0.0},
                "differential": [],
                "confidence_level": "LOW",
                "next_best_requests": [],
                "urgency": "ROUTINE",
            },
            "report_markdown": "Model output parse failed.",
        }


# =============================
# UI
# =============================
api_key = safe_get_secrets_api_key()
if not api_key:
    st.error("OPENAI_API_KEY bulunamadı (Streamlit Secrets veya environment variable).")
    st.stop()

client = OpenAI(api_key=api_key)

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("Settings")
    vision_model = st.selectbox("Vision model (extractor)", [DEFAULT_VISION_MODEL, "gpt-4o-mini"], index=0)
    reason_model = st.selectbox("Reasoning model (final)", [DEFAULT_REASON_MODEL, "gpt-4o"], index=0)
    use_rag = st.checkbox("Enable RAG (FAISS)", value=True)
    rag_k = st.slider("RAG hits (k)", min_value=2, max_value=10, value=MAX_RAG_HITS, step=1)
    show_debug = st.checkbox("Show debug panels", value=True)

    enabled, _idx, meta = load_faiss(RAG_DIR)
    st.caption(f"RAG index loaded? {enabled} | meta_len={len(meta) if isinstance(meta, list) else 0}")

col1, col2 = st.columns([1.05, 0.95], gap="large")

with col1:
    st.subheader("Clinical metadata")
    clinical_text = st.text_area(
        "Age, Sex, Symptoms, Duration, Laterality, History",
        height=140,
        placeholder="Example:\nAge: 27\nSex: Male\nSymptoms: acute blurred vision + central scotomas\nDuration: 3 days\nLaterality: bilateral\nHistory: recent URTI 1 week ago",
    )

    st.subheader("Upload retinal imaging")
    files = st.file_uploader(
        "Fundus / OCT / FAF / FA / OCTA (jpg/png/webp)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    st.subheader("Optional note")
    optional_note = st.text_input("Optional note (e.g., “suspect albinism?”)", value="")

with col2:
    st.subheader("Analyze")
    run = st.button("🔎 Run 2-pass analysis", type="primary")

    if show_debug:
        st.divider()
        st.subheader("Debug")
        debug_placeholder = st.empty()


# =============================
# RUN
# =============================
if run:
    if not files and not clinical_text.strip():
        st.warning("En az 1 görüntü veya klinik bilgi ile denemen daha iyi olur.")

    image_inputs = build_image_inputs(files)

    # Step 1 — Vision features
    if show_debug:
        debug_placeholder.info("Step 1/3: Vision feature extraction…")
    vf = vision_extract(
        client=client,
        model=vision_model,
        clinical_text=clinical_text,
        image_inputs=image_inputs,
        optional_note=optional_note,
    )

    # Step 2 — RAG retrieval using features + clinical
    rag_context = ""
    rag_status = {"enabled": False, "hits": 0, "index_dim": None, "error": None}

    if use_rag:
        # Build a retrieval query that is robust
        q_parts = []
        if clinical_text.strip():
            q_parts.append(clinical_text.strip())
        q_parts.append("MORPHOLOGY: " + _coerce_str(vf.get("one_line_morphology", "")))
        kws = vf.get("key_morphology_keywords") or []
        if isinstance(kws, list) and kws:
            q_parts.append("KEYWORDS: " + ", ".join([_coerce_str(x) for x in kws[:20]]))
        # Add strong discriminators
        q_parts.append(
            "retina fundus hemorrhage vs pigmentation hypopigmented fundus prominent choroidal vessels foveal hypoplasia"
        )
        retrieval_query = "\n".join(q_parts)

        if show_debug:
            debug_placeholder.info("Step 2/3: RAG retrieval…")
        rag_context, rag_status = rag_retrieve(client, retrieval_query, k=rag_k)

    # Step 3 — Final reasoner (uses images + features + RAG)
    if show_debug:
        debug_placeholder.info("Step 3/3: Final reasoning…")
    out = final_reason(
        client=client,
        model=reason_model,
        clinical_text=clinical_text,
        vision_features=vf,
        rag_context=rag_context,
        image_inputs=image_inputs,
    )

    structured = out.get("structured", {})
    report_md = out.get("report_markdown", "")

    st.success("Done.")

    if show_debug:
        st.divider()
        st.subheader("Vision extracted features (debug)")
        st.json(vf)

        st.subheader("RAG status (debug)")
        st.json(rag_status)

        with st.expander("Show retrieved cards (debug)", expanded=False):
            st.text(rag_context if rag_context else "(No RAG context)")

    st.divider()
    st.subheader("Human Report")
    st.markdown(report_md if report_md else "(No report)")

    st.subheader("Structured output (JSON)")
    st.json(structured)
