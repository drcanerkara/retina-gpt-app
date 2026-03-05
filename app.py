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
APP_TITLE = "RetinaGPT (2-pass) — Basic"
APP_SUBTITLE = "Vision extractor → (optional) RAG → Final reasoning (Educational only)"

DEFAULT_VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
DEFAULT_FINAL_MODEL = os.getenv("FINAL_MODEL", "gpt-4o-mini")

RAG_DIR = os.getenv("RAG_DIR", "data")  # expects data/index.faiss and data/meta.json
MAX_RAG_HITS = int(os.getenv("MAX_RAG_HITS", "6"))

st.set_page_config(page_title=APP_TITLE, page_icon="👁️", layout="wide")


# -----------------------------
# UTIL
# -----------------------------
def safe_api_key() -> Optional[str]:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def b64_data_url(file_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(file_bytes).decode('utf-8')}"


def json_loads_safe(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Fallback extractor: find first {...} block and json.loads.
    Keeps app from crashing if model outputs extra text.
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    return json_loads_safe(candidate)


# -----------------------------
# OPTIONAL RAG (FAISS)
# -----------------------------
@st.cache_resource
def load_rag_index(rag_dir: str) -> Tuple[bool, Optional[Any], Optional[List[Dict[str, Any]]], Optional[str]]:
    if faiss is None or np is None:
        return (False, None, None, "faiss/numpy not installed")
    index_path = os.path.join(rag_dir, "index.faiss")
    meta_path = os.path.join(rag_dir, "meta.json")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return (False, None, None, f"Missing {index_path} or {meta_path}")
    try:
        idx = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, list):
            return (False, None, None, "meta.json is not a list")
        return (True, idx, meta, None)
    except Exception as e:
        return (False, None, None, str(e))


def get_embedding(client: OpenAI, text: str) -> Optional[List[float]]:
    try:
        emb = client.embeddings.create(model="text-embedding-3-small", input=text)
        return emb.data[0].embedding
    except Exception:
        return None


def rag_retrieve(client: OpenAI, query: str, k: int) -> Tuple[str, Dict[str, Any]]:
    enabled, idx, meta, err = load_rag_index(RAG_DIR)
    status = {"enabled": False, "hits": 0, "dim": None, "error": err}
    if not enabled or idx is None or meta is None:
        return ("", status)

    try:
        status["enabled"] = True
        status["dim"] = int(idx.d)

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

        # Keep it short to avoid token bloat
        lines = ["REFERENCE CARDS (RAG):"]
        for n, (title, txt, dist) in enumerate(hits, start=1):
            snippet = txt.strip()
            if len(snippet) > 1800:
                snippet = snippet[:1800] + "..."
            lines.append(f"\n--- CARD {n}: {title} ---\n{snippet}\n")
        return ("\n".join(lines), status)

    except Exception as e:
        status["error"] = str(e)
        return ("", status)


# -----------------------------
# SCHEMAS (IMPORTANT: wrapper format!)
# -----------------------------
VISION_SCHEMA = {
    "name": "retina_vision_features",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "modality_guess": {"type": "array", "items": {"type": "string"}},
            "global_distribution": {"type": "string"},
            "fundus_pigmentation": {"type": "string"},
            "choroidal_vessel_visibility": {"type": "string"},
            "macular_reflex": {"type": "string"},
            "foveal_hypoplasia_suspected": {"type": "string"},
            "optic_disc_notes": {"type": "array", "items": {"type": "string"}},
            "vascular_pattern": {"type": "array", "items": {"type": "string"}},
            "hemorrhage_exudates": {"type": "string"},
            "lesion_shape_keywords": {"type": "array", "items": {"type": "string"}},
            "lesion_location_keywords": {"type": "array", "items": {"type": "string"}},
            "outer_retina_rpe_clues": {"type": "array", "items": {"type": "string"}},
            "key_findings_bullets": {"type": "array", "items": {"type": "string"}},
            "quality_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "modality_guess",
            "global_distribution",
            "fundus_pigmentation",
            "choroidal_vessel_visibility",
            "macular_reflex",
            "foveal_hypoplasia_suspected",
            "optic_disc_notes",
            "vascular_pattern",
            "hemorrhage_exudates",
            "lesion_shape_keywords",
            "lesion_location_keywords",
            "outer_retina_rpe_clues",
            "key_findings_bullets",
            "quality_notes",
        ],
    },
}

FINAL_SCHEMA = {
    "name": "retina_final_report",
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
            "modalities_detected": {"type": "array", "items": {"type": "string"}},
            "missing_modalities_suggested": {"type": "array", "items": {"type": "string"}},
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
                "additionalProperties": False,
                "properties": {"diagnosis": {"type": "string"}, "probability": {"type": "number"}},
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
                    "properties": {"request": {"type": "string"}, "why": {"type": "string"}},
                    "required": ["request", "why"],
                },
            },
            "urgency": {"type": "string"},
            "human_report": {"type": "string"},
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
            "human_report",
        ],
    },
}


# -----------------------------
# PROMPTS
# -----------------------------
VISION_SYSTEM = """You are a retina imaging feature extractor for DE-IDENTIFIED ophthalmic images (retina/OCT/FA/FAF/OCTA).
These are not faces/people. Do not identify anyone.
Task: output ONLY the requested JSON with retina morphology features.
Be conservative when uncertain; use 'UNCERTAIN' or empty lists.
"""

# This is the key: tell it explicitly what to look for (retina-specific)
VISION_USER_TEMPLATE = """Clinical note (may be brief):
{clinical_note}

Extract retina-relevant morphology features.
Focus on:
- hemorrhage vs pigment vs atrophy
- global hypopigmentation / tessellation / choroidal visibility (e.g., albinism clues)
- vascular tortuosity, caliber changes, ischemia clues
- macular lesion shape/location keywords (e.g., torpedo/teardrop temporal to fovea)
- optic disc anomalies (pit, coloboma, hypoplasia)
Return strict JSON only.
"""

FINAL_SYSTEM = """You are RetinaGPT (educational only).
You will receive:
(1) Clinical note
(2) Vision-extracted features (structured JSON)
(3) Optional REFERENCE CARDS (RAG)

Task:
- Produce an educational retina differential and structured report.
- Use the vision features as PRIMARY evidence of what is seen.
- Use RAG cards to refine discriminators/pitfalls (high level).
- No patient-identifying content. No individualized treatment plan/dosing.
Return strict JSON only in the specified schema.
"""

FINAL_USER_TEMPLATE = """CLINICAL NOTE:
{clinical_note}

VISION FEATURES (PRIMARY OBSERVATION SOURCE):
{vision_json}

{rag_block}

Write:
- detected modalities
- likely pattern(s)
- most likely diagnosis + weighted differential
- confidence + next best tests (high-level)
- include 'human_report' as a readable numbered report (1..13).
"""


# -----------------------------
# CORE CALLS (Responses API)
# -----------------------------
def responses_json_call(
    client: OpenAI,
    model: str,
    system_text: str,
    user_text: str,
    image_data_urls: List[str],
    json_schema: Dict[str, Any],
    max_output_tokens: int = 900,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Returns (parsed_json_or_none, raw_output_text)
    Uses strict json_schema wrapper (2026-safe).
    """
    input_content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_text}]
    for url in image_data_urls:
        input_content.append({"type": "input_image", "image_url": url, "detail": "high"})

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
                {"role": "user", "content": input_content},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema["name"],
                    "schema": json_schema["schema"],
                    "strict": True,
                },
            },
            max_output_tokens=max_output_tokens,
        )
        raw = getattr(resp, "output_text", "") or ""
        parsed = json_loads_safe(raw) or extract_json_from_text(raw)
        return (parsed, raw)
    except Exception as e:
        return (None, f"ERROR: {e}")


# -----------------------------
# UI
# -----------------------------
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

api_key = safe_api_key()
if not api_key:
    st.error("OPENAI_API_KEY not found. Add to Streamlit Secrets or env var.")
    st.stop()

client = OpenAI(api_key=api_key)

with st.sidebar:
    st.header("Settings")
    vision_model = st.selectbox("Vision model (extractor)", [DEFAULT_VISION_MODEL, "gpt-4o"], index=0)
    final_model = st.selectbox("Reasoning model (final)", [DEFAULT_FINAL_MODEL, "gpt-4o-mini", "gpt-4o"], index=0)
    use_rag = st.checkbox("Enable RAG (FAISS)", value=True)
    rag_k = st.slider("RAG hits (k)", 2, 10, min(MAX_RAG_HITS, 10), 1)
    show_debug = st.checkbox("Show debug panels", value=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Clinical note")
    clinical_note = st.text_area(
        "Age / sex / symptoms / duration / laterality / history",
        height=140,
        placeholder="Example: 30M, congenital nystagmus, low vision since childhood, bilateral.",
    )

    st.subheader("Upload retinal imaging")
    uploaded = st.file_uploader(
        "Fundus / OCT / FAF / FA / OCTA (jpg/png/webp)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    st.subheader("Optional note")
    optional_note = st.text_input("Optional hint (e.g., 'suspect albinism?')", value="")

with col2:
    st.subheader("Run")
    run = st.button("🔎 Analyze", type="primary")

    if show_debug:
        st.markdown("---")
        st.subheader("Debug")
        debug_box = st.empty()

# -----------------------------
# RUN
# -----------------------------
if run:
    if not uploaded:
        st.warning("Please upload at least 1 image.")
        st.stop()

    # Build data URLs
    image_urls: List[str] = []
    for f in uploaded:
        mime = f.type or "image/jpeg"
        image_urls.append(b64_data_url(f.getvalue(), mime))

    # 1) Vision extract (ChatGPT vision)
    with st.spinner("Step 1/3: Vision extractor running..."):
        user_text = VISION_USER_TEMPLATE.format(
            clinical_note=(clinical_note.strip() + ("\nHint: " + optional_note if optional_note else "")).strip()
        )
        vision_parsed, vision_raw = responses_json_call(
            client=client,
            model=vision_model,
            system_text=VISION_SYSTEM,
            user_text=user_text,
            image_data_urls=image_urls,
            json_schema=VISION_SCHEMA,
            max_output_tokens=700,
        )

    if show_debug:
        debug_box.info(f"Vision raw (first 800 chars):\n{(vision_raw or '')[:800]}")

    if vision_parsed is None:
        st.error("Vision extractor failed (JSON parse). See debug output above.")
        st.stop()

    st.success("Vision features extracted ✅")

    # 2) RAG retrieve (optional)
    rag_block = ""
    rag_status = {"enabled": False, "hits": 0, "dim": None, "error": None}
    if use_rag:
        with st.spinner("Step 2/3: RAG retrieval..."):
            query = (clinical_note or "").strip()
            query += "\nVISION FEATURES:\n" + json.dumps(vision_parsed, ensure_ascii=False)
            query += "\nretina differential diagnosis imaging"
            rag_block, rag_status = rag_retrieve(client, query, k=rag_k)

    # 3) Final reasoning
    with st.spinner("Step 3/3: Final reasoning..."):
        final_user = FINAL_USER_TEMPLATE.format(
            clinical_note=clinical_note.strip() if clinical_note.strip() else "(not provided)",
            vision_json=json.dumps(vision_parsed, ensure_ascii=False, indent=2),
            rag_block=(rag_block if rag_block else "(No RAG cards retrieved)"),
        )
        final_parsed, final_raw = responses_json_call(
            client=client,
            model=final_model,
            system_text=FINAL_SYSTEM,
            user_text=final_user,
            image_data_urls=[],  # IMPORTANT: final step uses extracted features; no need to resend images
            json_schema=FINAL_SCHEMA,
            max_output_tokens=1400,
        )

    if final_parsed is None:
        st.error("Final reasoning failed (JSON parse). Showing raw output below.")
        st.text(final_raw)
        st.stop()

    # OUTPUTS
    st.subheader("Human report")
    st.markdown(final_parsed.get("human_report", "(no report)"))

    if show_debug:
        st.subheader("Vision features (debug)")
        st.json(vision_parsed)

        st.subheader("RAG status (debug)")
        st.json(rag_status)
        if rag_block:
            with st.expander("Retrieved cards (debug)", expanded=False):
                st.text(rag_block[:6000])

        st.subheader("Final JSON (debug)")
        st.json(final_parsed)
