import os
import json
import base64
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from openai import OpenAI

# Optional RAG deps (FAISS)
try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    faiss = None
    np = None

# Optional image preprocessing (CLAHE)
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

# -----------------------------
# CONFIG
# -----------------------------
APP_TITLE = "RetinaGPT v3 (Vision → RAG → Reasoning)"
APP_SUBTITLE = "De-identified retinal images • Educational only"
DEFAULT_MODEL_VISION = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
DEFAULT_MODEL_REASON = os.getenv("OPENAI_REASON_MODEL", "gpt-4o-mini")

RAG_DIR = os.getenv("RAG_DIR", "data")  # expects data/index.faiss + data/meta.json
MAX_RAG_HITS = int(os.getenv("MAX_RAG_HITS", "6"))

st.set_page_config(page_title=APP_TITLE, page_icon="👁️", layout="wide")

# -----------------------------
# SAFETY PROMPT (shared)
# -----------------------------
SAFETY_CONTEXT = """
You will receive DE-IDENTIFIED ophthalmic images (retinal fundus/OCT/FAF/angiography).
These are not photos of faces/people. Do not identify any person.
Educational discussion only. No patient-specific treatment instructions.
"""

# -----------------------------
# VISION EXTRACTOR PROMPT
# (NO diagnosis, strict JSON only)
# -----------------------------
VISION_EXTRACTOR_SYSTEM = f"""
{SAFETY_CONTEXT}

ROLE
You are a retina imaging morphology extractor.

TASK
- Look at the uploaded ophthalmic image(s).
- Output ONLY strict JSON (no markdown, no extra text).
- Do NOT give a diagnosis. Do NOT mention patient identity.
- Focus on morphology: pigmentation level, choroidal vessel visibility, optic disc, macula/fovea appearance,
  hemorrhage/exudates, edema, scars/atrophy, traction, lesion shape/location.

OUTPUT JSON SCHEMA (STRICT):
{{
  "modality_guess": ["Fundus"|"OCT"|"FAF"|"FA"|"ICGA"|"OCTA"],
  "global_distribution": "NORMAL|HYPERPIGMENTED|HYPOPIGMENTED|MIXED|UNCERTAIN",
  "fundus_pigmentation": "NORMAL|REDUCED|INCREASED|UNCERTAIN",
  "choroidal_vessel_visibility": "LOW|MODERATE|HIGH|UNCERTAIN",
  "macular_reflex": "PRESENT|REDUCED|ABSENT|UNCERTAIN",
  "foveal_hypoplasia_suspected": "YES|NO|UNCERTAIN",
  "optic_disc_notes": ["..."],
  "vascular_pattern": ["NORMAL|TORTUOSITY|TELANGIECTASIA|NONPERFUSION_SUSPECTED|OTHER|UNCERTAIN"],
  "hemorrhage_exudates": "PRESENT|ABSENT|UNCERTAIN",
  "lesion_shape_keywords": ["..."],
  "lesion_location_keywords": ["..."],
  "outer_retina_rpe_clues": ["..."],
  "quality_notes": ["..."],
  "key_findings_bullets": ["...","...","..."]
}}
"""

# -----------------------------
# FINAL REASONER PROMPT
# (Uses extractor + RAG + clinical)
# -----------------------------
FINAL_REASONER_SYSTEM = f"""
{SAFETY_CONTEXT}

You are RetinaGPT v3: retina subspecialty educational imaging discussion + differential diagnosis assistant.

PURPOSE
Provide structured academic discussion of retinal imaging findings and differential diagnostic reasoning
for educational purposes only.

STYLE
- Formal medical English, objective, concise.
- Describe morphology first, then ranked differential.
- Use retina subspecialty terminology.

RAG RULE
If “REFERENCE CARDS (RAG)” are provided, treat them as the primary factual source for discriminators/pitfalls/work-up.
If RAG conflicts with image findings, explicitly state the discrepancy.

OUTPUT REQUIREMENTS (IMPORTANT)
Return TWO blocks:
(A) Strict JSON inside a ```json code fence
(B) Human-readable report with numbered headings.

(A) JSON schema:
{{
  "case_summary": {{
    "age": "string or null",
    "sex": "string or null",
    "symptoms": "string or null",
    "duration": "string or null",
    "laterality": "string or null",
    "history": "string or null"
  }},
  "modalities_detected": ["Fundus","OCT","FAF","FA","ICGA","OCTA"],
  "patterns": [{{"name":"string","confidence":0.0}}],
  "feature_checklist": {{
    "subretinal_fluid": "PRESENT|ABSENT|UNCERTAIN",
    "intraretinal_fluid": "PRESENT|ABSENT|UNCERTAIN",
    "hemorrhage_exudation": "PRESENT|ABSENT|UNCERTAIN",
    "inner_retinal_ischemia": "PRESENT|ABSENT|UNCERTAIN",
    "ez_disruption": "PRESENT|ABSENT|UNCERTAIN",
    "rpe_atrophy_hypertransmission": "PRESENT|ABSENT|UNCERTAIN",
    "vitelliform_material": "PRESENT|ABSENT|UNCERTAIN",
    "inflammatory_signs": "PRESENT|ABSENT|UNCERTAIN",
    "m_nv_suspected": "PRESENT|ABSENT|UNCERTAIN",
    "true_mass_lesion_suspected": "PRESENT|ABSENT|UNCERTAIN"
  }},
  "most_likely": {{"diagnosis":"string","probability":0.0}},
  "differential": [
    {{"diagnosis":"string","probability":0.0,"for":["..."],"against":["..."]}}
  ],
  "confidence_level": "LOW|MODERATE|HIGH",
  "next_best_requests": [{{"request":"string","why":"string"}}],
  "urgency": "CRITICAL|URGENT|ROUTINE"
}}

(B) Human report format:
1) Clinical Triage
2) Detected Modalities
3) Imaging-derived Key Findings (from extractor)
4) Integrated Pattern Discussion
5) Most Likely Diagnosis
6) Differential Diagnosis (ranked)
7) Confidence + Next Steps (general)
8) Educational limitation statement
"""

# -----------------------------
# HELPERS
# -----------------------------
def safe_get_api_key() -> Optional[str]:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def img_to_data_url(file_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(file_bytes).decode('utf-8')}"


def maybe_clahe_bytes(file_bytes: bytes) -> bytes:
    """Apply CLAHE to improve contrast (can hurt some images). Returns bytes (JPEG)."""
    if cv2 is None:
        return file_bytes
    try:
        arr = np.frombuffer(file_bytes, dtype=np.uint8) if np is not None else None
        if arr is None:
            return file_bytes
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return file_bytes

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return file_bytes
        return enc.tobytes()
    except Exception:
        return file_bytes


def parse_json_strict(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


def parse_json_block_from_markdown(text: str) -> Optional[Dict[str, Any]]:
    """Finds ```json ... ``` block and parses."""
    if "```json" not in text:
        return None
    try:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        block = text[start:end].strip()
        return json.loads(block)
    except Exception:
        return None


# -----------------------------
# RAG (FAISS) OPTIONAL
# data/meta.json: list[{"title":..., "text":...}]
# -----------------------------
@st.cache_resource
def load_rag_index() -> Tuple[bool, Optional[Any], Optional[List[Dict[str, Any]]]]:
    if faiss is None or np is None:
        return (False, None, None)

    index_path = os.path.join(RAG_DIR, "index.faiss")
    meta_path = os.path.join(RAG_DIR, "meta.json")

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
        emb = client.embeddings.create(model="text-embedding-3-small", input=text)
        return emb.data[0].embedding
    except Exception:
        return None


def rag_retrieve(client: OpenAI, query: str, k: int) -> Tuple[str, Dict[str, Any]]:
    enabled, idx, meta = load_rag_index()
    status: Dict[str, Any] = {"enabled": False, "hits": 0, "error": None, "dim": None}

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

        hits = []
        for j, row_id in enumerate(I[0].tolist()):
            if 0 <= row_id < len(meta):
                chunk = meta[row_id]
                title = chunk.get("title", f"chunk_{row_id}")
                text = chunk.get("text", "")
                hits.append((title, text, float(D[0][j])))

        status["hits"] = len(hits)
        if not hits:
            return ("", status)

        lines = ["REFERENCE CARDS (RAG):"]
        for n, (title, text, _dist) in enumerate(hits, start=1):
            lines.append(f"\n--- CARD {n}: {title} ---\n{text}\n")
        return ("\n".join(lines), status)

    except Exception as e:
        status["error"] = str(e)
        return ("", status)


# -----------------------------
# OPENAI CALLS (Responses API)
# -----------------------------
def responses_vision_extract(
    client: OpenAI,
    model: str,
    clinical_text: str,
    image_data_urls: List[str],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Returns (parsed_json, raw_text). Raw text is the model output string."""
    user_text = (
        "CLINICAL (may be empty):\n"
        + (clinical_text.strip() if clinical_text.strip() else "(none)")
        + "\n\nREMINDER: Output MUST be strict JSON only."
    )

    content = [{"type": "input_text", "text": user_text}]
    for url in image_data_urls:
        content.append({"type": "input_image", "image_url": url, "detail": "high"})

    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "system", "content": [{"type": "input_text", "text": VISION_EXTRACTOR_SYSTEM}]},
                   {"role": "user", "content": content}],
            temperature=0.0,
            max_output_tokens=500,
        )
        raw = resp.output_text or ""
        parsed = parse_json_strict(raw.strip())
        return (parsed, raw)
    except Exception as e:
        return (None, f"VISION EXTRACT ERROR: {e}")


def responses_final_reason(
    client: OpenAI,
    model: str,
    clinical_text: str,
    extractor_json: Dict[str, Any],
    rag_context: str,
    extra_user_note: str = "",
) -> str:
    user_payload = {
        "clinical_text": clinical_text.strip() if clinical_text.strip() else None,
        "extractor_findings": extractor_json,
        "extra_note": extra_user_note.strip() if extra_user_note.strip() else None,
    }

    user_text = (
        "Use the extractor_findings as the PRIMARY image evidence. "
        "Do not contradict it unless you explicitly explain why.\n\n"
        "USER PAYLOAD (JSON):\n"
        + json.dumps(user_payload, ensure_ascii=False, indent=2)
    )

    sys_parts = [{"type": "input_text", "text": FINAL_REASONER_SYSTEM}]
    if rag_context:
        sys_parts.append({"type": "input_text", "text": rag_context})

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": sys_parts},
                {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
            ],
            temperature=0.2,
            max_output_tokens=1800,
        )
        return resp.output_text or ""
    except Exception as e:
        return f"```json\n{json.dumps({'error':'FINAL REASON ERROR','details':str(e)}, ensure_ascii=False, indent=2)}\n```\n\nFinal reasoner error: {e}"


# -----------------------------
# UI
# -----------------------------
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

api_key = safe_get_api_key()
if not api_key:
    st.error("OPENAI_API_KEY bulunamadı. Streamlit Secrets veya environment variable olarak ekle.")
    st.stop()

client = OpenAI(api_key=api_key)

with st.sidebar:
    st.header("Settings")
    vision_model = st.selectbox("Vision model (extractor)", [DEFAULT_MODEL_VISION, "gpt-4o-mini"], index=0)
    reason_model = st.selectbox("Reasoning model (final)", [DEFAULT_MODEL_REASON, "gpt-4o"], index=0)
    use_rag = st.checkbox("Enable RAG (FAISS)", value=True)
    rag_k = st.slider("RAG hits (k)", 2, 10, MAX_RAG_HITS, 1)
    use_clahe = st.checkbox("Preprocess images (CLAHE)", value=False)
    st.divider()
    debug = st.checkbox("Show debug panels", value=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Clinical metadata")
    clinical_text = st.text_area(
        "Age, Sex, Symptoms, Duration, Laterality, History",
        height=140,
        placeholder="Example:\nAge: 30\nSex: Male\nSymptoms: low vision since childhood + nystagmus\nDuration: chronic\nLaterality: bilateral\nHistory: ..."
    )
    st.subheader("Upload retinal imaging")
    files = st.file_uploader(
        "Fundus / OCT / FAF / FA / OCTA (jpg/png/webp)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )
    extra_note = st.text_input("Optional note (e.g., 'suspect albinism?')", value="")

with col2:
    st.subheader("Run")
    run = st.button("🔎 Analyze (Vision → RAG → Reasoning)", type="primary")

    st.divider()
    if debug:
        st.subheader("Debug")
        extractor_box = st.empty()
        rag_box = st.empty()

# -----------------------------
# RUN
# -----------------------------
if run:
    if not files:
        st.warning("En az 1 görüntü yüklemen önerilir.")
    # 1) Build data URLs (+ optional CLAHE)
    image_urls: List[str] = []
    for f in (files or []):
        b = f.getvalue()
        if use_clahe:
            b = maybe_clahe_bytes(b)
        mime = f.type or "image/jpeg"
        image_urls.append(img_to_data_url(b, mime))

    # 2) Vision Extract (strict JSON)
    with st.spinner("Step 1/3: Extracting image findings (vision)..."):
        extractor_json, extractor_raw = responses_vision_extract(
            client=client,
            model=vision_model,
            clinical_text=clinical_text,
            image_data_urls=image_urls,
        )

    if debug:
        extractor_box.info("Extractor raw output (first 1200 chars):\n" + (extractor_raw[:1200] if extractor_raw else "(empty)"))

    if extractor_json is None:
        st.error("Extractor JSON parse başarısız. (Model strict JSON üretmedi veya boş döndü.)")
        st.stop()

    # 3) RAG Retrieve (optional) using clinical + extracted bullets
    rag_context = ""
    rag_status = {"enabled": False, "hits": 0, "error": None}
    if use_rag:
        with st.spinner("Step 2/3: Retrieving reference cards (RAG)..."):
            # Build a strong retrieval query from extractor bullets
            bullets = extractor_json.get("key_findings_bullets") or []
            bullets_txt = "\n".join([f"- {x}" for x in bullets[:12]])
            query = (clinical_text.strip() + "\n\n" + bullets_txt).strip()
            # Add anchor keywords that help rare entities
            query += "\n\nretina fundus hypopigmentation choroidal vessels foveal hypoplasia nystagmus albinism"
            rag_context, rag_status = rag_retrieve(client, query, k=rag_k)

        if debug:
            msg = f"RAG enabled={rag_status.get('enabled')} hits={rag_status.get('hits')} error={rag_status.get('error')}"
            rag_box.success(msg) if rag_status.get("enabled") else rag_box.warning(msg)
            with st.expander("Show retrieved cards (debug)", expanded=False):
                st.text(rag_context if rag_context else "(no cards)")

    # 4) Final Reasoning
    with st.spinner("Step 3/3: Writing final report..."):
        final_text = responses_final_reason(
            client=client,
            model=reason_model,
            clinical_text=clinical_text,
            extractor_json=extractor_json,
            rag_context=rag_context,
            extra_user_note=extra_note,
        )

    st.subheader("Assistant output")
    st.markdown(final_text)

    parsed = parse_json_block_from_markdown(final_text)
    if parsed:
        st.subheader("Structured JSON (debug)")
        st.json(parsed)
    else:
        st.warning("Final output JSON block parse edilemedi (model JSON bloğunu format dışı yazmış olabilir).")

    if debug:
        st.subheader("Extractor findings (parsed)")
        st.json(extractor_json)
