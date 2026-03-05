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
APP_TITLE = "RetinaGPT (2-pass) — Basic (Adaptive)"
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
    """Fallback: find first {...} and try json.loads."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return json_loads_safe(text[start : end + 1])


def to_input_content(user_text: str, image_data_urls: List[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_text}]
    for url in image_data_urls:
        content.append({"type": "input_image", "image_url": url, "detail": "high"})
    return content


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
# PROMPTS
# -----------------------------
VISION_SYSTEM = """You are a retina imaging feature extractor for DE-IDENTIFIED ophthalmic images (retina/OCT/FA/FAF/OCTA).
These are not faces/people. Do not identify anyone.
Return STRICT JSON ONLY. No markdown. No commentary.
If uncertain, use 'UNCERTAIN' or empty lists.
"""

VISION_USER_TEMPLATE = """Clinical note (may be brief):
{clinical_note}

Extract retina-specific morphology features (be literal):
- modality guess (Fundus/OCT/FAF/FA/OCTA/ICGA)
- hemorrhage vs pigment vs atrophy
- global hypopigmentation / tessellation / choroidal show-through (albinism clues)
- foveal hypoplasia suspected? (YES/NO/UNCERTAIN)
- vascular tortuosity / caliber abnormalities / ischemia clues
- optic disc anomalies (pit/coloboma/hypoplasia)
- lesion shape/location keywords (e.g., torpedo/teardrop temporal to fovea)

Return JSON with keys:
modality_guess (list[str]),
global_distribution (string),
fundus_pigmentation (string),
choroidal_vessel_visibility (string),
macular_reflex (string),
foveal_hypoplasia_suspected (string),
optic_disc_notes (list[str]),
vascular_pattern (list[str]),
hemorrhage_exudates (string),
lesion_shape_keywords (list[str]),
lesion_location_keywords (list[str]),
outer_retina_rpe_clues (list[str]),
key_findings_bullets (list[str]),
quality_notes (list[str]).
"""

FINAL_SYSTEM = """You are RetinaGPT (educational only).
You will receive:
(1) Clinical note
(2) Vision-extracted features (structured JSON)
(3) Optional REFERENCE CARDS (RAG)

Use vision features as PRIMARY observation. Use RAG only to refine discriminators/pitfalls.
No patient-identifying content. No individualized treatment dosing.
Return STRICT JSON ONLY with keys:
most_likely, differential, confidence_level, urgency, human_report, next_best_requests,
modalities_detected, missing_modalities_suggested, patterns, feature_checklist, case_summary.
"""

FINAL_USER_TEMPLATE = """CLINICAL NOTE:
{clinical_note}

VISION FEATURES (PRIMARY OBSERVATION SOURCE):
{vision_json}

{rag_block}

Now produce a structured retina differential + human report.
Return JSON ONLY (no markdown).
"""


# -----------------------------
# ADAPTIVE Responses call
# -----------------------------
def responses_call_json_only(
    client: OpenAI,
    model: str,
    system_text: str,
    user_text: str,
    image_data_urls: List[str],
    max_output_tokens: int = 900,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Works on older/newer openai libs:
    - First tries Responses API with response_format (if supported).
    - If library rejects response_format, retries WITHOUT it.
    Always instructs JSON-only, then parses.
    """
    input_payload = [
        {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
        {"role": "user", "content": to_input_content(user_text, image_data_urls)},
    ]

    # 1) Try with response_format (newer)
    try:
        resp = client.responses.create(
            model=model,
            input=input_payload,
            response_format={"type": "json_object"},
            max_output_tokens=max_output_tokens,
        )
        raw = getattr(resp, "output_text", "") or ""
        parsed = json_loads_safe(raw) or extract_json_from_text(raw)
        return (parsed, raw)
    except TypeError as e:
        # Older library: unexpected keyword response_format
        err_txt = f"{e}"
        # 2) Retry without response_format
        try:
            resp = client.responses.create(
                model=model,
                input=input_payload,
                max_output_tokens=max_output_tokens,
            )
            raw = getattr(resp, "output_text", "") or ""
            parsed = json_loads_safe(raw) or extract_json_from_text(raw)
            return (parsed, f"(fallback-no-response_format)\n{raw}")
        except Exception as e2:
            return (None, f"ERROR (no response_format supported): {err_txt} | retry error: {e2}")
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

    image_urls: List[str] = []
    for f in uploaded:
        mime = f.type or "image/jpeg"
        image_urls.append(b64_data_url(f.getvalue(), mime))

    note = (clinical_note.strip() + ("\nHint: " + optional_note if optional_note else "")).strip()

    # 1) Vision extract (ChatGPT vision)
    with st.spinner("Step 1/3: Vision extractor running..."):
        v_user = VISION_USER_TEMPLATE.format(clinical_note=note if note else "(not provided)")
        vision_parsed, vision_raw = responses_call_json_only(
            client=client,
            model=vision_model,
            system_text=VISION_SYSTEM,
            user_text=v_user,
            image_data_urls=image_urls,
            max_output_tokens=700,
        )

    if show_debug:
        debug_box.info(f"Vision raw (first 900 chars):\n{(vision_raw or '')[:900]}")

    if vision_parsed is None:
        st.error("Vision extractor failed. See debug output above.")
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

    # 3) Final reasoning (NO images; uses vision JSON)
    with st.spinner("Step 3/3: Final reasoning..."):
        f_user = FINAL_USER_TEMPLATE.format(
            clinical_note=clinical_note.strip() if clinical_note.strip() else "(not provided)",
            vision_json=json.dumps(vision_parsed, ensure_ascii=False, indent=2),
            rag_block=(rag_block if rag_block else "(No RAG cards retrieved)"),
        )
        final_parsed, final_raw = responses_call_json_only(
            client=client,
            model=final_model,
            system_text=FINAL_SYSTEM,
            user_text=f_user,
            image_data_urls=[],
            max_output_tokens=1400,
        )

    if final_parsed is None:
        st.error("Final reasoning failed. Showing raw output below.")
        st.text(final_raw)
        st.stop()

    # OUTPUT (best-effort fields)
    human_report = final_parsed.get("human_report") or "(No human_report field returned. Showing full JSON below.)"
    st.subheader("Human report")
    st.markdown(human_report)

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
