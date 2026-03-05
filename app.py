import os
import json
import base64
import re
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
APP_TITLE = "RetinaGPT (Vision-first, ChatGPT-like)"
DEFAULT_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
DEFAULT_FINAL_MODEL = os.getenv("OPENAI_FINAL_MODEL", "gpt-4o")  # gpt-4o-mini is cheaper but weaker for vision-heavy reasoning
RAG_DIR = os.getenv("RAG_DIR", "data")
MAX_RAG_HITS = int(os.getenv("MAX_RAG_HITS", "6"))

st.set_page_config(page_title=APP_TITLE, page_icon="👁️", layout="wide")


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


def b64_data_url(file_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(file_bytes).decode('utf-8')}"


def extract_json_anywhere(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust fallback: finds the first {...} JSON object in text and parses it.
    """
    if not text:
        return None
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find first JSON object-like block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    block = m.group(0)
    try:
        return json.loads(block)
    except Exception:
        return None


def build_image_content(files) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if not files:
        return content
    for f in files:
        mime = f.type or "image/jpeg"
        url = b64_data_url(f.getvalue(), mime)
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


# -----------------------------
# RAG (FAISS)
# -----------------------------
@st.cache_resource
def load_rag_index(rag_dir: str) -> Tuple[bool, Optional[Any], Optional[List[Dict[str, Any]]]]:
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
        emb = client.embeddings.create(model="text-embedding-3-small", input=text)
        return emb.data[0].embedding
    except Exception:
        return None


def rag_retrieve(client: OpenAI, query: str, k: int = MAX_RAG_HITS) -> Tuple[str, Dict[str, Any]]:
    enabled, idx, meta = load_rag_index(RAG_DIR)
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

        context_lines = ["REFERENCE CARDS (RAG):"]
        for n, (title, text, dist) in enumerate(hits, start=1):
            context_lines.append(f"\n--- CARD {n}: {title} (dist={dist:.4f}) ---\n{text}\n")

        return ("\n".join(context_lines), status)

    except Exception as e:
        status["error"] = str(e)
        return ("", status)


# -----------------------------
# PROMPTS
# -----------------------------
VISION_SYSTEM = (
    "You are a retina subspecialty imaging describer. "
    "The user provides DE-IDENTIFIED ophthalmic images (fundus/OCT/FAF/FA/OCTA). "
    "First: describe morphology precisely. Then: propose a ranked differential. "
    "Avoid over-commitment; use probabilities. Educational only."
)

# A compact JSON schema we ask the model to output.
# (We still implement fallback parsing because models sometimes deviate.)
VISION_JSON_INSTRUCTIONS = """
Return ONLY valid JSON (no markdown).
JSON schema:
{
  "modalities_detected": ["Fundus","OCT","FAF","FA","ICGA","OCTA"],
  "image_quality": "GOOD|FAIR|POOR",
  "key_findings": ["..."],
  "most_likely": {"diagnosis": "...", "probability": 0.0},
  "differential": [
    {"diagnosis":"...", "probability":0.0, "for":["..."], "against":["..."]}
  ],
  "missing_next": [{"request":"...", "why":"..."}]
}
"""

FINAL_SYSTEM = """
You are RetinaGPT (educational). Use retina subspecialty terminology.
You will be given:
(1) Clinical metadata text
(2) Vision-first JSON findings from a strong model
(3) Optional REFERENCE CARDS (RAG)
Task: produce a final structured human report + a final JSON summary.
If RAG conflicts with vision findings, say so explicitly and prioritize image morphology.
No patient-specific treatment dosing; educational only.
Return TWO blocks:
(A) JSON in a ```json``` fence
(B) Human report with headings.
"""


# -----------------------------
# MODEL CALLS
# -----------------------------
def call_vision_json(
    client: OpenAI,
    model: str,
    clinical_text: str,
    optional_note: str,
    image_content: List[Dict[str, Any]],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    user_text = (
        "DE-IDENTIFIED OPHTHALMIC IMAGES (retina) — not faces/people.\n\n"
        f"CLINICAL METADATA:\n{clinical_text.strip() if clinical_text.strip() else '(none)'}\n\n"
        f"OPTIONAL NOTE:\n{optional_note.strip() if optional_note.strip() else '(none)'}\n\n"
        f"{VISION_JSON_INSTRUCTIONS}"
    )

    messages = [
        {"role": "system", "content": VISION_SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": user_text}] + (image_content or [])},
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=900,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = extract_json_anywhere(raw)
        return raw, parsed
    except Exception as e:
        return f"ERROR: {e}", None


def call_final_report(
    client: OpenAI,
    model: str,
    clinical_text: str,
    optional_note: str,
    vision_json: Dict[str, Any],
    rag_context: str,
) -> str:
    payload = {
        "clinical_text": clinical_text,
        "optional_note": optional_note,
        "vision_first": vision_json,
    }

    user_text = (
        "INPUTS:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\n"
        + ("RAG CONTEXT:\n" + rag_context if rag_context else "RAG CONTEXT: (none)")
    )

    messages = [
        {"role": "system", "content": FINAL_SYSTEM},
        {"role": "user", "content": user_text},
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=1700,
    )
    return (resp.choices[0].message.content or "").strip()


# -----------------------------
# UI
# -----------------------------
st.title(APP_TITLE)
st.caption("Vision-first (ChatGPT-like) → optional RAG → final report")

api_key = safe_get_api_key()
if not api_key:
    st.error("OPENAI_API_KEY bulunamadı. Streamlit Secrets veya environment variable olarak ekle.")
    st.stop()

client = OpenAI(api_key=api_key)

with st.sidebar:
    st.header("Settings")
    vision_model = st.selectbox("Vision model (analysis)", options=[DEFAULT_VISION_MODEL, "gpt-4o-mini"], index=0)
    final_model = st.selectbox("Final model (report)", options=[DEFAULT_FINAL_MODEL, "gpt-4o-mini", "gpt-4o"], index=0)
    use_rag = st.checkbox("Enable RAG (FAISS)", value=True)
    rag_k = st.slider("RAG hits (k)", 2, 10, MAX_RAG_HITS, 1)
    st.caption("Not: CLAHE kapalı. Fundus/FFA’da artefakt yaratabiliyor.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Clinical metadata")
    clinical_text = st.text_area(
        "Age / Sex / symptoms / duration / laterality / history",
        height=140,
        placeholder="Example:\nAge: 60\nSex: Male\nSymptoms: sudden vision loss\nDuration: 2 days\nLaterality: unilateral\nHistory: HT, DM..."
    )
    optional_note = st.text_input("Optional hint (e.g., 'suspect albinism?')", value="")

    st.subheader("Upload retinal imaging")
    files = st.file_uploader(
        "Fundus / OCT / FAF / FA / OCTA (jpg/png/webp)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    run = st.button("🔎 Analyze", type="primary")

with col2:
    st.subheader("Output")
    st.caption("First: strong vision JSON. Then: (optional) RAG. Then: final report.")

    debug_exp = st.expander("Debug (vision JSON + RAG)", expanded=True)
    out_exp = st.expander("Final report", expanded=True)

if run:
    img_payload = build_image_content(files)

    if not img_payload:
        st.warning("En az 1 görüntü yüklemen gerekiyor.")
        st.stop()

    # 1) Vision-first extraction (ChatGPT-like)
    with st.spinner("Vision-first analysis (ChatGPT-like) running..."):
        vision_raw, vision_parsed = call_vision_json(
            client=client,
            model=vision_model,
            clinical_text=clinical_text,
            optional_note=optional_note,
            image_content=img_payload,
        )

    with debug_exp:
        st.markdown("### Vision raw output")
        st.code(vision_raw[:2000] + ("..." if len(vision_raw) > 2000 else ""))

        st.markdown("### Vision parsed JSON")
        if vision_parsed:
            st.json(vision_parsed)
        else:
            st.error("Vision JSON parse failed. (Fallback da çıkaramadı.)")
            st.stop()

    # 2) RAG retrieval (optional)
    rag_context = ""
    rag_status = {"enabled": False, "hits": 0, "index_dim": None, "error": None}
    if use_rag:
        # Use the vision findings as the retrieval query (this is the key!)
        q = "Retina imaging differential diagnosis.\n"
        q += "Key findings:\n" + "\n".join(vision_parsed.get("key_findings", [])[:12])
        if clinical_text.strip():
            q += "\nClinical:\n" + clinical_text.strip()

        with st.spinner("RAG retrieve running..."):
            rag_context, rag_status = rag_retrieve(client, q, k=rag_k)

        with debug_exp:
            st.markdown("### RAG status")
            st.write(rag_status)
            st.markdown("### Retrieved cards")
            st.text(rag_context if rag_context else "(No RAG context)")

    # 3) Final report
    with st.spinner("Final report generating..."):
        final_text = call_final_report(
            client=client,
            model=final_model,
            clinical_text=clinical_text,
            optional_note=optional_note,
            vision_json=vision_parsed,
            rag_context=rag_context,
        )

    with out_exp:
        st.markdown(final_text)
