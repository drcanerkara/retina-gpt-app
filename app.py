import os, json, base64
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

APP_TITLE = "RetinaGPT — Vision-first (ChatGPT-like)"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # vision-capable
RAG_DIR = os.getenv("RAG_DIR", "data")
MAX_RAG_HITS = int(os.getenv("MAX_RAG_HITS", "6"))

st.set_page_config(page_title=APP_TITLE, page_icon="👁️", layout="wide")


def safe_api_key() -> Optional[str]:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def b64_image(file_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(file_bytes).decode('utf-8')}"


@st.cache_resource
def load_rag_index(rag_dir: str) -> Tuple[bool, Optional[Any], Optional[List[Dict[str, Any]]]]:
    if faiss is None or np is None:
        return (False, None, None)
    idx_path = os.path.join(rag_dir, "index.faiss")
    meta_path = os.path.join(rag_dir, "meta.json")
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        return (False, None, None)
    try:
        idx = faiss.read_index(idx_path)
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
    enabled, idx, meta = load_rag_index(RAG_DIR)
    status = {"enabled": False, "hits": 0, "dim": None, "error": None}
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
            if len(txt) > 1600:
                txt = txt[:1600] + "..."
            lines.append(f"\n--- CARD {n}: {title} ---\n{txt}\n")
        return ("\n".join(lines), status)

    except Exception as e:
        status["error"] = str(e)
        return ("", status)


VISION_FIRST_SYSTEM = """
You are a retina subspecialist analyzing DE-IDENTIFIED ophthalmic images (retina). Not faces/people.
Task: produce a high-fidelity morphology description and a SHORT ranked differential.
Be extremely careful distinguishing hemorrhage vs pigment vs atrophy.
If you see sectoral hemorrhage, venous dilation/tortuosity, cotton-wool spots, or capillary nonperfusion signs,
consider retinal vein occlusion patterns.
Return STRICT JSON ONLY (no markdown).
JSON keys:
- key_morphology_bullets: list[str] (8-15)
- strongest_patterns: list[{"name":str,"confidence":float}]
- top_differential: list[{"dx":str,"prob":float,"for":list[str],"against":list[str]}]
- confidence_level: "LOW"|"MODERATE"|"HIGH"
- needs_rag: true|false
- rag_query: str
"""

FINAL_SYSTEM = """
You are RetinaGPT (educational only). You will receive:
(1) Clinical note
(2) Images
(3) Vision-first JSON (primary)
(4) Optional RAG cards (secondary)

You MUST prioritize IMAGE + morphology. Use RAG only to refine discriminators/pitfalls.
Return TWO blocks:
(A) STRICT JSON in ```json fence
(B) Human-readable report.

The JSON must include:
case_summary, modalities_detected, missing_modalities_suggested, patterns, feature_checklist,
most_likely, differential, confidence_level, urgency, next_best_requests.
"""

def call_chat_completion(client: OpenAI, model: str, messages: List[Dict[str, Any]], max_tokens: int = 1200, temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def parse_json_strict(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


def parse_json_fenced(text: str) -> Optional[Dict[str, Any]]:
    if "```json" not in text:
        return None
    try:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        block = text[start:end].strip()
        return json.loads(block)
    except Exception:
        return None


st.title(APP_TITLE)

api_key = safe_api_key()
if not api_key:
    st.error("OPENAI_API_KEY missing (Streamlit secrets or env var).")
    st.stop()

client = OpenAI(api_key=api_key)

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", options=[DEFAULT_MODEL, "gpt-4o", "gpt-4o-mini"], index=0)
    use_rag = st.checkbox("Enable RAG (FAISS)", value=True)
    rag_k = st.slider("RAG hits (k)", 2, 10, 6, 1)
    st.caption("Tip: For fine vascular cases, keep model=gpt-4o.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    clinical_note = st.text_area("Clinical note", height=140, placeholder="Age/sex/symptoms/duration/laterality/history...")
    files = st.file_uploader("Upload images (jpg/png/webp)", type=["jpg","jpeg","png","webp"], accept_multiple_files=True)

with col2:
    run = st.button("🔎 Analyze", type="primary")

if run:
    if not files:
        st.warning("Upload at least 1 image.")
        st.stop()

    # Build image payload
    image_content = []
    for f in files:
        mime = f.type or "image/jpeg"
        url = b64_image(f.getvalue(), mime)
        image_content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})

    # -------------------------
    # PASS 1: Vision-first (ChatGPT-like)
    # -------------------------
    with st.spinner("Pass 1/3: Vision-first analysis (images + morphology)..."):
        user_text_1 = "CLINICAL NOTE:\n" + (clinical_note.strip() if clinical_note.strip() else "(not provided)")
        msgs1 = [
            {"role": "system", "content": VISION_FIRST_SYSTEM},
            {"role": "user", "content": [{"type":"text","text": user_text_1}] + image_content},
        ]
        raw1 = call_chat_completion(client, model, msgs1, max_tokens=900, temperature=0.0)

        # Expect JSON only; fallback: try to extract first {..}
        vision_json = parse_json_strict(raw1)
        if vision_json is None:
            # crude extraction
            s = raw1.find("{")
            e = raw1.rfind("}")
            if s != -1 and e != -1 and e > s:
                vision_json = parse_json_strict(raw1[s:e+1])

        if vision_json is None:
            st.error("Vision-first JSON parse failed. Showing raw output:")
            st.text(raw1[:4000])
            st.stop()

    st.success("Pass 1 done ✅")
    with st.expander("Vision-first JSON (debug)", expanded=False):
        st.json(vision_json)

    # -------------------------
    # PASS 2: Conditional RAG
    # -------------------------
    rag_block = ""
    rag_status = {"enabled": False, "hits": 0, "dim": None, "error": None}

    needs_rag = bool(vision_json.get("needs_rag", False))
    rag_query = (vision_json.get("rag_query") or "").strip()

    if use_rag and (needs_rag or (vision_json.get("confidence_level") in ["LOW"])):
        with st.spinner("Pass 2/3: RAG retrieval (conditional)..."):
            if not rag_query:
                rag_query = json.dumps(vision_json, ensure_ascii=False) + "\nretina differential diagnosis imaging"
            rag_block, rag_status = rag_retrieve(client, rag_query, k=rag_k)

    with st.expander("RAG status (debug)", expanded=False):
        st.json(rag_status)
        if rag_block:
            st.text(rag_block[:6000])

    # -------------------------
    # PASS 3: Final (images + vision_json + rag)
    # -------------------------
    with st.spinner("Pass 3/3: Final report (images + RAG)..."):
        user_text_3 = (
            "CLINICAL NOTE:\n"
            + (clinical_note.strip() if clinical_note.strip() else "(not provided)")
            + "\n\nVISION-FIRST JSON (PRIMARY):\n"
            + json.dumps(vision_json, ensure_ascii=False, indent=2)
            + "\n\n"
            + (rag_block if rag_block else "(No RAG cards used)")
        )

        msgs3 = [
            {"role": "system", "content": FINAL_SYSTEM},
            {"role": "user", "content": [{"type":"text","text": user_text_3}] + image_content},
        ]
        final_text = call_chat_completion(client, model, msgs3, max_tokens=1800, temperature=0.2)

    st.subheader("Assistant output")
    st.markdown(final_text)

    parsed = parse_json_fenced(final_text)
    if parsed:
        st.subheader("Structured JSON (debug)")
        st.json(parsed)
    else:
        st.warning("Final JSON fenced block not found/parse failed (model format drift).")
