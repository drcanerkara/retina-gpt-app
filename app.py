import os
import json
import base64
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI

# Optional RAG deps (FAISS)
try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    faiss = None
    np = None

# -----------------------------
# CONFIG
# -----------------------------
APP_TITLE = "RetinaGPT — Vision Findings → RAG → Final"
APP_SUBTITLE = "GPT-4o Vision (text findings) + optional FAISS RAG + final structured report (educational)"

DEFAULT_VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")       # IMPORTANT: strong vision
DEFAULT_FINAL_MODEL = os.getenv("FINAL_MODEL", "gpt-4o-mini")    # can be gpt-4o for best accuracy
RAG_DIR = os.getenv("RAG_DIR", "data")                           # expects data/index.faiss + data/meta.json
MAX_RAG_HITS = int(os.getenv("MAX_RAG_HITS", "6"))

st.set_page_config(page_title=APP_TITLE, page_icon="👁️", layout="wide")

# -----------------------------
# PROMPTS
# -----------------------------
VISION_FINDINGS_SYSTEM = """
You are a retina subspecialist analyzing DE-IDENTIFIED ophthalmic images (retina/OCT/FA/FAF/OCTA).
These are NOT faces/people. Do NOT identify any person.
Task: produce a high-fidelity morphology description as TEXT (not JSON).
Be extremely careful distinguishing hemorrhage vs pigment vs atrophy vs artifact.

Output format (plain text):
A) Modalities detected (Fundus/OCT/FAF/FA/ICGA/OCTA)
B) Image quality (GOOD/FAIR/POOR) + why
C) Key findings (8–15 bullets): location, distribution (sectoral vs diffuse), hemorrhage/exudate, vessel changes, macular changes, disc changes
D) Pattern impression (NOT a final diagnosis): 2–4 patterns (e.g., "venous occlusion pattern", "pigmentary dystrophy pattern") with brief rationale
E) 3 "must-not-miss" red flags (if any)
F) One-line RAG query suggestion: write a compact query string based on your findings (no diagnoses unless truly obvious)
"""

FINAL_SYSTEM = """
You are RetinaGPT (educational only). You will receive:
1) Clinical metadata text
2) Vision Findings TEXT (primary observation)
3) Optional REFERENCE CARDS (RAG)

Rules:
- Prioritize vision findings; RAG is secondary (refine discriminators/pitfalls/work-up at high level).
- No patient-specific treatment instructions, dosing, or urgent step-by-step plans.
- Output MUST be TWO blocks:
(A) STRICT JSON inside a Markdown fence labeled json
(B) Human-readable report with headings:
1) Clinical Triage
2) Detected Modalities + Missing Modalities
3) Detected Pattern(s) (with confidence)
4) Imaging Quality
5) Findings by Modality
6) Integrated Pattern Discussion
7) Image Feature Checklist
8) Most Likely Diagnosis (educational impression)
9) Differential Diagnosis (ranked, weighted)
10) Confidence Level
11) Additional imaging/tests to clarify
12) Emergency triage label
13) Educational limitations statement
"""

FINAL_JSON_SCHEMA_HINT = """
(A) JSON schema (STRICT JSON, no trailing commas):
{
  "case_summary": {
    "age": "string or null",
    "sex": "string or null",
    "symptoms": "string or null",
    "duration": "string or null",
    "laterality": "string or null",
    "history": "string or null"
  },
  "modalities_detected": ["Fundus","OCT","FAF","FA","ICGA","OCTA"],
  "missing_modalities_suggested": ["OCT","FAF","FA","ICGA","OCTA"],
  "patterns": [{"name":"string","confidence":0.0}],
  "feature_checklist": {
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
  },
  "most_likely": {"diagnosis":"string","probability":0.0},
  "differential": [{"diagnosis":"string","probability":0.0,"for":["..."],"against":["..."]}],
  "confidence_level":"LOW|MODERATE|HIGH",
  "next_best_requests":[{"request":"string","why":"string"}],
  "urgency":"CRITICAL|URGENT|ROUTINE"
}
"""

# -----------------------------
# HELPERS
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


def parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    if "```json" not in text:
        return None
    try:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        block = text[start:end].strip()
        return json.loads(block)
    except Exception:
        return None


def normalize_probs(items: List[Dict[str, Any]], key: str = "probability") -> List[Dict[str, Any]]:
    total = 0.0
    for it in items:
        try:
            total += float(it.get(key, 0.0))
        except Exception:
            pass
    if total <= 0:
        return items
    for it in items:
        try:
            it[key] = float(it.get(key, 0.0)) / total
        except Exception:
            pass
    return items


def build_image_content(files) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for f in files or []:
        mime = f.type or "image/jpeg"
        url = b64_data_url(f.getvalue(), mime)
        content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
    return content


def call_chat(client: OpenAI, model: str, messages: List[Dict[str, Any]], max_tokens: int, temperature: float) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


# -----------------------------
# RAG (FAISS)
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
            txt = chunk.get("text", "")
            hits.append((title, txt, float(D[0][j])))

        status["hits"] = len(hits)
        if not hits:
            return ("", status)

        lines = ["REFERENCE CARDS (RAG):"]
        for n, (title, txt, _) in enumerate(hits, start=1):
            if len(txt) > 1800:
                txt = txt[:1800] + "..."
            lines.append(f"\n--- CARD {n}: {title} ---\n{txt}\n")
        return ("\n".join(lines), status)

    except Exception as e:
        status["error"] = str(e)
        return ("", status)


# -----------------------------
# UI
# -----------------------------
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

api_key = safe_api_key()
if not api_key:
    st.error("OPENAI_API_KEY not found. Add to Streamlit secrets or env var.")
    st.stop()

client = OpenAI(api_key=api_key)

with st.sidebar:
    st.header("Settings")
    vision_model = st.selectbox("Vision model (must be strong)", [DEFAULT_VISION_MODEL, "gpt-4o"], index=0)
    final_model = st.selectbox("Final model", [DEFAULT_FINAL_MODEL, "gpt-4o-mini", "gpt-4o"], index=0)
    use_rag = st.checkbox("Enable RAG (FAISS)", value=True)
    rag_k = st.slider("RAG hits (k)", 2, 10, min(MAX_RAG_HITS, 10), 1)
    st.caption("Note: No CLAHE here. We want ChatGPT-like raw-image reading.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Clinical metadata")
    clinical_text = st.text_area(
        "Age / Sex / Symptoms / Duration / Laterality / History",
        height=140,
        placeholder="Example: 30M, congenital nystagmus, low vision since childhood, bilateral."
    )

    st.subheader("Upload imaging")
    files = st.file_uploader(
        "Fundus / OCT / FAF / FA / OCTA (jpg/png/webp)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    run = st.button("🔎 Analyze", type="primary")

with col2:
    st.subheader("Outputs")
    vision_box = st.empty()
    rag_box = st.empty()
    final_box = st.empty()

    with st.expander("Debug panels", expanded=True):
        dbg1 = st.empty()
        dbg2 = st.empty()

# -----------------------------
# RUN
# -----------------------------
if run:
    if not files:
        st.warning("Upload at least 1 image.")
        st.stop()

    image_content = build_image_content(files)
    dbg1.info(f"vision_model={vision_model} | final_model={final_model} | images={len(image_content)} | rag={use_rag}")

    # 1) GPT-4o Vision Analysis -> Findings TEXT
    with st.spinner("Step 1/3: GPT-4o vision analysis → Findings (text)..."):
        user_text = "CLINICAL METADATA:\n" + (clinical_text.strip() if clinical_text.strip() else "(not provided)")
        msgs1 = [
            {"role": "system", "content": VISION_FINDINGS_SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": user_text}] + image_content},
        ]
        findings_text = call_chat(client, vision_model, msgs1, max_tokens=900, temperature=0.0).strip()

    vision_box.success("✅ Vision Findings (text) generated")
    vision_box.text(findings_text[:8000])

    # Extract a RAG query hint line if present (optional)
    rag_query = ""
    for line in findings_text.splitlines():
        if line.strip().lower().startswith("f)"):
            rag_query = line.split(":", 1)[-1].strip()
            break

    # 2) RAG retrieval (optional)
    rag_context = ""
    rag_status = {"enabled": False, "hits": 0, "dim": None, "error": None}
    if use_rag:
        with st.spinner("Step 2/3: RAG retrieval (FAISS)..."):
            query = rag_query if rag_query else (findings_text + "\n" + clinical_text).strip()
            query = query[:8000] + "\nretina imaging differential diagnosis"
            rag_context, rag_status = rag_retrieve(client, query, k=rag_k)

    if use_rag:
        if rag_status.get("enabled"):
            rag_box.success(f"✅ RAG retrieved | hits={rag_status.get('hits')} | dim={rag_status.get('dim')}")
        else:
            rag_box.warning("⚠️ RAG disabled/not available. Check data/index.faiss + data/meta.json + faiss-cpu.")
        if rag_status.get("error"):
            rag_box.error(f"RAG error: {rag_status['error']}")
        with st.expander("Retrieved cards (debug)", expanded=False):
            st.text(rag_context if rag_context else "(No RAG context)")

    # 3) Final reasoning (RetinaGPT) using Findings TEXT + RAG
    with st.spinner("Step 3/3: Final report (JSON + Human report)..."):
        final_user = (
            "CLINICAL METADATA:\n"
            + (clinical_text.strip() if clinical_text.strip() else "(not provided)")
            + "\n\nVISION FINDINGS (TEXT, PRIMARY):\n"
            + findings_text
            + "\n\n"
            + FINAL_JSON_SCHEMA_HINT
            + "\n\nNow produce the two-block output."
        )

        msgs3: List[Dict[str, Any]] = [{"role": "system", "content": FINAL_SYSTEM}]
        if rag_context:
            msgs3.append({"role": "system", "content": rag_context})
        msgs3.append({"role": "user", "content": final_user})

        final_text = call_chat(client, final_model, msgs3, max_tokens=1900, temperature=0.2)

    final_box.subheader("Final output")
    final_box.markdown(final_text)

    parsed = parse_json_block(final_text)
    if parsed:
        if isinstance(parsed.get("differential"), list):
            parsed["differential"] = normalize_probs(parsed["differential"], "probability")
        dbg2.success("Final JSON parsed ✅")
        dbg2.json(parsed)
    else:
        dbg2.warning("Final JSON fenced block parse edilemedi (model format drift).")
