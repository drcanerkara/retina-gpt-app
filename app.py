import os
import json
import base64
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from openai import OpenAI

# Optional RAG deps (only if you use local FAISS index)
try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    faiss = None
    np = None

# -----------------------------
# CONFIG
# -----------------------------
APP_TITLE = "RetinaGPT v2"
APP_SUBTITLE = "Academic + Decision Support (Educational only)"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RAG_DIR = os.getenv("RAG_DIR", "data")  # expects data/index.faiss and data/meta.json
MAX_RAG_HITS = int(os.getenv("MAX_RAG_HITS", "6"))

st.set_page_config(page_title=APP_TITLE, page_icon="👁️", layout="wide")

# -----------------------------
# SYSTEM PROMPT (v2)
# -----------------------------
SYSTEM_PROMPT = """
You are RetinaGPT v2, a retina subspecialty educational discussion and decision-support system.

PURPOSE
Provide structured academic discussion of retinal imaging findings and differential diagnostic reasoning
for educational purposes only. Not medical advice.

STYLE
- Formal medical English, objective, concise.
- Use retina subspecialty terminology (e.g., EZ/ellipsoid zone, RPE, SHRM, PED, hypertransmission, etc.).
- Avoid over-commitment for rare/atypical patterns; describe morphology first, then offer a ranked differential.

REFERENCE KNOWLEDGE (RAG)
If “REFERENCE CARDS” are provided, you MUST treat them as the primary factual source.
- Use them to refine discriminators, pitfalls, work-up, and management.
- If imaging suggests a different pattern than retrieved cards, explicitly state discrepancy and explain why.
- Do not invent facts not supported by imaging/metadata/reference cards.

GLOBAL SAFETY
- Educational purposes only. No patient-specific medical advice or treatment instructions.
- For emergency patterns, recommend urgent evaluation but do not prescribe.

OUTPUT REQUIREMENTS (IMPORTANT)
You MUST respond in TWO blocks:
(A) A JSON block inside a Markdown code fence labeled json (STRICT JSON, no trailing commas)
(B) A human-readable report in the specified format.

(A) JSON schema:
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
  "missing_modalities_suggested": ["OCT", "FAF", ...],
  "patterns": [
    {"name": "string", "confidence": 0.0},
    {"name": "string", "confidence": 0.0}
  ],
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
  "most_likely": {"diagnosis": "string", "probability": 0.0},
  "differential": [
    {"diagnosis": "string", "probability": 0.0, "for": ["..."], "against": ["..."]},
    {"diagnosis": "string", "probability": 0.0, "for": ["..."], "against": ["..."]}
  ],
  "confidence_level": "LOW|MODERATE|HIGH",
  "next_best_requests": [
    {"request": "string", "why": "string"}
  ],
  "urgency": "CRITICAL|URGENT|ROUTINE"
}

(B) Human report format:
1) Clinical Triage
2) Detected Modalities + Missing Modalities
3) Detected Pattern(s) (with confidence)
4) Imaging Quality
5) Findings by Modality (Fundus / OCT / FAF / Angiography-OCTA)
6) Integrated Pattern Discussion
7) Image Feature Checklist
8) Most Likely Diagnosis
9) Differential Diagnosis (ranked, weighted)
10) Confidence Level
11) Additional imaging/tests to clarify (if needed)
12) Emergency triage label
13) Educational limitations statement

GUARDRAILS
- Do NOT label “mass/tumor/elevated lesion” unless OCT shows clear dome-shaped thickening or solid structure.
- If lesion is flat with outer retinal/RPE alteration without solid mass, describe accordingly.
- If torpedo-shaped hypopigmented lesion temporal to fovea + OCT outer retinal/RPE changes, include torpedo maculopathy in top ddx.
"""

# -----------------------------
# HELPERS
# -----------------------------
def b64_image(file_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(file_bytes).decode('utf-8')}"


def safe_get_secrets_api_key() -> Optional[str]:
    # Streamlit Cloud: st.secrets["OPENAI_API_KEY"]
    # Local: env var OPENAI_API_KEY
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Expects assistant to include:
    ```json
    {...}
    ```
    Returns dict or None.
    """
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


# -----------------------------
# RAG (FAISS) - optional
# Expects:
# data/index.faiss
# data/meta.json  -> list of chunks: [{"id":..., "title":..., "text":...}, ...]
# -----------------------------
@st.cache_resource
def load_rag_index(rag_dir: str) -> Tuple[bool, Optional[Any], Optional[List[Dict[str, Any]]]]:
    index_path = os.path.join(rag_dir, "index.faiss")
    meta_path = os.path.join(rag_dir, "meta.json")

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
            # dist kept for debugging; not shown in final to keep context clean
            context_lines.append(f"\n--- CARD {n}: {title} ---\n{text}\n")

        return ("\n".join(context_lines), status)

    except Exception as e:
        status["error"] = str(e)
        return ("", status)


# -----------------------------
# CASE MEMORY
# -----------------------------
def init_session():
    if "case_id" not in st.session_state:
        st.session_state.case_id = 1
    if "case_notes" not in st.session_state:
        st.session_state.case_notes = ""  # running summary
    if "case_history" not in st.session_state:
        st.session_state.case_history = []  # chat turns (role/content)
    if "last_json" not in st.session_state:
        st.session_state.last_json = None  # last structured output
    if "last_missing_requests" not in st.session_state:
        st.session_state.last_missing_requests = []


def update_case_memory(new_json: Dict[str, Any]):
    st.session_state.last_json = new_json

    ml = (new_json.get("most_likely") or {}).get("diagnosis")
    conf = new_json.get("confidence_level")
    urgency = new_json.get("urgency")
    patterns = new_json.get("patterns") or []
    pat_txt = ", ".join(
        [f"{p.get('name')} ({p.get('confidence')})" for p in patterns[:2] if p.get("name")]
    )

    mem = f"Most likely: {ml}; Confidence: {conf}; Urgency: {urgency}; Patterns: {pat_txt}"
    st.session_state.case_notes = mem
    st.session_state.last_missing_requests = new_json.get("next_best_requests", []) or []


# -----------------------------
# UI
# -----------------------------
init_session()

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

api_key = safe_get_secrets_api_key()
if not api_key:
    st.error("OPENAI_API_KEY bulunamadı. Streamlit Secrets veya environment variable olarak ekle.")
    st.stop()

client = OpenAI(api_key=api_key)

with st.sidebar:
    st.header("Settings")

    model_options = list(dict.fromkeys([DEFAULT_MODEL, "gpt-4o"]))
    model = st.selectbox("Model", options=model_options, index=0)

    use_rag = st.checkbox("Enable RAG (FAISS)", value=True)
    rag_k = st.slider("RAG hits (k)", min_value=2, max_value=10, value=MAX_RAG_HITS, step=1)

    st.divider()
    if st.button("🧹 New case (reset memory)"):
        st.session_state.case_id += 1
        st.session_state.case_notes = ""
        st.session_state.case_history = []
        st.session_state.last_json = None
        st.session_state.last_missing_requests = []
        st.rerun()

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

    st.subheader("Case memory (v2)")
    st.write(f"**Case #{st.session_state.case_id}**")
    if st.session_state.case_notes:
        st.info(st.session_state.case_notes)
    else:
        st.caption("No memory yet. Run analysis to build case memory.")

with col2:
    st.subheader("RAG status")
    rag_context = ""
    rag_status: Dict[str, Any] = {"enabled": False, "hits": 0, "index_dim": None, "error": None}

    retrieval_query = clinical_text.strip()
    if st.session_state.case_notes:
        retrieval_query += "\n\nPrior summary: " + st.session_state.case_notes

    if use_rag:
        rag_context, rag_status = rag_retrieve(client, retrieval_query, k=rag_k)

    if rag_status.get("enabled"):
        st.success(
            f"RAG enabled: FAISS index + meta.json loaded. "
            f"Hits: {rag_status.get('hits')} | dim: {rag_status.get('index_dim')}"
        )
    else:
        st.warning("RAG disabled or not available (check data/index.faiss + data/meta.json and faiss-cpu).")

    if rag_status.get("error"):
        st.error(f"RAG error: {rag_status['error']}")

    with st.expander("Show retrieved cards (debug)", expanded=False):
        st.text(rag_context if rag_context else "(No RAG context)")

    st.divider()
    st.subheader("Analyze")
    run = st.button("🔎 Analyze / Continue this case", type="primary")


# -----------------------------
# RUN ANALYSIS
# -----------------------------
def build_user_payload_images(uploaded_files) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if not uploaded_files:
        return content
    for f in uploaded_files:
        mime = f.type or "image/jpeg"
        b64 = b64_image(f.getvalue(), mime)
        content.append({"type": "image_url", "image_url": {"url": b64}})
    return content


def call_model(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    rag_context: str,
    clinical_text: str,
    image_content: List[Dict[str, Any]],
    case_memory: str,
    chat_history: List[Dict[str, Any]]
) -> str:
    messages: List[Dict[str, Any]] = []
    messages.append({"role": "system", "content": system_prompt})

    if rag_context:
        messages.append({"role": "system", "content": rag_context})

    if case_memory:
        messages.append({"role": "system", "content": f"CASE MEMORY (same patient, continue analysis): {case_memory}"})

    # Keep last few turns only to avoid token bloat
    for m in chat_history[-6:]:
        messages.append(m)

    user_text = "CLINICAL METADATA:\n" + (clinical_text.strip() if clinical_text.strip() else "(not provided)")
    if st.session_state.last_missing_requests:
        user_text += "\n\nPREVIOUSLY REQUESTED (if still missing):\n"
        for r in st.session_state.last_missing_requests[:5]:
            user_text += f"- {r.get('request')}: {r.get('why')}\n"

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    user_content.extend(image_content)

    messages.append({"role": "user", "content": user_content})

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=1800,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        # Return a readable error string to render in UI
        return (
            "```json\n"
            + json.dumps(
                {
                    "error": "OpenAI API call failed",
                    "details": str(e),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n```\n\n"
            + f"OpenAI API call failed: {e}"
        )


if run:
    img_payload = build_user_payload_images(files)

    if not clinical_text.strip() and not img_payload:
        st.warning("Clinical metadata veya en az 1 görüntü yüklemen daha iyi olur. Yine de çalıştırıyorum.")

    assistant_text = call_model(
        client=client,
        model_name=model,
        system_prompt=SYSTEM_PROMPT,
        rag_context=rag_context,
        clinical_text=clinical_text,
        image_content=img_payload,
        case_memory=st.session_state.case_notes,
        chat_history=st.session_state.case_history,
    )

    # Save turn to history (keep text-only history to reduce future format issues)
    st.session_state.case_history.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"[Clinical]\n{clinical_text[:800]}{'...' if len(clinical_text) > 800 else ''}\n"
                            f"[Images] {len(img_payload)}"
                }
            ],
        }
    )
    st.session_state.case_history.append({"role": "assistant", "content": assistant_text})

    parsed = parse_json_block(assistant_text)
    if not parsed:
        st.warning("JSON block parse edilemedi (model strict JSON üretmemiş olabilir).")

    if parsed:
        if isinstance(parsed.get("differential"), list):
            parsed["differential"] = normalize_probs(parsed["differential"], "probability")

        update_case_memory(parsed)

    # Display
    st.subheader("Assistant output")
    st.markdown(assistant_text)

    if parsed:
        st.subheader("Structured output (debug)")
        st.json(parsed)

        missing = parsed.get("missing_modalities_suggested") or []
        next_req = parsed.get("next_best_requests") or []
        if missing or next_req:
            st.info("If you upload the suggested missing modality images, I can continue the SAME case with updated probabilities.")
            with st.expander("What to upload next (adaptive)", expanded=True):
                if missing:
                    st.write("**Suggested missing modalities:** " + ", ".join(missing))
                if next_req:
                    for r in next_req[:6]:
                        st.write(f"- **{r.get('request')}** — {r.get('why')}")
