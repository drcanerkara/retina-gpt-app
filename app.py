# app.py — RetinaGPT v4 (Single-file, Vision-first + Optional RAG + Dual-model consensus)
# -------------------------------------------------------------------------------
# GOAL
# 1) Use OpenAI Vision to extract structured findings (JSON) from uploaded retinal images.
# 2) Use Gemini Vision to extract structured findings (JSON) from the SAME images.
# 3) Build a consensus (agreement / disagreement + what to clarify).
# 4) (Optional) Use local FAISS RAG cards to refine discriminators/pitfalls/work-up.
# 5) Generate a final human-readable educational report (NO individualized treatment).
#
# REQUIREMENTS (example)
#   streamlit
#   openai>=1.40.0
#   google-genai>=0.6.0
#   faiss-cpu
#   numpy
#
# SECRETS
#   OPENAI_API_KEY  (Streamlit secrets or env)
#   GEMINI_API_KEY  (Streamlit secrets or env)
#
# RAG FILES (optional)
#   data/index.faiss
#   data/meta.json   (list of {"title": "...", "text": "..."} chunks)
# -------------------------------------------------------------------------------

import os
import json
import base64
import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI

# Gemini (google-genai)
try:
    from google import genai
except Exception:
    genai = None

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
APP_TITLE = "RetinaGPT v4 — Vision-first + Dual Consensus + RAG (optional)"
DEFAULT_OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
DEFAULT_OPENAI_REASON_MODEL = os.getenv("OPENAI_REASON_MODEL", "gpt-4o-mini")

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

RAG_DIR = os.getenv("RAG_DIR", "data")
MAX_RAG_HITS = int(os.getenv("MAX_RAG_HITS", "6"))

st.set_page_config(page_title=APP_TITLE, page_icon="👁️", layout="wide")


# -----------------------------
# SAFETY SYSTEM PROMPT (final report)
# -----------------------------
FINAL_SYSTEM_PROMPT = """
You are RetinaGPT v4, a retina subspecialty educational imaging discussion system.

IMPORTANT CONTEXT
- The user uploads DE-IDENTIFIED ophthalmic images (fundus/OCT/FAF/FA/OCTA). Not faces/people.
- Your job is to describe morphology and provide an EDUCATIONAL differential diagnosis.
- Do NOT provide patient-specific treatment instructions, dosing, or step-by-step urgent management.
- If findings could be vision-threatening, recommend prompt clinical evaluation in general terms.

STYLE
- Formal medical English, objective, concise.
- Describe morphology first, then offer a ranked differential with discriminators.
- Use retina terminology (EZ, RPE, IRF/SRF, ischemia, leakage, nonperfusion, etc).
"""


# -----------------------------
# JSON SCHEMAS (for consistent extraction)
# -----------------------------
# Minimal but useful. You can extend later.
EXTRACT_SCHEMA_DESCRIPTION = """
Return STRICT JSON with this schema:

{
  "modalities_detected": ["Fundus","OCT","FAF","FA","ICGA","OCTA"],
  "image_quality": "GOOD|FAIR|POOR",
  "laterality_guess": "RIGHT|LEFT|BILATERAL|UNCERTAIN",
  "key_findings": [
    "short bullet finding",
    "short bullet finding"
  ],
  "imaging_signals": {
    "hemorrhage": "PRESENT|ABSENT|UNCERTAIN",
    "exudates": "PRESENT|ABSENT|UNCERTAIN",
    "cotton_wool_spots": "PRESENT|ABSENT|UNCERTAIN",
    "venous_tortuosity_dilation": "PRESENT|ABSENT|UNCERTAIN",
    "arterial_attenuation": "PRESENT|ABSENT|UNCERTAIN",
    "neovascularization": "PRESENT|ABSENT|UNCERTAIN",
    "capillary_nonperfusion": "PRESENT|ABSENT|UNCERTAIN",
    "macular_edema_or_thickening": "PRESENT|ABSENT|UNCERTAIN",
    "blocked_fluorescence": "PRESENT|ABSENT|UNCERTAIN",
    "late_leakage": "PRESENT|ABSENT|UNCERTAIN"
  },
  "top_diagnoses": [
    {"name": "string", "prob": 0.0, "for": ["..."], "against": ["..."]},
    {"name": "string", "prob": 0.0, "for": ["..."], "against": ["..."]},
    {"name": "string", "prob": 0.0, "for": ["..."], "against": ["..."]}
  ],
  "questions_to_clarify": [
    "one short question"
  ]
}

Rules:
- STRICT JSON only. No markdown, no commentary.
- probs do not need to sum to 1, but keep them reasonable (0-1).
- If unsure, use "UNCERTAIN" and include clarifying questions.
"""


# -----------------------------
# HELPERS
# -----------------------------
def safe_get_secret(name: str) -> Optional[str]:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name)


def b64_data_url(file_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(file_bytes).decode('utf-8')}"


def robust_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """
    Try hard to parse JSON even if the model returns extra text.
    We first attempt direct json.loads; if fails, we extract the first {...} block.
    """
    if not text:
        return None
    t = text.strip()
    # direct
    try:
        return json.loads(t)
    except Exception:
        pass

    # try extracting first JSON object
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def normalize_probs(top_dxs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total = 0.0
    for d in top_dxs:
        try:
            total += float(d.get("prob", 0.0))
        except Exception:
            pass
    if total <= 0:
        return top_dxs
    for d in top_dxs:
        try:
            d["prob"] = float(d.get("prob", 0.0)) / total
        except Exception:
            pass
    return top_dxs


def top_dx_names(js: Dict[str, Any], n: int = 3) -> List[str]:
    out: List[str] = []
    for d in (js.get("top_diagnoses") or [])[:n]:
        name = (d or {}).get("name")
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


def overlap_score(list_a: List[str], list_b: List[str]) -> float:
    a = {x.lower() for x in list_a}
    b = {x.lower() for x in list_b}
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / max(union, 1)


# -----------------------------
# OPTIONAL RAG (FAISS)
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


def openai_embed(client: OpenAI, text: str) -> Optional[List[float]]:
    try:
        emb = client.embeddings.create(model="text-embedding-3-small", input=text)
        return emb.data[0].embedding
    except Exception:
        return None


def rag_retrieve(client: OpenAI, query: str, k: int = MAX_RAG_HITS) -> Tuple[str, Dict[str, Any]]:
    enabled, idx, meta = load_rag_index(RAG_DIR)
    status: Dict[str, Any] = {"enabled": False, "hits": 0, "dim": None, "error": None}
    if not enabled or idx is None or meta is None:
        return ("", status)

    status["enabled"] = True
    status["dim"] = int(getattr(idx, "d", 0))

    try:
        vec = openai_embed(client, query)
        if vec is None:
            status["error"] = "Embedding failed"
            return ("", status)

        x = np.array([vec], dtype="float32")
        D, I = idx.search(x, k)

        hits: List[Tuple[str, str]] = []
        for row_id in I[0].tolist():
            if row_id < 0 or row_id >= len(meta):
                continue
            chunk = meta[row_id]
            title = str(chunk.get("title", f"chunk_{row_id}"))
            text = str(chunk.get("text", ""))
            if text.strip():
                hits.append((title, text))

        status["hits"] = len(hits)
        if not hits:
            return ("", status)

        ctx = ["REFERENCE CARDS (RAG):"]
        for i, (title, text) in enumerate(hits, start=1):
            ctx.append(f"\n--- CARD {i}: {title} ---\n{text}\n")
        return ("\n".join(ctx), status)

    except Exception as e:
        status["error"] = str(e)
        return ("", status)


# -----------------------------
# OPENAI VISION JSON EXTRACTION
# -----------------------------
def call_openai_vision_json(
    client: OpenAI,
    model: str,
    clinical_text: str,
    images: List[Dict[str, Any]],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    system = (
        "You are a retina imaging feature extractor for DE-IDENTIFIED ophthalmic images. "
        "Describe morphology and generate a differential. Output STRICT JSON only."
    )
    user_text = (
        "DE-IDENTIFIED RETINAL IMAGES (not faces/people).\n\n"
        f"CLINICAL INFO:\n{clinical_text.strip() if clinical_text.strip() else '(not provided)'}\n\n"
        f"{EXTRACT_SCHEMA_DESCRIPTION}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": user_text}] + images},
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=900,
        )
        raw = (resp.choices[0].message.content or "").strip()
        js = robust_json_parse(raw)
        if js and isinstance(js.get("top_diagnoses"), list):
            js["top_diagnoses"] = normalize_probs(js["top_diagnoses"])
        return raw, js
    except Exception as e:
        raw = f'{{"error":"openai_call_failed","details":"{str(e)}"}}'
        return raw, None


# -----------------------------
# GEMINI VISION JSON EXTRACTION
# -----------------------------
def call_gemini_vision_json(
    gemini_client: Any,
    model: str,
    clinical_text: str,
    uploaded_files: List[Any],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Uses google-genai SDK. We keep it simple and robust:
    - contents is a list: [prompt_string, {mime_type,data}, {mime_type,data}, ...]
    - Gemini returns text; we parse JSON from it.
    """
    prompt = (
        "You are a retina imaging feature extractor for DE-IDENTIFIED ophthalmic images.\n"
        "Describe morphology and generate a differential. Output STRICT JSON only.\n\n"
        "DE-IDENTIFIED RETINAL IMAGES (not faces/people).\n\n"
        f"CLINICAL INFO:\n{clinical_text.strip() if clinical_text.strip() else '(not provided)'}\n\n"
        f"{EXTRACT_SCHEMA_DESCRIPTION}"
    )

    contents: List[Any] = [prompt]

    for f in uploaded_files:
        # Streamlit UploadedFile supports .type and .getvalue()
        contents.append({"mime_type": f.type or "image/jpeg", "data": f.getvalue()})

    try:
        resp = gemini_client.models.generate_content(model=model, contents=contents)
        raw = (getattr(resp, "text", "") or "").strip()
        js = robust_json_parse(raw)
        if js and isinstance(js.get("top_diagnoses"), list):
            js["top_diagnoses"] = normalize_probs(js["top_diagnoses"])
        return raw, js
    except Exception as e:
        raw = f'{{"error":"gemini_call_failed","details":"{str(e)}"}}'
        return raw, None


# -----------------------------
# CONSENSUS LOGIC
# -----------------------------
def build_consensus(
    openai_js: Optional[Dict[str, Any]],
    gemini_js: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Returns a consensus object:
    - agreement_level
    - shared_diagnoses
    - disagreements
    - what_to_clarify
    """
    cons: Dict[str, Any] = {
        "agreement_level": "UNKNOWN",
        "shared_diagnoses": [],
        "openai_top": [],
        "gemini_top": [],
        "what_to_clarify": [],
        "notes": [],
    }

    if not openai_js and not gemini_js:
        cons["agreement_level"] = "NONE"
        cons["notes"].append("Both model extractions failed.")
        return cons

    openai_top = top_dx_names(openai_js or {}, 3)
    gemini_top = top_dx_names(gemini_js or {}, 3)
    cons["openai_top"] = openai_top
    cons["gemini_top"] = gemini_top

    if not openai_top or not gemini_top:
        cons["agreement_level"] = "LOW"
    else:
        score = overlap_score(openai_top, gemini_top)
        if score >= 0.50:
            cons["agreement_level"] = "HIGH"
        elif score >= 0.20:
            cons["agreement_level"] = "MODERATE"
        else:
            cons["agreement_level"] = "LOW"

        shared = list({x for x in openai_top for y in gemini_top if x.lower() == y.lower()})
        cons["shared_diagnoses"] = shared

    # Gather clarifying questions from both
    q: List[str] = []
    for js in [openai_js or {}, gemini_js or {}]:
        for item in (js.get("questions_to_clarify") or [])[:6]:
            if isinstance(item, str) and item.strip():
                q.append(item.strip())
    # de-dup while preserving order
    seen = set()
    q2 = []
    for item in q:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        q2.append(item)
    cons["what_to_clarify"] = q2[:8]

    # Add notes about hard disagreement
    if cons["agreement_level"] == "LOW" and openai_top and gemini_top:
        cons["notes"].append("Top differentials diverge; recommend additional modality/clinical clarification.")

    return cons


# -----------------------------
# FINAL REPORT GENERATION (OpenAI)
# -----------------------------
def build_final_report(
    client: OpenAI,
    model: str,
    clinical_text: str,
    openai_js: Optional[Dict[str, Any]],
    gemini_js: Optional[Dict[str, Any]],
    consensus: Dict[str, Any],
    rag_context: str,
) -> str:
    """
    Use the structured outputs (and optionally RAG cards) to produce a clinician-friendly report.
    This is where you want your RetinaGPT style to shine.
    """
    payload = {
        "clinical_text": clinical_text.strip() if clinical_text.strip() else None,
        "openai_vision_json": openai_js,
        "gemini_vision_json": gemini_js,
        "consensus": consensus,
    }

    user_prompt = """
Generate an EDUCATIONAL retina imaging report based on:
- Clinical text (if any)
- OpenAI vision JSON
- Gemini vision JSON
- Consensus object
- Optional REFERENCE CARDS (RAG)

Rules:
- Start with morphology (what is seen).
- Then give a ranked differential with brief discriminators.
- If the two model opinions disagree, explicitly say what differs and how to reconcile (what would clarify).
- Avoid patient-specific treatment instructions or dosing.
- If potentially vision-threatening, recommend prompt evaluation in general terms.
- Keep it clinically useful and concise.

Use this structure:
1) Clinical Triage (educational)
2) Modalities detected + missing modalities to consider
3) Key findings (integrated)
4) Most likely pattern/diagnosis (educational impression)
5) Differential diagnosis (ranked)
6) Disagreement analysis (if any)
7) What would clarify (extra imaging/tests/clinical details)
8) Educational limitations statement
"""

    messages = [{"role": "system", "content": FINAL_SYSTEM_PROMPT}]
    if rag_context:
        messages.append({"role": "system", "content": rag_context})

    messages.append(
        {
            "role": "user",
            "content": user_prompt + "\n\nDATA (JSON):\n" + json.dumps(payload, ensure_ascii=False, indent=2),
        }
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1400,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Final report generation failed: {e}"


# -----------------------------
# UI
# -----------------------------
st.title(APP_TITLE)
st.caption("Upload images + clinical info → dual vision extraction (OpenAI + Gemini) → consensus → optional RAG → final educational report.")

openai_key = safe_get_secret("OPENAI_API_KEY")
gemini_key = safe_get_secret("GEMINI_API_KEY")

if not openai_key:
    st.error("OPENAI_API_KEY not found. Add it to Streamlit secrets or environment variables.")
    st.stop()

if genai is None:
    st.warning("google-genai is not installed/importable. Gemini calls will be disabled until you add google-genai to requirements.")
else:
    if not gemini_key:
        st.warning("GEMINI_API_KEY not found. Gemini calls will be disabled until you add it to secrets/env.")

# Clients
openai_client = OpenAI(api_key=openai_key)
gemini_client = None
if genai is not None and gemini_key:
    try:
        gemini_client = genai.Client(api_key=gemini_key)
    except Exception:
        gemini_client = None

with st.sidebar:
    st.header("Settings")

    st.subheader("OpenAI")
    openai_vision_model = st.text_input("OpenAI vision model", value=DEFAULT_OPENAI_VISION_MODEL)
    openai_reason_model = st.text_input("OpenAI reasoning model (final report)", value=DEFAULT_OPENAI_REASON_MODEL)

    st.divider()
    st.subheader("Gemini")
    gemini_model = st.text_input("Gemini vision model", value=DEFAULT_GEMINI_MODEL)
    use_gemini = st.checkbox("Enable Gemini (if key available)", value=True)

    st.divider()
    st.subheader("RAG (FAISS)")
    use_rag = st.checkbox("Enable RAG (local FAISS)", value=True)
    rag_k = st.slider("RAG hits (k)", min_value=2, max_value=10, value=MAX_RAG_HITS, step=1)

    st.divider()
    show_debug = st.checkbox("Show debug panels", value=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Clinical metadata")
    clinical_text = st.text_area(
        "Age, symptoms, duration, laterality, history (optional but helpful)",
        height=150,
        placeholder="Example:\nAge: 58\nSex: Male\nSymptoms: sudden vision loss\nDuration: 2 days\nLaterality: unilateral (OD)\nHistory: HTN, DM, glaucoma, etc."
    )

    st.subheader("Upload retinal imaging")
    uploads = st.file_uploader(
        "Fundus / OCT / FAF / FA / OCTA (jpg/png/webp)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if uploads:
        st.caption(f"{len(uploads)} file(s) uploaded.")
        with st.expander("Preview images", expanded=False):
            for f in uploads[:6]:
                st.image(f, caption=f.name, use_container_width=True)

with col2:
    st.subheader("Run")
    run = st.button("🔎 Analyze", type="primary")

    st.divider()
    st.subheader("Outputs")
    st.caption("The app extracts findings from OpenAI and Gemini, builds consensus, optionally retrieves RAG cards, then produces a final report.")


# -----------------------------
# RUN PIPELINE
# -----------------------------
def build_openai_image_payload(uploaded_files) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if not uploaded_files:
        return content
    for f in uploaded_files:
        mime = f.type or "image/jpeg"
        data_url = b64_data_url(f.getvalue(), mime)
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    return content


if run:
    if not uploads:
        st.error("Please upload at least 1 retinal image.")
        st.stop()

    # 1) OpenAI Vision extraction (JSON)
    st.info("Step 1/4 — OpenAI Vision extraction…")
    openai_imgs = build_openai_image_payload(uploads)
    openai_raw, openai_js = call_openai_vision_json(
        client=openai_client,
        model=openai_vision_model.strip(),
        clinical_text=clinical_text,
        images=openai_imgs,
    )

    # 2) Gemini Vision extraction (JSON)
    gemini_raw, gemini_js = "", None
    if use_gemini and gemini_client is not None:
        st.info("Step 2/4 — Gemini Vision extraction…")
        gemini_raw, gemini_js = call_gemini_vision_json(
            gemini_client=gemini_client,
            model=gemini_model.strip(),
            clinical_text=clinical_text,
            uploaded_files=uploads,
        )
    else:
        st.warning("Gemini is disabled or not available (missing google-genai or GEMINI_API_KEY).")

    # 3) Consensus
    st.info("Step 3/4 — Building consensus…")
    consensus = build_consensus(openai_js, gemini_js)

    # 4) Optional RAG retrieve (based on integrated cues)
    rag_context = ""
    rag_status = {"enabled": False, "hits": 0, "dim": None, "error": None}
    if use_rag:
        # Build a retrieval query from BOTH model findings + dx candidates
        query_parts: List[str] = []
        if clinical_text.strip():
            query_parts.append("CLINICAL:\n" + clinical_text.strip())

        if openai_js:
            query_parts.append("OPENAI_KEY_FINDINGS:\n" + "\n".join(openai_js.get("key_findings") or []))
            query_parts.append("OPENAI_TOP_DX:\n" + ", ".join(top_dx_names(openai_js, 3)))

        if gemini_js:
            query_parts.append("GEMINI_KEY_FINDINGS:\n" + "\n".join(gemini_js.get("key_findings") or []))
            query_parts.append("GEMINI_TOP_DX:\n" + ", ".join(top_dx_names(gemini_js, 3)))

        query_parts.append("retina imaging differential discriminators pitfalls work-up")
        retrieval_query = "\n\n".join([p for p in query_parts if p.strip()])

        rag_context, rag_status = rag_retrieve(openai_client, retrieval_query, k=rag_k)
        if rag_status.get("enabled"):
            st.success(f"RAG: enabled | hits={rag_status.get('hits')} | dim={rag_status.get('dim')}")
        else:
            st.warning("RAG not available (check FAISS install + data/index.faiss + data/meta.json).")
        if rag_status.get("error"):
            st.error(f"RAG error: {rag_status['error']}")

    # 5) Final report (OpenAI)
    st.info("Step 4/4 — Generating final report…")
    final_report = build_final_report(
        client=openai_client,
        model=openai_reason_model.strip(),
        clinical_text=clinical_text,
        openai_js=openai_js,
        gemini_js=gemini_js,
        consensus=consensus,
        rag_context=rag_context,
    )

    # -----------------------------
    # DISPLAY
    # -----------------------------
    st.subheader("Final educational report")
    st.markdown(final_report)

    if show_debug:
        st.divider()
        st.subheader("Debug panels")

        cA, cB = st.columns(2)

        with cA:
            st.markdown("### OpenAI Vision (raw)")
            st.code(openai_raw[:4000] if openai_raw else "(empty)", language="json")
            st.markdown("### OpenAI Vision (parsed)")
            st.json(openai_js or {"error": "openai_json_parse_failed"})

        with cB:
            st.markdown("### Gemini Vision (raw)")
            st.code(gemini_raw[:4000] if gemini_raw else "(empty or disabled)", language="json")
            st.markdown("### Gemini Vision (parsed)")
            st.json(gemini_js or {"error": "gemini_json_parse_failed_or_disabled"})

        st.markdown("### Consensus")
        st.json(consensus)

        st.markdown("### RAG context (first 2000 chars)")
        st.code((rag_context[:2000] if rag_context else "(no rag context)"), language="markdown")
