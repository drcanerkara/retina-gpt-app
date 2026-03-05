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

# Optional OpenCV for gamma
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


# -----------------------------
# CONFIG
# -----------------------------
APP_TITLE = "RetinaGPT v2 (Vision-first + Gated RAG)"
APP_SUBTITLE = "Step1: morphology (gpt-4o) → Step2: diagnosis + optional RAG support (Educational only)"

# We will use gpt-4o for vision analysis by default
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
FORMAT_MODEL = os.getenv("OPENAI_FORMAT_MODEL", "gpt-4o-mini")

RAG_DIR = os.getenv("RAG_DIR", "data")  # expects data/index.faiss and data/meta.json
MAX_RAG_HITS = int(os.getenv("MAX_RAG_HITS", "6"))

st.set_page_config(page_title=APP_TITLE, page_icon="👁️", layout="wide")


# -----------------------------
# SYSTEM PROMPT (main analysis)
# -----------------------------
SYSTEM_PROMPT = """
You are RetinaGPT v2, a retina subspecialty educational imaging discussion system.

CONTEXT
- The user provides DE-IDENTIFIED ophthalmic images (fundus/OCT/FAF/angiography). Not faces/people.
- Describe morphology first; then provide an EDUCATIONAL ranked differential.
- Do NOT provide patient-specific treatment instructions (no dosing, no individualized plan). Educational only.

STYLE
- Formal medical English.
- Avoid over-commitment; include discriminators and uncertainty.

RAG (if provided)
- Use REFERENCE CARDS as supportive background.
- Do NOT let RAG override what is actually visible in the images.

OUTPUT (MUST)
Return TWO blocks:
(A) STRICT JSON inside ```json ... ``` (valid JSON, no trailing commas, no comments)
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
11) Additional imaging/tests to clarify (if needed)
12) Emergency triage label
13) Educational limitations statement

JSON schema:
{
  "case_summary": {"age": null, "sex": null, "symptoms": null, "duration": null, "laterality": null, "history": null},
  "modalities_detected": [],
  "missing_modalities_suggested": [],
  "patterns": [{"name":"", "confidence":0.0}],
  "feature_checklist": {
    "subretinal_fluid":"PRESENT|ABSENT|UNCERTAIN",
    "intraretinal_fluid":"PRESENT|ABSENT|UNCERTAIN",
    "hemorrhage_exudation":"PRESENT|ABSENT|UNCERTAIN",
    "inner_retinal_ischemia":"PRESENT|ABSENT|UNCERTAIN",
    "ez_disruption":"PRESENT|ABSENT|UNCERTAIN",
    "rpe_atrophy_hypertransmission":"PRESENT|ABSENT|UNCERTAIN",
    "vitelliform_material":"PRESENT|ABSENT|UNCERTAIN",
    "inflammatory_signs":"PRESENT|ABSENT|UNCERTAIN",
    "m_nv_suspected":"PRESENT|ABSENT|UNCERTAIN",
    "true_mass_lesion_suspected":"PRESENT|ABSENT|UNCERTAIN"
  },
  "most_likely": {"diagnosis":"", "probability":0.0},
  "differential": [{"diagnosis":"", "probability":0.0, "for":[], "against":[]}],
  "confidence_level":"LOW|MODERATE|HIGH",
  "next_best_requests":[{"request":"", "why":""}],
  "urgency":"CRITICAL|URGENT|ROUTINE"
}

HARD RULE (ALBINISM TRIGGER):
If fundus shows diffuse hypopigmentation and/or very prominent choroidal vessels
AND clinical history includes nystagmus and/or lifelong low vision,
you MUST include "Albinism spectrum / Ocular albinism" in the TOP differential
and suggest OCT to evaluate for foveal hypoplasia (educational).
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


def b64_image(file_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(file_bytes).decode('utf-8')}"


def parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    if "```json" not in text:
        return None
    try:
        s = text.index("```json") + len("```json")
        e = text.index("```", s)
        return json.loads(text[s:e].strip())
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


def gamma_preprocess(image_bytes: bytes, gamma: float = 1.15) -> bytes:
    """
    Gentle preprocessing (safer than CLAHE). Returns PNG bytes.
    If cv2 is unavailable, returns original bytes.
    """
    if cv2 is None or np is None:
        return image_bytes
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes

    inv = 1.0 / max(gamma, 0.01)
    table = (np.array([(i / 255.0) ** inv * 255 for i in range(256)])).astype("uint8")
    out = cv2.LUT(img, table)

    ok, buf = cv2.imencode(".png", out)
    return buf.tobytes() if ok else image_bytes


def build_payload(files, use_gamma: bool, gamma: float) -> List[Dict[str, Any]]:
    """
    Send ORIGINAL always; optionally also send GAMMA-enhanced.
    """
    payload: List[Dict[str, Any]] = []
    if not files:
        return payload

    for i, f in enumerate(files, start=1):
        raw = f.getvalue()
        mime = f.type or "image/jpeg"

        payload.append({"type": "text", "text": f"[Image {i}] ORIGINAL"})
        payload.append({"type": "image_url", "image_url": {"url": b64_image(raw, mime)}})

        if use_gamma:
            enh = gamma_preprocess(raw, gamma=gamma)
            payload.append({"type": "text", "text": f"[Image {i}] GAMMA enhanced (gamma={gamma})"})
            payload.append({"type": "image_url", "image_url": {"url": b64_image(enh, "image/png")}})

    return payload


# -----------------------------
# RAG (FAISS)
# -----------------------------
@st.cache_resource
def load_rag_index(rag_dir: str) -> Tuple[bool, Optional[Any], Optional[List[Dict[str, Any]]]]:
    if faiss is None or np is None:
        return (False, None, None)

    ip = os.path.join(rag_dir, "index.faiss")
    mp = os.path.join(rag_dir, "meta.json")
    if not (os.path.exists(ip) and os.path.exists(mp)):
        return (False, None, None)

    try:
        idx = faiss.read_index(ip)
        with open(mp, "r", encoding="utf-8") as f:
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
        for rid in I[0].tolist():
            if 0 <= rid < len(meta):
                ch = meta[rid]
                hits.append((ch.get("title", f"chunk_{rid}"), ch.get("text", "")))

        status["hits"] = len(hits)
        if not hits:
            return ("", status)

        out = ["REFERENCE CARDS (RAG):"]
        for n, (t, tx) in enumerate(hits, start=1):
            out.append(f"\n--- CARD {n}: {t} ---\n{tx}\n")
        return ("\n".join(out), status)

    except Exception as e:
        status["error"] = str(e)
        return ("", status)


# -----------------------------
# VISION-FIRST MORPHOLOGY PASS
# -----------------------------
def morphology_pass(
    client: OpenAI,
    clinical_text: str,
    images_original_only: List[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    """
    Returns raw bullets + parsed keyword list.
    """
    system = (
        "You are a retina imaging morphology summarizer for DE-IDENTIFIED retinal images (NOT faces). "
        "Do NOT diagnose. Output 10-18 bullet keywords/phrases describing visible morphology. "
        "Include: diffuse hypopigmentation? prominent choroidal vessels? macular reflex/foveal definition? "
        "optic disc appearance (general), vessels (tortuosity/attenuation), hemorrhage/exudates, "
        "any focal lesion with shape+location."
    )

    user = (
        "Return ONLY a bullet list (each line starts with '-') with 10-18 items.\n\n"
        f"CLINICAL:\n{clinical_text.strip() if clinical_text.strip() else '(none)'}"
    )

    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": user}] + images_original_only},
    ]

    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=msgs,
        temperature=0.0,
        max_tokens=320,
    )
    raw = (resp.choices[0].message.content or "").strip()

    kws = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("-"):
            kws.append(line.lstrip("-").strip())

    return raw, kws


def rag_gate(keywords: List[str], clinical_text: str) -> bool:
    """
    Gate RAG to avoid wrong anchoring.
    """
    t = (" ".join(keywords) + " " + (clinical_text or "")).lower()

    triggers = [
        # special shapes / lesions
        "torpedo", "teardrop", "nevus", "melanoma", "orange pigment",
        "disc pit", "morning glory", "coloboma",
        # inherited / congenital hints
        "diffuse hypopigmentation", "hypopigmented fundus", "prominent choroidal vessels",
        "bone spicule", "attenuated vessels", "waxy pallor",
        # inflammatory patterns
        "white dot", "placoid", "serpigin", "birdshot",
    ]
    return any(x in t for x in triggers)


def main_analysis(
    client: OpenAI,
    clinical_text: str,
    image_payload_main: List[Dict[str, Any]],
    morphology_keywords: List[str],
    rag_context: str,
) -> str:
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if rag_context:
        messages.append({"role": "system", "content": rag_context})

    user_text = (
        "DE-IDENTIFIED retinal images (NOT faces). Provide morphology-first analysis and EDUCATIONAL differential.\n\n"
        f"CLINICAL:\n{clinical_text.strip() if clinical_text.strip() else '(not provided)'}\n\n"
        "MORPHOLOGY KEYWORDS (from a separate vision-only pass; treat as hints, but follow the images):\n"
        + "\n".join([f"- {k}" for k in morphology_keywords[:18]])
    )

    messages.append({"role": "user", "content": [{"type": "text", "text": user_text}] + image_payload_main})

    resp = client.chat.completions.create(
        model=VISION_MODEL,   # main analysis uses gpt-4o
        messages=messages,
        temperature=0.2,
        max_tokens=1800,
    )
    return resp.choices[0].message.content or ""


# -----------------------------
# OPTIONAL: JSON FIXER PASS (format model)
# -----------------------------
def fix_json_if_needed(client: OpenAI, text: str) -> str:
    """
    If model fails strict JSON, attempt a cheap "format repair" pass with FORMAT_MODEL.
    """
    if "```json" in text:
        # already has a json fence - still might be invalid though
        parsed = parse_json_block(text)
        if parsed is not None:
            return text

    system = (
        "You are a strict JSON repair assistant. "
        "Given an ophthalmology report draft, output TWO blocks exactly:\n"
        "(A) strict valid JSON inside ```json ... ``` following the schema provided\n"
        "(B) the human-readable report.\n"
        "Do not add extra text. Ensure JSON parses."
    )

    user = (
        "Repair the output to match strict JSON + report format.\n\n"
        "SCHEMA reminder:\n"
        + SYSTEM_PROMPT.split("JSON schema:")[1].split("HARD RULE")[0].strip()
        + "\n\nORIGINAL OUTPUT:\n"
        + text
    )

    resp = client.chat.completions.create(
        model=FORMAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=1900,
    )
    return resp.choices[0].message.content or text


# -----------------------------
# SESSION
# -----------------------------
def init_session():
    for k, v in {
        "morph_raw": "",
        "morph_keywords": [],
        "rag_context": "",
        "rag_status": {"enabled": False, "hits": 0, "index_dim": None, "error": None},
        "debug": [],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

def log(msg: str):
    st.session_state.debug.append(msg)

init_session()


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

    # RAG defaults
    use_rag = st.checkbox("Enable RAG (FAISS)", value=False)
    rag_mode_auto = st.checkbox("RAG auto-gate (recommended)", value=True)
    rag_k = st.slider("RAG hits (k)", 2, 10, MAX_RAG_HITS, 1)

    st.divider()

    # Preprocess defaults
    use_gamma = st.checkbox("Preprocess: Gamma (gentle)", value=False)
    gamma_val = st.slider("Gamma", 0.9, 1.6, 1.15, 0.05)

    st.divider()

    use_json_fixer = st.checkbox("Auto-fix JSON (gpt-4o-mini)", value=True)

    enabled_idx, idx_obj, meta_obj = load_rag_index(RAG_DIR)
    st.caption(f"RAG index loaded? {enabled_idx} | meta_len={len(meta_obj) if meta_obj else 0}")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Clinical metadata")
    clinical_text = st.text_area(
        "Age, Sex, Symptoms, Duration, Laterality, History",
        height=150,
        placeholder="Example:\nAge: 37\nSex: Female\nSymptoms: lifelong low vision + nystagmus\nDuration: chronic\nLaterality: bilateral\nHistory: congenital"
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
    with st.expander("Steps", expanded=True):
        if st.session_state.debug:
            for s in st.session_state.debug[-15:]:
                st.write(s)
        else:
            st.caption("(no steps yet)")

    with st.expander("Morphology pass output (debug)", expanded=False):
        st.text(st.session_state.morph_raw if st.session_state.morph_raw else "(Run Analyze)")

    with st.expander("RAG status / context (debug)", expanded=False):
        st.json(st.session_state.rag_status)
        st.text(st.session_state.rag_context if st.session_state.rag_context else "(No RAG context)")


# -----------------------------
# RUN
# -----------------------------
if run:
    st.session_state.debug = []
    log("Step 0: Analyze clicked ✅")

    if not files:
        st.warning("En az 1 görüntü yükle.")
        st.stop()

    # Main payload (original + optional gamma)
    payload_main = build_payload(files, use_gamma=use_gamma, gamma=gamma_val)

    # Original-only payload for morphology pass
    payload_original = []
    for i, f in enumerate(files, start=1):
        raw = f.getvalue()
        mime = f.type or "image/jpeg"
        payload_original.append({"type": "text", "text": f"[Image {i}] ORIGINAL"})
        payload_original.append({"type": "image_url", "image_url": {"url": b64_image(raw, mime)}})

    # 1) Morphology pass (gpt-4o)
    log("Step 1: morphology pass (gpt-4o) ...")
    morph_raw, morph_keywords = morphology_pass(client, clinical_text, payload_original)
    st.session_state.morph_raw = morph_raw
    st.session_state.morph_keywords = morph_keywords
    log(f"Step 1 done ✅ keywords={len(morph_keywords)}")

    # 2) RAG gated
    rag_context = ""
    rag_status = {"enabled": False, "hits": 0, "index_dim": None, "error": None}
    rag_should_run = False
    if use_rag:
        if rag_mode_auto:
            rag_should_run = rag_gate(morph_keywords, clinical_text)
            log(f"Step 2: RAG gate auto={rag_should_run}")
        else:
            rag_should_run = True
            log("Step 2: RAG manual ON")

    if use_rag and rag_should_run:
        query = (clinical_text.strip() + "\n" + "\n".join(morph_keywords)).strip()
        rag_context, rag_status = rag_retrieve(client, query, k=rag_k)
        log(f"Step 2 done ✅ RAG hits={rag_status.get('hits')}")
    else:
        log("Step 2 skipped (RAG off or gate=false)")

    st.session_state.rag_context = rag_context
    st.session_state.rag_status = rag_status

    # 3) Main analysis (gpt-4o)
    log("Step 3: main analysis (gpt-4o) ...")
    out = main_analysis(client, clinical_text, payload_main, morph_keywords, rag_context)
    log("Step 3 done ✅")

    # 4) Optional JSON fixer (gpt-4o-mini)
    if use_json_fixer:
        parsed = parse_json_block(out)
        if parsed is None:
            log("Step 4: JSON fixer (gpt-4o-mini) ...")
            out2 = fix_json_if_needed(client, out)
            if out2:
                out = out2
            log("Step 4 done ✅")

    st.subheader("Assistant output")
    st.markdown(out)

    parsed = parse_json_block(out)
    if parsed:
        if isinstance(parsed.get("differential"), list):
            parsed["differential"] = normalize_probs(parsed["differential"], "probability")
        st.subheader("Structured output (debug)")
        st.json(parsed)
    else:
        st.warning("JSON parse edilemedi. JSON fixer açıkken bile parse olmadıysa, çıktıyı buraya yapıştır; düzeltelim.")
