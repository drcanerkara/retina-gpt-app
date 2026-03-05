import os
import json
import base64
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from openai import OpenAI

# Optional RAG deps
try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    faiss = None
    np = None

# Optional preprocessing
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


APP_TITLE = "RetinaGPT v2 (Vision-first)"
APP_SUBTITLE = "Image-first morphology → then differential (Educational only)"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RAG_DIR = os.getenv("RAG_DIR", "data")
MAX_RAG_HITS = int(os.getenv("MAX_RAG_HITS", "6"))

st.set_page_config(page_title=APP_TITLE, page_icon="👁️", layout="wide")


SYSTEM_PROMPT = """
You are RetinaGPT v2, a retina subspecialty educational imaging discussion and decision-support system.

IMPORTANT CONTEXT
- DE-IDENTIFIED ophthalmic images (retina/OCT/FAF/angiography). Not faces/people.
- Do NOT identify any person. Focus on imaging morphology.
- Educational differential only. No patient-specific treatment instructions.

STYLE
- Formal medical English. Morphology first. Then ranked differential.
- If uncertain, say UNCERTAIN and give discriminators.

RAG
If "REFERENCE CARDS" are provided, treat them as supportive background.
Do NOT let RAG override what is visible in the images.

OUTPUT
Return TWO blocks:
(A) STRICT JSON in ```json
(B) Human report

Use this JSON schema keys:
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

Hard rule:
- If diffuse fundus hypopigmentation + prominent choroidal vessels AND history of nystagmus/lifelong low vision → include "Albinism spectrum / Ocular albinism" in TOP differential and suggest OCT for foveal hypoplasia (educational).
"""


def b64_image(file_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(file_bytes).decode('utf-8')}"


def safe_get_api_key() -> Optional[str]:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


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


def maybe_clahe(image_bytes: bytes) -> bytes:
    if cv2 is None or np is None:
        return image_bytes
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    ok, buf = cv2.imencode(".png", out)
    return buf.tobytes() if ok else image_bytes


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
        return (True, idx, meta if isinstance(meta, list) else None)
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
        for j, rid in enumerate(I[0].tolist()):
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


def build_image_payload(files, use_clahe: bool) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    if not files:
        return payload
    for i, f in enumerate(files, start=1):
        raw = f.getvalue()
        mime = f.type or "image/jpeg"
        payload.append({"type": "text", "text": f"[Image {i}] ORIGINAL"})
        payload.append({"type": "image_url", "image_url": {"url": b64_image(raw, mime)}})
        if use_clahe:
            enh = maybe_clahe(raw)
            payload.append({"type": "text", "text": f"[Image {i}] ENHANCED (CLAHE)"})
            payload.append({"type": "image_url", "image_url": {"url": b64_image(enh, "image/png")}})
    return payload


def vision_first_keywords(
    client: OpenAI,
    model_name: str,
    clinical_text: str,
    image_payload_original_only: List[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    """
    Returns:
      raw_text, keywords(list)
    """
    system = (
        "You are a retina imaging morphology summarizer for DE-IDENTIFIED retinal images (NOT faces). "
        "Do NOT diagnose. Output 8-15 short keywords/phrases that describe what is VISIBLE. "
        "Include: pigmentation (hypopigmented fundus?), choroidal vessel visibility, foveal reflex, "
        "optic disc appearance, macular changes, hemorrhage/exudates, any focal lesion (shape/location)."
    )
    user = (
        "Return ONLY a bullet list (each line starts with '-') with 8-15 items.\n\n"
        f"CLINICAL:\n{clinical_text.strip() if clinical_text.strip() else '(none)'}"
    )
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": user}] + image_payload_original_only},
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=msgs,
        temperature=0.0,
        max_tokens=250,
    )
    raw = (resp.choices[0].message.content or "").strip()
    kws = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("-"):
            kws.append(line.lstrip("-").strip())
    return raw, kws


def should_use_rag(keywords: List[str], clinical_text: str) -> bool:
    """
    Gate RAG to avoid 'wrong anchor' cards.
    Use RAG only if clear disease-hints appear.
    """
    text = (" ".join(keywords) + " " + (clinical_text or "")).lower()

    triggers = [
        "torpedo", "teardrop", "temporal to fovea",
        "diffuse hypopigmentation", "hypopigmented fundus", "prominent choroidal vessels",
        "white dot", "placoid", "serpigin", "birdshot",
        "bone spicule", "attenuated vessels", "waxy pallor",
        "nevus", "melanoma", "orange pigment",
        "coloboma", "disc pit", "morning glory",
    ]
    return any(t in text for t in triggers)


def call_main(
    client: OpenAI,
    model_name: str,
    clinical_text: str,
    image_payload: List[Dict[str, Any]],
    keywords: List[str],
    rag_context: str,
) -> str:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if rag_context:
        messages.append({"role": "system", "content": rag_context})

    user_text = (
        "DE-IDENTIFIED retinal images (NOT faces). Provide morphology-first analysis and EDUCATIONAL differential.\n\n"
        f"CLINICAL:\n{clinical_text.strip() if clinical_text.strip() else '(not provided)'}\n\n"
        "VISION-FIRST MORPHOLOGY KEYWORDS (from a separate morphology pass; do not override the images if wrong):\n"
        + "\n".join([f"- {k}" for k in keywords[:15]])
    )
    messages.append({"role": "user", "content": [{"type": "text", "text": user_text}] + image_payload})

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=1800,
    )
    return resp.choices[0].message.content or ""


# -----------------------------
# SESSION
# -----------------------------
def init_session():
    for k, v in {
        "kw_raw": "",
        "kw_list": [],
        "rag_context": "",
        "rag_status": {"enabled": False, "hits": 0, "index_dim": None, "error": None},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# -----------------------------
# UI
# -----------------------------
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

api_key = safe_get_api_key()
if not api_key:
    st.error("OPENAI_API_KEY bulunamadı.")
    st.stop()

client = OpenAI(api_key=api_key)

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", options=list(dict.fromkeys([DEFAULT_MODEL, "gpt-4o"])), index=0)

    # IMPORTANT: default RAG off
    use_rag = st.checkbox("Enable RAG (FAISS)", value=False)
    rag_k = st.slider("RAG hits (k)", 2, 10, MAX_RAG_HITS, 1)

    # IMPORTANT: default CLAHE off
    use_clahe = st.checkbox("Preprocess images (CLAHE)", value=False)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Clinical metadata")
    clinical_text = st.text_area(
        "Age, Sex, Symptoms, Duration, Laterality, History",
        height=140,
        placeholder="Example:\nAge: 30\nSex: Female\nSymptoms: lifelong low vision + nystagmus\nDuration: chronic\nLaterality: bilateral\nHistory: congenital"
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
    with st.expander("Vision-first keywords (debug)", expanded=True):
        if st.session_state.kw_raw:
            st.text(st.session_state.kw_raw)
        else:
            st.caption("(Run Analyze)")

    with st.expander("RAG status / retrieved cards (debug)", expanded=False):
        st.json(st.session_state.rag_status)
        st.text(st.session_state.rag_context if st.session_state.rag_context else "(No RAG context)")

if run:
    if not files:
        st.warning("En az 1 görüntü yüklemen daha iyi olur.")
    # Build payloads
    payload_main = build_image_payload(files, use_clahe=use_clahe)

    # ORIGINAL-only payload for vision-first pass
    payload_original_only = []
    for i, f in enumerate(files, start=1):
        raw = f.getvalue()
        mime = f.type or "image/jpeg"
        payload_original_only.append({"type": "text", "text": f"[Image {i}] ORIGINAL"})
        payload_original_only.append({"type": "image_url", "image_url": {"url": b64_image(raw, mime)}})

    # 1) Vision-first keywords
    kw_raw, kw_list = vision_first_keywords(client, "gpt-4o", clinical_text, payload_original_only)
    st.session_state.kw_raw = kw_raw
    st.session_state.kw_list = kw_list

    # 2) Gate RAG
    rag_context = ""
    rag_status = {"enabled": False, "hits": 0, "index_dim": None, "error": None}
    if use_rag and should_use_rag(kw_list, clinical_text):
        query = (clinical_text.strip() + "\n" + "\n".join(kw_list)).strip()
        rag_context, rag_status = rag_retrieve(client, query, k=rag_k)
    st.session_state.rag_context = rag_context
    st.session_state.rag_status = rag_status

    # 3) Main analysis
    out = call_main(client, model, clinical_text, payload_main, kw_list, rag_context)

    st.subheader("Assistant output")
    st.markdown(out)

    parsed = parse_json_block(out)
    if parsed:
        if isinstance(parsed.get("differential"), list):
            parsed["differential"] = normalize_probs(parsed["differential"])
        st.subheader("Structured output (debug)")
        st.json(parsed)
    else:
        st.warning("JSON parse edilemedi. (Model strict JSON üretmemiş olabilir.)")
