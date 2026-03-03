import streamlit as st
import base64
import json
import os
from typing import List, Dict, Any, Optional

import numpy as np
from openai import OpenAI

# FAISS is optional: app still runs without RAG
try:
    import faiss  # type: ignore
except Exception:
    faiss = None


# ---------------- Page config ----------------
st.set_page_config(page_title="RetinaGPT", page_icon="👁️")


# ---------------- System prompt ----------------
SYSTEM_PROMPT = """
You are a retina subspecialty educational discussion system.
Your purpose is to provide structured academic discussion of retinal imaging findings and differential diagnostic reasoning for educational purposes.

All outputs must be structured, objective, concise, and written in formal medical English.

TERMINOLOGY STANDARD:
Use formal ophthalmic subspecialty terminology (e.g., cotton-wool spot, ellipsoid zone, RPE disruption, outer retinal cavitation).

STRUCTURED ANALYSIS FORMAT:

1) Imaging Quality
2) Structural Findings
3) Vascular Findings
4) Peripheral Assessment
5) Pattern Discussion
6) Pathophysiologic Considerations
7) Differential Diagnosis (max 3)
8) Confidence Level (Low / Moderate / High)

IMPORTANT INTERPRETATION RULES:

• Do NOT label a lesion as a “mass,” “tumor,” or “elevated lesion” unless there is clear dome-shaped thickening or a discrete hyperreflective solid structure on OCT.

• If OCT shows outer retinal thinning, RPE disruption, ellipsoid zone loss, or outer retinal cavitation/excavation WITHOUT a solid mass, describe it as an outer retinal/RPE abnormality — not a tumor.

• If a lesion appears flat, well-demarcated, and non-exudative on color fundus, prioritize congenital or RPE-related anomalies before neoplastic causes.

• When evaluating OCT, explicitly comment on:
  - Ellipsoid zone integrity
  - RPE continuity
  - Presence or absence of subretinal fluid
  - Presence or absence of a solid hyperreflective mass
  - Outer retinal cavitation or focal excavation
  - Choroidal contour beneath the lesion

• If a solitary, well-demarcated hypopigmented lesion temporal to the fovea with a torpedo/ovoid configuration is observed,
  and OCT demonstrates outer retinal/RPE alteration with or without cavitation, explicitly consider TORPEDO MACULOPATHY among the top differentials.

• If findings appear atypical or rare, avoid over-commitment. First describe morphology precisely, then provide differential diagnoses.

ANALYSIS FLOW:
First provide:
“Findings only (no diagnosis).”
Then provide:
“Differential diagnosis (max 3)” strictly based on the described findings.

Include:
• Arguments for the top diagnosis
• Arguments against the top diagnosis
• Additional imaging/tests that would clarify the pattern

LIMITATIONS:
Educational purposes only. Not medical advice.
"""

# ---------------- RAG paths ----------------
INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.json"

# ---------------- API key (hidden) ----------------
# Supports either OPENAI_API_KEY or OPEN_API_KEY if you ever used that name before.
api_key = None
try:
    api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("OPEN_API_KEY")
except Exception:
    api_key = None

if not api_key:
    st.error("OPENAI_API_KEY not found. Please add it in the Streamlit Cloud Secrets panel.")
    st.stop()

client = OpenAI(api_key=api_key)


# ---------------- State init ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "case_done" not in st.session_state:
    st.session_state.case_done = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def reset_case():
    st.session_state.messages = []
    st.session_state.case_done = False
    st.session_state.uploader_key += 1  # resets uploader
    st.rerun()


def file_to_data_url(file) -> str:
    """Convert an uploaded file to a data URL (safe for multiple files)."""
    b = file.getvalue()  # do NOT use read() repeatedly
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{file.type};base64,{b64}"


# ---------------- RAG loader ----------------
@st.cache_resource(show_spinner=False)
def load_rag_assets() -> Dict[str, Any]:
    """
    Loads FAISS index + metadata if available.
    Returns dict with keys: enabled(bool), index, meta(list), dim(int), error(str|None)
    """
    out = {"enabled": False, "index": None, "meta": None, "dim": None, "error": None}

    if faiss is None:
        out["error"] = "FAISS is not installed. RAG disabled."
        return out

    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        out["error"] = "RAG index files not found (data/index.faiss + data/meta.json). RAG disabled."
        return out

    try:
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        out["enabled"] = True
        out["index"] = index
        out["meta"] = meta
        out["dim"] = index.d
        return out
    except Exception as e:
        out["error"] = f"Failed to load RAG assets: {e}"
        return out


def embed_text(text: str) -> np.ndarray:
    """
    Returns a normalized embedding vector (float32) for retrieval.
    """
    # Keep it compact to reduce cost; retrieval doesn't need the largest model.
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8000],  # safety cap
    ).data[0].embedding
    v = np.array(emb, dtype=np.float32)

    # Normalize for cosine-style similarity if index was built that way;
    # even if not, normalization is usually safe.
    norm = np.linalg.norm(v) + 1e-12
    v = v / norm
    return v


def retrieve_cards(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Searches FAISS index and returns list of {title, text, score, id}.
    If RAG not available, returns [].
    """
    assets = load_rag_assets()
    if not assets["enabled"]:
        return []

    index = assets["index"]
    meta = assets["meta"]

    qv = embed_text(query).reshape(1, -1)
    # Some indices are IP, some L2. We just take the top_k results.
    D, I = index.search(qv, top_k)

    results = []
    for rank, idx in enumerate(I[0].tolist()):
        if idx < 0 or idx >= len(meta):
            continue
        item = meta[idx]
        # expected meta fields: {"id":..., "title":..., "text":...} (your ingest writes this)
        results.append(
            {
                "id": item.get("id", idx),
                "title": item.get("title", f"Card {idx}"),
                "text": item.get("text", ""),
                "score": float(D[0][rank]),
            }
        )
    return results


def build_rag_context(cards: List[Dict[str, Any]], max_chars: int = 3500) -> str:
    """
    Creates a compact reference block to inject into the prompt.
    """
    if not cards:
        return ""

    chunks = []
    for c in cards:
        title = (c.get("title") or "").strip()
        text = (c.get("text") or "").strip()

        # Keep each card short to avoid token explosion.
        text = text.replace("\n\n", "\n").strip()
        if len(text) > 900:
            text = text[:900].rstrip() + "…"

        chunks.append(f"[{title}]\n{text}")

    blob = "\n\n---\n\n".join(chunks)
    if len(blob) > max_chars:
        blob = blob[:max_chars].rstrip() + "…"
    return blob


# ---------------- Header (centered) ----------------
st.markdown(
    """
    <div style="text-align: center;">
        <h1>👁️ RetinaGPT</h1>
        <p style="font-size:16px; margin-top:-10px;">
            Prepared by Mehmet ÇITIRIK & Caner KARA
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")


# ---------------- Clinical details (ABOVE image upload) ----------------
clinical_text = st.text_area(
    "Please provide clinical details",
    placeholder="Age, Sex, Symptoms, Duration, Laterality, History",
    height=110,
    disabled=st.session_state.case_done,
)

# ---------------- Upload (multi-file) ----------------
uploaded_files = st.file_uploader(
    "Please upload retinal imaging (Fundus / OCT / FAF / FA) — jpg/png/webp",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
    disabled=st.session_state.case_done,
)

# ---------------- RAG status (small, not noisy) ----------------
assets = load_rag_assets()
with st.expander("RAG status (optional)", expanded=False):
    if assets["enabled"]:
        st.success("RAG enabled: FAISS index + meta.json loaded.")
        st.caption(f"Index dim: {assets.get('dim')}")
    else:
        st.warning(f"RAG disabled: {assets.get('error')}")


# ---------------- Preview (optional but helpful) ----------------
if uploaded_files and len(uploaded_files) > 0:
    st.subheader("Image Preview")
    for i, f in enumerate(uploaded_files, start=1):
        st.image(f, caption=f"Image {i}: {f.name}")

# ---------------- Chat history ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------- New Patient button after analysis ----------------
if st.session_state.case_done:
    st.divider()
    if st.button("🧼 Ask New Patient", use_container_width=True):
        reset_case()
    st.caption("Clears the current case (messages + images) and starts a fresh patient.")


# ---------------- Analyze button ----------------
analyze = st.button("🔍 Analyze", use_container_width=True, disabled=st.session_state.case_done)

if analyze:
    if not uploaded_files or len(uploaded_files) == 0:
        st.error("Please upload at least one image (required for analysis).")
        st.stop()

    user_payload = (clinical_text or "").strip()
    if not user_payload:
        user_payload = "No clinical details provided."

    # Store user message for display/history
    st.session_state.messages.append({"role": "user", "content": user_payload})
    with st.chat_message("user"):
        st.markdown(user_payload)

    # ---------------- RAG retrieve ----------------
    # Use clinical text + filenames to enrich retrieval.
    fname_blob = " ".join([f.name for f in uploaded_files])
    rag_query = f"{user_payload}\n\nImages: {fname_blob}"

    cards = retrieve_cards(rag_query, top_k=5)
    rag_context = build_rag_context(cards)

    # Build multimodal content blocks (multiple images in one case)
    content_blocks: List[Dict[str, Any]] = [
        {"type": "input_text", "text": user_payload},
        {"type": "input_text", "text": "Multiple images uploaded. Interpret them as a single case and integrate findings across modalities."},
    ]

    if rag_context:
        content_blocks.append(
            {
                "type": "input_text",
                "text": "Reference knowledge (RAG). Use ONLY if relevant; do not force-fit diagnoses:\n\n" + rag_context,
            }
        )

    for i, f in enumerate(uploaded_files, start=1):
        content_blocks.append({"type": "input_text", "text": f"Image {i} filename: {f.name}"})
        content_blocks.append({"type": "input_image", "image_url": file_to_data_url(f)})

    with st.chat_message("assistant"):
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *[
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ],
                {"role": "user", "content": content_blocks},
            ],
        )

        full_response = response.output_text
        st.markdown(full_response)

        # Optional: show which cards were retrieved (debug)
        if cards:
            with st.expander("Retrieved RAG cards (debug)", expanded=False):
                for c in cards:
                    st.markdown(f"- **{c.get('title')}** (score: {c.get('score'):.4f})")

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # finish case -> show New Patient button
    st.session_state.case_done = True
    st.rerun()
