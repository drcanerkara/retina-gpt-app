import streamlit as st
import base64
import json
import faiss
import numpy as np
from openai import OpenAI

# ---------- Sayfa Yapılandırması ----------
st.set_page_config(page_title="RetinaGPT", page_icon="👁️", layout="wide")

# ---------- RAG FONKSİYONLARI ----------
@st.cache_resource
def load_rag_assets():
    try:
        index = faiss.read_index("data/index.faiss")
        with open("data/meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    except Exception as e:
        st.error(f"RAG dosyaları yüklenemedi: {e}")
        return None, None

def get_embedding(text, client):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def search_rag(query_text, client, index, meta, k=3):
    if not index or not meta or not query_text.strip():
        return ""
    query_vector = get_embedding(query_text, client)
    distances, indices = index.search(np.array([query_vector]).astype('float32'), k)
    results = []
    for i in indices[0]:
        if i != -1 and i < len(meta):
            results.append(meta[i].get("text", ""))
    return "\n\n".join(results)

# ---------- SYSTEM PROMPT (ORİJİNAL - KORUNDU) ----------
SYSTEM_PROMPT = """
[Buraya senin ilk mesajdaki 13 adımlık devasa SYSTEM_PROMPT metnin gelecek...]
"""

# ---------- API Key ve Kurulum ----------
api_key = st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    st.error("API Key bulunamadı.")
    st.stop()

client = OpenAI(api_key=api_key)
index, meta = load_rag_assets()

def file_to_data_url(file) -> str:
    return f"data:{file.type};base64,{base64.b64encode(file.getvalue()).decode('utf-8')}"

# ---------- UI TASARIMI ----------
st.markdown("<div style='text-align: center;'><h1>👁️ RetinaGPT (Smart RAG)</h1></div>", unsafe_allow_html=True)
st.markdown("---")

clinical_text = st.text_area("Clinical Details", placeholder="Optional: Age, Sex, Symptoms...", height=100)
uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png", "webp"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("Image Preview")
    cols = st.columns(len(uploaded_files))
    for i, f in enumerate(uploaded_files):
        cols[i].image(f, caption=f.name)

analyze = st.button("🔍 Run Dual-Pass Analysis", use_container_width=True)

# ---------- ÇİFT AŞAMALI ANALİZ MANTIĞI ----------
if analyze:
    if not uploaded_files:
        st.error("Please upload images.")
        st.stop()

    with st.spinner("Step 1: AI is scanning images to identify keywords..."):
        # --- AŞAMA 1: VISION PASS (Görüntüden anahtar kelime çıkarma) ---
        vision_content = [{"type": "text", "text": "Describe the main retinal findings in this image in 5-10 technical words for a database search. (e.g. 'torpedo shaped hypopigmented lesion' or 'subretinal fluid with PED')"}]
        for f in uploaded_files:
            vision_content.append({"type": "image_url", "image_url": {"url": file_to_data_url(f)}})

        vision_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": vision_content}],
            max_tokens=50
        )
        ai_keywords = vision_resp.choices[0].message.content
        st.write(f"**AI Detected Keywords:** `{ai_keywords}`")

    with st.spinner("Step 2: Searching RAG database and generating final report..."):
        # --- AŞAMA 2: RAG ARAMASI (AI'nın bulduğu kelimeler + Kullanıcı metni ile) ---
        search_query = f"{clinical_text} {ai_keywords}"
        rag_context = search_rag(search_query, client, index, meta)

        # RAG Kontrol Paneli
        with st.expander("📚 RAG Knowledge Used"):
            st.info(rag_context if rag_context else "No matching cards found.")

        # Final Mesaj Blokları
        final_content = []
        if rag_context:
            final_content.append({"type": "text", "text": f"### REFERENCE CARDS (RAG):\n{rag_context}\n\n---"})
        
        final_content.append({"type": "text", "text": f"CLINICAL DATA: {clinical_text}\nAI PRE-SCAN: {ai_keywords}"})
        for f in uploaded_files:
            final_content.append({"type": "image_url", "image_url": {"url": file_to_data_url(f)}})

        # --- AŞAMA 3: FİNAL ANALİZ (Orijinal 8 Adımlı Rapor) ---
        try:
            final_resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": final_content},
                ],
                max_tokens=2500
            )
            st.markdown(final_resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Educational use only.")
