import streamlit as st
import base64
import json
import faiss
import numpy as np
import os
from openai import OpenAI

# ---------- Sayfa Yapılandırması ----------
st.set_page_config(page_title="RetinaGPT", page_icon="👁️", layout="wide")

# ---------- RAG SİSTEMİ (HATA AYIKLAMALI) ----------
@st.cache_resource
def load_rag_assets():
    index_path = "data/index.faiss"
    meta_path = "data/meta.json"
    
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        st.error(f"KRİTİK HATA: Veritabanı dosyaları bulunamadı! Yol: {os.getcwd()}")
        return None, None
    
    try:
        index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return None, None

def get_embedding(text, client):
    # Veritabanını oluştururken kullanılan modelle AYNI olmalı
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def search_rag(query_text, client, index, meta, k=3):
    if not index or not meta or not query_text.strip():
        return ""
    
    try:
        query_vector = get_embedding(query_text, client)
        distances, indices = index.search(np.array([query_vector]).astype('float32'), k)
        
        results = []
        for i in indices[0]:
            if i != -1 and i < len(meta):
                # meta.json yapısına göre 'text' alanını çekiyoruz
                content = meta[i].get("text", "No content found")
                results.append(content)
        return "\n\n".join(results)
    except Exception as e:
        return f"Arama sırasında hata: {e}"

# ---------- SYSTEM PROMPT (ORİJİNAL YAPINIZ) ----------
SYSTEM_PROMPT = """
You are RetinaGPT, a retina subspecialty educational discussion and decision-support system.
[... 13 ADIMLIK TÜM PROMPTUNUZU BURAYA YAPIŞTIRIN ...]
"""

# ---------- KURULUM ----------
api_key = st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
index, meta = load_rag_assets()

def file_to_data_url(file) -> str:
    return f"data:{file.type};base64,{base64.b64encode(file.getvalue()).decode()}"

# ---------- UI ----------
st.markdown("<h1 style='text-align: center;'>👁️ RetinaGPT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Prepared by Mehmet ÇITIRIK & Caner KARA</p>", unsafe_allow_html=True)
st.markdown("---")

clinical_text = st.text_area("Please provide clinical details", placeholder="CNRKR123456 testini buraya yazın...", height=100)
uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png", "webp"], accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(len(uploaded_files))
    for idx, f in enumerate(uploaded_files):
        cols[idx].image(f, caption=f.name)

analyze = st.button("🔍 Dual-Pass Analyze", use_container_width=True)

# ---------- ÇİFT GEÇİŞLİ ANALİZ MANTIĞI ----------
if analyze:
    if not uploaded_files:
        st.error("Lütfen görüntü yükleyin.")
        st.stop()

    # AŞAMA 1: VISION (Kendi kendine fark etmesi için)
    with st.spinner("AI scanning image..."):
        vision_content = [{"type": "text", "text": "Identify the key retinal finding. Respond with 3 technical words only (e.g. 'combined hamartoma' or 'torpedo maculopathy')."}]
        for f in uploaded_files:
            vision_content.append({"type": "image_url", "image_url": {"url": file_to_data_url(f)}})
        
        v_resp = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": vision_content}], max_tokens=20)
        ai_keywords = v_resp.choices[0].message.content
        st.info(f"AI Keywords: {ai_keywords}")

    # AŞAMA 2: RAG ARAMASI
    with st.spinner("Searching RAG database..."):
        # Hem kullanıcı metnini hem AI'nın bulduğu kelimeyi aratıyoruz
        combined_query = f"{clinical_text} {ai_keywords}"
        rag_context = search_rag(combined_query, client, index, meta)

        with st.expander("📚 RAG Search Results (VERİTABANI KONTROLÜ)"):
            if rag_context:
                st.success("Veritabanından bilgi çekildi!")
                st.markdown(rag_context)
            else:
                st.warning("Veritabanında eşleşme bulunamadı. CNRKR123456 yazdığınızdan emin olun.")

    # AŞAMA 3: FİNAL RAPOR
    with st.spinner("Generating final report..."):
        final_blocks = []
        if rag_context:
            final_blocks.append({"type": "text", "text": f"### REFERENCE CARDS (RAG):\n{rag_context}\n\n---"})
        
        final_blocks.append({"type": "text", "text": f"CLINICAL DATA: {clinical_text}\nVISION SCAN: {ai_keywords}"})
        for f in uploaded_files:
            final_blocks.append({"type": "image_url", "image_url": {"url": file_to_data_url(f)}})

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": final_blocks}],
            max_tokens=2500
        )
        st.markdown(resp.choices[0].message.content)

st.markdown("---")
st.caption("Educational use only.")
