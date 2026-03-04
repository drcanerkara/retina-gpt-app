import streamlit as st
import base64
import json
import faiss
import numpy as np
from openai import OpenAI

# ---------- Sayfa Yapılandırması ----------
st.set_page_config(page_title="RetinaGPT", page_icon="👁️", layout="wide")

# ---------- RAG FONKSİYONLARI (Kritik Bölüm) ----------
@st.cache_resource
def load_rag_assets():
    """Veritabanını ve Metadata'yı yükler."""
    try:
        # Görselindeki 'data/' klasör yapısına göre ayarlandı
        index = faiss.read_index("data/index.faiss")
        with open("data/meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    except Exception as e:
        st.error(f"RAG dosyaları yüklenemedi: {e}")
        return None, None

def get_embedding(text, client):
    """Metni arama yapılabilir vektöre çevirir."""
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def search_rag(query_text, client, index, meta, k=3):
    """Veritabanında arama yapar."""
    if not index or not meta or not query_text.strip():
        return ""
    
    query_vector = get_embedding(query_text, client)
    distances, indices = index.search(np.array([query_vector]).astype('float32'), k)
    
    results = []
    for i in indices[0]:
        if i != -1 and i < len(meta):
            results.append(meta[i].get("text", ""))
    return "\n\n".join(results)

# ---------- ORİJİNAL PROMPT (KORUNDU) ----------
SYSTEM_PROMPT = """
You are RetinaGPT, a retina subspecialty educational discussion and decision-support system.
[... Senin orijinal 13 adımlık promptunun tamamı buraya gelecek ...]
"""

# ---------- API Kurulumu ----------
api_key = st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    st.error("API Key bulunamadı. Lütfen Streamlit Secrets'a ekleyin.")
    st.stop()

client = OpenAI(api_key=api_key)
index, meta = load_rag_assets()

def file_to_data_url(file) -> str:
    return f"data:{file.type};base64,{base64.b64encode(file.getvalue()).decode('utf-8')}"

# ---------- UI TASARIMI ----------
st.markdown("<h1 style='text-align: center;'>👁️ RetinaGPT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Prepared by Mehmet ÇITIRIK & Caner KARA</p>", unsafe_allow_html=True)
st.markdown("---")

clinical_text = st.text_area("Please provide clinical details", placeholder="Age, Symptoms, CNRKR123456...", height=100)
uploaded_files = st.file_uploader("Upload images", type=["jpg", "png", "webp"], accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(len(uploaded_files))
    for idx, f in enumerate(uploaded_files):
        cols[idx].image(f, caption=f.name)

analyze = st.button("🔍 Analyze Case", use_container_width=True)

# ---------- ÇİFT AŞAMALI (DUAL-PASS) ANALİZ ----------
if analyze:
    if not uploaded_files:
        st.error("Lütfen resim yükleyin.")
        st.stop()

    with st.spinner("AI scanning image... (Dual-Pass Phase 1)"):
        # 1. AŞAMA: Görseli Tarama (Keywords üretme)
        vision_blocks = [{"type": "text", "text": "Describe the main finding in 3-5 technical keywords for a database search."}]
        for f in uploaded_files:
            vision_blocks.append({"type": "image_url", "image_url": {"url": file_to_data_url(f)}})
        
        vision_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": vision_blocks}],
            max_tokens=30
        )
        ai_keywords = vision_resp.choices[0].message.content
        st.info(f"AI Keywords: {ai_keywords}")

    with st.spinner("Searching RAG and Finalizing... (Phase 2)"):
        # 2. AŞAMA: RAG Araması (Kullanıcı metni + AI anahtar kelimeleri)
        full_query = f"{clinical_text} {ai_keywords}"
        rag_content = search_rag(full_query, client, index, meta)

        # Görünürlük testi için RAG kutusu
        with st.expander("📚 RAG Search Results (See what AI found)"):
            st.write(rag_content if rag_content else "No cards found.")

        # 3. AŞAMA: Final Rapor
        final_blocks = []
        if rag_content:
            final_blocks.append({"type": "text", "text": f"### REFERENCE CARDS (RAG):\n{rag_content}\n\n---"})
        
        final_blocks.append({"type": "text", "text": f"CLINICAL DATA: {clinical_text}\nAI SCAN: {ai_keywords}"})
        for f in uploaded_files:
            final_blocks.append({"type": "image_url", "image_url": {"url": file_to_data_url(f)}})

        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": final_blocks}],
                max_tokens=2500
            )
            st.markdown(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Error: {e}")
