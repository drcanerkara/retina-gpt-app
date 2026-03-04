import streamlit as st
import base64
import json
import faiss
import numpy as np
from openai import OpenAI

# Sayfa Yapılandırması
st.set_page_config(page_title="RetinaGPT", page_icon="👁️")

# ---------- RAG FONKSİYONLARI ----------

@st.cache_resource
def load_rag_assets():
    """FAISS indexini ve metadata (meta.json) dosyalarını yükler."""
    try:
        # Görseldeki yapıya göre index ve meta dosyalarını yüklüyoruz
        index = faiss.read_index("data/index.faiss")
        with open("data/meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    except Exception as e:
        st.error(f"RAG dosyaları yüklenemedi: {e}")
        return None, None

def get_embedding(text, client, model="text-embedding-3-small"):
    """Kullanıcının metnini arama yapmak için vektöre dönüştürür."""
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def search_rag(query_text, client, index, meta, k=3):
    """Vektör veritabanında en yakın sonuçları bulur ve metinleri çeker."""
    if not index or not meta:
        return ""
    
    # 1. Metni vektöre çevir
    query_vector = get_embedding(query_text, client)
    
    # 2. FAISS üzerinde arama yap
    distances, indices = index.search(np.array([query_vector]).astype('float32'), k)
    
    # 3. meta.json yapısına göre metinleri topla
    results = []
    for i in indices[0]:
        if i != -1 and i < len(meta):
            # meta.json içindeki "text" alanını çekiyoruz
            entry_text = meta[i].get("text", "")
            source = meta[i].get("source_file", "Retina Cards")
            results.append(f"[KAYNAK: {source}]\n{entry_text}")
    
    return "\n\n".join(results)

# ---------- PROMPT ----------

SYSTEM_PROMPT = """
You are RetinaGPT, a retina subspecialty educational discussion and decision-support system.

PURPOSE
Provide structured academic discussion of retinal imaging findings and differential diagnostic reasoning for educational purposes only.

STYLE
- Formal medical English, objective, concise.
- Use retina subspecialty terminology.

REFERENCE KNOWLEDGE (RAG)
If "Reference knowledge (RAG)" is provided, you MUST treat it as the primary factual source. 
Use it to refine discriminators, pitfalls, work-up, and management.

[... Orijinal promptundaki diğer tüm adımlar buraya gelecek ...]
"""

# ---------- KURULUM VE API ----------

api_key = st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    st.error("Lütfen Streamlit Secrets'a OPENAI_API_KEY ekleyin.")
    st.stop()

client = OpenAI(api_key=api_key)
index, meta = load_rag_assets()

def file_to_data_url(file) -> str:
    b = file.getvalue()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{file.type};base64,{b64}"

# ---------- UI ----------
st.markdown(
    """
    <div style="text-align: center;">
        <h1>👁️ RetinaGPT</h1>
        <p style="font-size:16px; margin-top:-10px;">Prepared by Mehmet ÇITIRIK & Caner KARA</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

clinical_text = st.text_area(
    "Clinical Details",
    placeholder="Age, Sex, Symptoms, Duration, History...",
    height=100,
)

uploaded_files = st.file_uploader(
    "Upload Retinal Imaging",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.subheader("Image Preview")
    cols = st.columns(len(uploaded_files))
    for i, f in enumerate(uploaded_files):
        cols[i].image(f, caption=f.name)

analyze = st.button("🔍 Analyze Case", use_container_width=True)

if analyze:
    if not uploaded_files:
        st.error("Please upload images.")
        st.stop()

    with st.spinner("RAG Knowledge Base searching and analyzing..."):
        # 1. RAG Araması Yap (Sadece klinik metin varsa arama yapıyoruz)
        rag_context = ""
        if clinical_text.strip():
            rag_context = search_rag(clinical_text, client, index, meta)

        # 2. Mesaj Yapısını Kur
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        user_content = []
        
        # Eğer RAG sonucu varsa en başa ekliyoruz
        if rag_context:
            user_content.append({
                "type": "text", 
                "text": f"### REFERENCE KNOWLEDGE (RAG):\n{rag_context}\n\n---"
            })
        
        user_content.append({
            "type": "text", 
            "text": f"CLINICAL CASE DATA:\n{clinical_text if clinical_text else 'No details provided.'}"
        })

        for f in uploaded_files:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": file_to_data_url(f)}
            })
        
        messages.append({"role": "user", "content": user_content})

        # 3. GPT-4o API Çağrısı
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=2500,
            )
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.markdown("---")
st.caption("Educational use only. Not for clinical diagnosis.")
