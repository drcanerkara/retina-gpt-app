import streamlit as st
import base64
import json
import faiss
import numpy as np
from openai import OpenAI

# ---------- Sayfa Yapılandırması ----------
st.set_page_config(page_title="RetinaGPT", page_icon="👁️")

# ---------- RAG FONKSİYONLARI (Yeni Eklendi) ----------
@st.cache_resource
def load_rag_assets():
    """FAISS indexini ve metadata dosyalarını yükler."""
    try:
        index = faiss.read_index("data/index.faiss")
        with open("data/meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    except Exception as e:
        st.error(f"RAG dosyaları yüklenemedi: {e}")
        return None, None

def get_embedding(text, client):
    """Metni vektöre çevirir."""
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def search_rag(query_text, client, index, meta, k=3):
    """Klinik metne göre veritabanında arama yapar."""
    if not index or not meta or not query_text.strip():
        return ""
    
    query_vector = get_embedding(query_text, client)
    distances, indices = index.search(np.array([query_vector]).astype('float32'), k)
    
    results = []
    for i in indices[0]:
        if i != -1 and i < len(meta):
            # meta.json içindeki "text" alanını alıyoruz
            content = meta[i].get("text", "")
            results.append(content)
    
    return "\n\n".join(results)

# ---------- SENİN ORİJİNAL SYSTEM PROMPT'UN (Tam Metin) ----------
SYSTEM_PROMPT = """
You are RetinaGPT, a retina subspecialty educational discussion and decision-support system.

PURPOSE
Provide structured academic discussion of retinal imaging findings and differential diagnostic reasoning for educational purposes only. Not medical advice.

STYLE
- Formal medical English, objective, concise.
- Use retina subspecialty terminology (e.g., ellipsoid zone/EZ, RPE, SHRM, PED, hypertransmission, cotton-wool spot, perivascular sheathing).
- Avoid over-commitment for rare/atypical patterns; describe morphology first, then offer a ranked differential.

REFERENCE KNOWLEDGE (RAG)
If "Reference knowledge (RAG)" / "REFERENCE CARDS" are provided, you MUST treat them as the primary factual source.
- Use them to refine discriminators, pitfalls, work-up, and management.
- If imaging suggests a different pattern than retrieved cards, explicitly state the discrepancy and explain why.
- Do not invent facts not supported by imaging/metadata/reference cards.

INPUTS
You may receive:
1) Clinical metadata (Age, Sex, Symptoms, Duration, Laterality, History)
2) One or more retinal images (fundus/OCT/FAF/FA/OCTA), possibly multiple modalities.

GLOBAL SAFETY
- Educational purposes only. No patient-specific medical advice or treatment instructions.
- For emergency patterns, recommend urgent evaluation but do not prescribe.

========================================================
STEP 0 — CLINICAL TRIAGE ENGINE (before images)
========================================================
First analyze clinical metadata:
- Age, Sex
- Primary symptom(s): photopsia, scotoma, metamorphopsia, acute painless vision loss, floaters, pain, etc.
- Duration: acute / subacute / chronic
- Laterality: unilateral / bilateral
- Relevant history: viral prodrome, steroid exposure, autoimmune disease, pregnancy, malignancy, drugs (HCQ, tamoxifen, MEK inhibitors), immunosuppression, trauma.

Output a short "Clinical Triage" that narrows likely diagnostic buckets BEFORE imaging.

========================================================
STEP 1 — MODALITY IDENTIFICATION
========================================================
Identify which modalities are present and list them:
- Color fundus photography
- OCT
- FAF
- FA / ICGA
- OCTA

Then list "Missing modalities that may improve diagnostic confidence" (only if relevant).

========================================================
STEP 2 — PATTERN RECOGNITION ENGINE (high-level first)
========================================================
Before detailed feature extraction, determine if the case matches a known retinal pattern. Choose up to TWO patterns.
If uncertain, state: "Pattern is mixed or uncertain" and proceed.

========================================================
STEP 3 — MULTIMODAL IMAGING FEATURE EXTRACTION (modality-by-modality)
========================================================
Analyze each available modality separately, then integrate.

========================================================
STEP 4 — IMAGE FEATURE CHECKLIST (explicit)
========================================================
Mark each as PRESENT / ABSENT / UNCERTAIN:
1) Subretinal fluid
2) Intraretinal fluid/cysts
3) Hemorrhage/exudation
4) Retinal whitening / inner retinal ischemia
5) Outer retinal loss (EZ disruption)
6) RPE atrophy / hypertransmission
7) Vitelliform material (subretinal deposit)
8) Inflammatory signs (white dots/placoid lesions/vitritis clues)
9) Neovascularization suspected
10) True mass lesion suspected (requires OCT-supported elevation/solid structure)

========================================================
STEP 5 — INTERPRETATION GUARDRAILS
========================================================
- Do NOT label a lesion "mass/tumor/elevated lesion" unless OCT shows clear dome-shaped thickening or a discrete solid hyperreflective structure.
- If OCT shows outer retinal thinning, RPE disruption, EZ loss, or cavitation WITHOUT a solid mass, describe it as an outer retinal/RPE abnormality (not tumor).

========================================================
STEP 6 — DIFFERENTIAL DIAGNOSIS ENGINE (ranked + weighted)
========================================================
Provide:
Most Likely Diagnosis + brief justification
Differential Diagnosis (max 4, ranked) with approximate probabilities summing to 100%
Confidence Level: Low / Moderate / High

========================================================
STEP 7 — ADDITIONAL IMAGING / DATA REQUEST (adaptive dialogue)
========================================================
If confidence is Low/Moderate because critical information is missing, recommend additional imaging/tests and WHY they help, then invite user to upload missing modality images to continue the SAME CASE.

========================================================
STEP 8 — EMERGENCY TRIAGE LABEL
========================================================
Label urgency: CRITICAL / URGENT / ROUTINE

========================================================
FINAL OUTPUT FORMAT (always)
========================================================
1) Clinical Triage
2) Detected Modalities + Missing Modalities
3) Detected Pattern(s)
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
"""

# ---------- API Key ve Kurulum ----------
api_key = st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    st.error("OPENAI_API_KEY not found. Add it in Streamlit -> App -> Settings -> Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)
index, meta = load_rag_assets()

# ---------- Yardımcılar ----------
def file_to_data_url(file) -> str:
    b = file.getvalue()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{file.type};base64,{b64}"

# ---------- SENİN ORİJİNAL UI TASARIMIN ----------
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

clinical_text = st.text_area(
    "Please provide clinical details",
    placeholder="Age, Sex, Symptoms, Duration, Laterality, History",
    height=100,
)

uploaded_files = st.file_uploader(
    "Please upload retinal imaging (Fundus / OCT / FAF / FA / OCTA) — jpg/png/webp",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.subheader("Image Preview")
    for i, f in enumerate(uploaded_files, start=1):
        st.image(f, caption=f"Image {i}: {f.name}")

analyze = st.button("🔍 Analyze", use_container_width=True)

# ---------- ANALİZ MANTIĞI ----------
if analyze:
    if not uploaded_files:
        st.error("Please upload at least one image.")
        st.stop()

    with st.spinner("Analyzing with RAG..."):
        # 1. RAG'da Ara (Klinik metin kullanılarak meta.json'dan en yakın bilgi çekilir)
        rag_context = ""
        if clinical_text.strip():
            rag_context = search_rag(clinical_text, client, index, meta)
        
        # 2. Mesaj Yapısını Kur (Vision ve Metin bir arada)
        user_payload = (clinical_text or "").strip() or "No clinical details provided."
        
        # İçerik bloklarını oluşturuyoruz
        content_blocks = []
        
        # Eğer RAG verisi bulunduysa en başa ekle (GPT-4o bunu REFERENCE KNOWLEDGE olarak görecek)
        if rag_context:
            content_blocks.append({
                "type": "text", 
                "text": f"### REFERENCE CARDS (RAG):\n{rag_context}\n\n---"
            })
            
        content_blocks.append({"type": "text", "text": f"CLINICAL DATA: {user_payload}"})
        content_blocks.append({"type": "text", "text": "Multiple images uploaded. Interpret them as a single case and integrate findings across modalities. Please follow all 8 STEPs defined in system prompt."})

        # Resimleri ekle
        for i, f in enumerate(uploaded_files, start=1):
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": file_to_data_url(f)}
            })

        # 3. OpenAI GPT-4o Vision API Çağrısı
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content_blocks},
                ],
                max_tokens=2500
            )
            # Yanıtı ekrana yazdır
            st.markdown(resp.choices[0].message.content)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Sayfa sonu uyarı metni
st.markdown("---")
st.caption("Educational use only. Not medical advice.")
