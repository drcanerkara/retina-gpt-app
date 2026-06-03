import os
import json
import base64
import re
import time
import datetime
import streamlit as st

from openai import OpenAI
from google import genai
try:
    import gspread
    from google.oauth2.service_account import Credentials
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False


# =========================
# Page
# =========================
st.set_page_config(page_title="RetinaGPT", page_icon="👁️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem;
    max-width: 860px;
}

/* ── Header ── */
.rgpt-header {
    background: linear-gradient(135deg, #0a1628 0%, #0d2240 50%, #0a1e38 100%);
    border-radius: 0 0 24px 24px;
    padding: 28px 32px 24px;
    margin: -1rem -1rem 1.5rem -1rem;
    position: relative;
    overflow: hidden;
}
.rgpt-header::before {
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.rgpt-header::after {
    content: "";
    position: absolute;
    bottom: -20px; left: 30%;
    width: 300px; height: 80px;
    background: radial-gradient(ellipse, rgba(99,179,237,0.08) 0%, transparent 70%);
}
.rgpt-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #f0f9ff;
    letter-spacing: -0.5px;
    margin: 0 0 4px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.rgpt-eye {
    width: 32px; height: 32px;
    background: radial-gradient(circle at 40% 40%, #38bdf8, #0369a1);
    border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 14px;
    box-shadow: 0 0 16px rgba(56,189,248,0.4);
}
.rgpt-subtitle {
    color: #7dd3fc;
    font-size: 0.82rem;
    font-weight: 400;
    margin: 0 0 14px 0;
    letter-spacing: 0.2px;
}
.rgpt-badges {
    display: flex; gap: 8px; flex-wrap: wrap;
}
.rgpt-badge {
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.25);
    color: #7dd3fc;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.3px;
    font-family: 'DM Mono', monospace;
}

/* ── Section cards ── */
.rgpt-section {
    background: #ffffff;
    border: 1px solid #e8edf4;
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.rgpt-section-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #64748b;
    margin: 0 0 14px 0;
    display: flex; align-items: center; gap: 7px;
}
.rgpt-section-title span {
    width: 3px; height: 14px;
    background: #38bdf8;
    border-radius: 2px;
    display: inline-block;
}
.rgpt-divider {
    height: 1px;
    background: #f1f5f9;
    margin: 14px 0;
}

/* ── Image modality tags ── */
.img-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    background: #f8fafc;
    border: 1px solid #e8edf4;
    border-radius: 10px;
    margin-bottom: 8px;
}
.img-thumb {
    width: 48px; height: 48px;
    object-fit: cover;
    border-radius: 6px;
    flex-shrink: 0;
}
.img-name {
    font-size: 0.8rem;
    color: #374151;
    font-weight: 500;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: 'DM Mono', monospace;
}
.modality-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 4px;
}

/* ── Confidence badges ── */
.conf-high   { background:#dcfce7; color:#15803d; border:1px solid #bbf7d0; }
.conf-mod    { background:#fef9c3; color:#a16207; border:1px solid #fef08a; }
.conf-low    { background:#fee2e2; color:#b91c1c; border:1px solid #fecaca; }
.pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 6px;
}
.pill-muted {
    background: #f1f5f9;
    color: #64748b;
    border: 1px solid #e2e8f0;
}

/* ── Report box ── */
.report-box {
    border-left: 3px solid #38bdf8;
    padding: 14px 16px;
    background: #f8faff;
    border-radius: 0 10px 10px 0;
    margin-top: 10px;
}

/* ── Agree badges ── */
.agree-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.agree-yes { background:#dcfce7; color:#15803d; }
.agree-no  { background:#fee2e2; color:#b91c1c; }

/* ── Debate round cards ── */
.debate-card {
    background: #f8fafc;
    border: 1px solid #e8edf4;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
}
.debate-model-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 4px;
}

/* ── Analyze button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    padding: 12px !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 14px rgba(3,105,161,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(3,105,161,0.45) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    border-radius: 10px !important;
    border-color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox > div > div {
    border-radius: 10px !important;
}

/* ── Small note ── */
.small-note { font-size: 0.78rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown('''
<div class="rgpt-header">
  <div class="rgpt-title">
    <div class="rgpt-eye">👁</div>
    RetinaGPT
  </div>
  <div class="rgpt-subtitle">Multi-agent deliberative retinal diagnostic reasoning</div>
  <div class="rgpt-badges">
    <span class="rgpt-badge">GPT-4o Vision</span>
    <span class="rgpt-badge">Gemini 2.5 Flash</span>
    <span class="rgpt-badge">3-Round Debate</span>
    <span class="rgpt-badge">Arbiter Synthesis</span>
  </div>
</div>
''', unsafe_allow_html=True)


# =========================
# Keys
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
SHEETS_ID      = os.getenv("SHEETS_ID")      or st.secrets.get("SHEETS_ID", None)

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY.")
    st.stop()

if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY.")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ── Sheets connection test (debug sidebar) ───────────────────────────────────
if st.sidebar.button("Test Sheets Connection"):
    with st.sidebar:
        try:
            if not SHEETS_AVAILABLE:
                st.error("gspread not available")
            elif not SHEETS_ID:
                st.error(f"SHEETS_ID missing. Value: {SHEETS_ID!r}")
            elif "gcp_service_account" not in st.secrets:
                st.error("gcp_service_account not in secrets")
            else:
                creds_dict = dict(st.secrets["gcp_service_account"])
                st.write("Keys:", list(creds_dict.keys()))
                scopes = ["https://www.googleapis.com/auth/spreadsheets",
                          "https://www.googleapis.com/auth/drive"]
                creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
                gc = gspread.authorize(creds)
                sh = gc.open_by_key(SHEETS_ID)
                st.success(f"Connected! Sheet: {sh.title}")
        except Exception as e:
            st.error(f"Error: {str(e)}")


# =========================
# Google Sheets
# =========================
SHEETS_COLUMNS = [
    # ── 1. Study identifiers ───────────────────────────────────────
    "StudyCaseID",          # P001-M1-K1
    "PatientCode",          # P001
    "Eye",                  # OD / OS
    "M_Level",              # M1 / M2 / M3
    "K_Level",              # K0 / K1 / K2 / K3
    # ── 2. Reference diagnosis (researcher fills BEFORE analysis) ──
    "DiagnosisCategory",    # VA / MK / KD / UV / CR / TU
    "PrimaryDiagnosis",     # ref_diagnosis — researcher fills
    "Expert1Diagnosis",     # researcher fills
    "Expert2Diagnosis",     # researcher fills
    "AdjudicatorDiagnosis", # if experts disagree
    "ConsensusStatus",      # AGREE / DISAGREE / ADJUDICATED
    "FinalReferenceDiagnosis",  # final ground truth
    # ── 3. Clinical input (assistant fills via form) ───────────────
    "Age","Sex","InvolvementPattern",
    "VisualAcuity","PrimarySymptoms","Duration",
    "SystemicDiseases","OcularHistory","Medications","FamilyHistory","FreeText",
    "UploaderID",           # A1 / A2 / A3
    # ── 4. Imaging (auto-detected) ─────────────────────────────────
    "ImageSetID",
    "Fundus_File","OCT_File","FFA_File","FAF_File",
    "OCTA_File","ICGA_File","Widefield_File",
    # ── 5. AI outputs (auto-filled after analysis) ─────────────────
    "ArmA_Top1","ArmA_Top2","ArmA_Top3","ArmA_Confidence",
    "ArmB_Top1","ArmB_Top2","ArmB_Top3","ArmB_Confidence",
    "ArmC_Agreement",
    "ArmD_Top1","ArmD_Top2","ArmD_Top3","ArmD_Confidence",
    # ── 6. Debate metrics (auto-filled) ────────────────────────────
    "AgreementStatus_R1",
    "AgreementStatus_R3",
    "DebateChangedDiagnosis",
    "SpecialistA_RevisionType",
    "SpecialistB_RevisionType",
    "CritiqueFailed",
    "NeedsHumanReview",
    # ── 7. Scoring (researcher fills AFTER analysis) ───────────────
    "CorrectTop1_ArmA","CorrectTop3_ArmA","CorrectCategory_ArmA",
    "CorrectTop1_ArmB","CorrectTop3_ArmB","CorrectCategory_ArmB",
    "CorrectTop1_ArmD","CorrectTop3_ArmD","CorrectCategory_ArmD",
    "GraderNotes",
    # ── 8. Run metadata (auto-filled) ──────────────────────────────
    "AnalysisDate",
    "Model","ModelFingerprint",
    "PromptVersion","AnalysisLanguage",
    "TemperatureOpenAI","TemperatureGemini","SeedOpenAI",
    "RunID",
    "ClinicalInput_Text",   # full clinical summary sent to model
]

def get_sheets_client():
    """Initialize gspread client using service account from Streamlit secrets."""
    if not SHEETS_AVAILABLE or not SHEETS_ID:
        st.session_state.sheets_error = "SHEETS_AVAILABLE or SHEETS_ID missing"
        return None
    try:
        if "gcp_service_account" not in st.secrets:
            st.session_state.sheets_error = "gcp_service_account not in secrets"
            return None
        creds_dict = dict(st.secrets["gcp_service_account"])
        if not creds_dict:
            st.session_state.sheets_error = "gcp_service_account is empty"
            return None
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(creds)
        return gc
    except Exception as e:
        st.session_state.sheets_error = f"get_sheets_client error: {str(e)}"
        return None


def ensure_header(ws):
    """Write header row if sheet is empty."""
    try:
        if ws.cell(1, 1).value != "timestamp":
            ws.insert_row(SHEETS_COLUMNS, 1)
    except Exception:
        pass


def log_to_sheets(export_data: dict):
    """Append one row to the Google Sheet. Silent fail on error."""
    try:
        gc = get_sheets_client()
        if not gc or not SHEETS_ID:
            return False
        sh = gc.open_by_key(SHEETS_ID)
        ws = sh.sheet1
        ensure_header(ws)

        arm_a  = export_data.get("arm_a_gpt4o_solo") or {}
        arm_b  = export_data.get("arm_b_gemini_solo") or {}
        arm_c  = export_data.get("arm_c_parallel_no_debate") or {}
        arm_d  = export_data.get("arm_d_debate_full") or {}
        debate = (arm_d.get("debate_transcript") or {})
        conv   = debate.get("convergence") or {}
        r2     = debate.get("round_2") or {}
        oa_c   = r2.get("openai_critique_of_gemini") or {}
        gem_c  = r2.get("gemini_critique_of_openai") or {}

        # Modality detection from clinical text
        clin = export_data.get("clinical","")
        mods_raw = clin.split("Imaging modalities provided:")[-1] if "Imaging modalities provided:" in clin else ""
        mods = [m.strip() for m in mods_raw.split("\n") if ":" in m]
        mod_values = [m.split(":")[-1].strip() for m in mods]

        def has_mod(keyword):
            kw = keyword.lower()
            for m in mod_values:
                ml = m.lower()
                # Exact match for short keywords to avoid FA matching FAF etc.
                if kw == "fa":
                    if ml in ("fa","fluorescein angiography","ffa","fluorescein angiogram"):
                        return "TRUE"
                elif kw == "faf":
                    if "faf" in ml or "fundus autofluorescence" in ml:
                        return "TRUE"
                elif kw in ml:
                    return "TRUE"
            return "FALSE"

        # Parse patient_id and disease_code from additional_notes
        notes = st.session_state.get("additional_notes","")
        patient_id   = ""
        disease_code = ""
        for line in notes.split("\n"):
            if "Case ID:" in line:
                parts = line.replace("Case ID:","").strip().split("-")
                if len(parts) >= 1: patient_id   = parts[0].strip()
                if len(parts) >= 2: disease_code = parts[1].strip()

        row = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # ── Study identifiers ──────────────────────────────────────
            st.session_state.get("case_id_str", export_data.get("case_id","")),  # StudyCaseID
            patient_id,                                                            # PatientCode
            st.session_state.get("laterality",""),                                # Eye
            st.session_state.get("modality_set","M1"),                            # M_Level
            export_data.get("clinical_layer","K0"),                               # K_Level
            f"{st.session_state.get('modality_set','M1')}-{'+'.join([v[:4].replace(' ','') for v in mod_values[:3]])}",  # ImageSetID
            has_mod("Fundus"), has_mod("OCT"), has_mod("FA"),
            has_mod("FAF"), has_mod("OCTA"), has_mod("ICGA"), has_mod("Wide"),
            # ── Clinical input ──────────────────────────────────────────
            st.session_state.get("age",""),
            st.session_state.get("sex",""),
            st.session_state.get("involvement",""),
            st.session_state.get("visual_acuity",""),
            ", ".join(st.session_state.get("primary_symptom",[])) if isinstance(st.session_state.get("primary_symptom",[]),list) else st.session_state.get("primary_symptom",""),
            st.session_state.get("duration",""),
            ", ".join(st.session_state.get("systemic_diseases",[])) if isinstance(st.session_state.get("systemic_diseases",[]),list) else st.session_state.get("systemic_diseases",""),
            st.session_state.get("ocular_history",""),
            st.session_state.get("medications",""),
            st.session_state.get("family_history",""),
            st.session_state.get("additional_notes",""),
            export_data.get("clinical",""),                                        # ClinicalInput_Text
            "",                                                                    # UploaderID — manuel
            # ── Run metadata ────────────────────────────────────────────
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),                # AnalysisDate
            st.session_state.get("openai_model_used", OPENAI_VISION_MODEL),       # Model
            st.session_state.get("openai_fingerprint",""),                        # ModelFingerprint
            PROMPT_VERSION,                                                        # PromptVersion
            ANALYSIS_LANGUAGE,                                                    # AnalysisLanguage
            str(TEMPERATURE_OPENAI),                                              # TemperatureOpenAI
            str(TEMPERATURE_GEMINI),                                              # TemperatureGemini
            str(SEED_OPENAI),                                                     # SeedOpenAI
            f"{st.session_state.get('case_id_str','')}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",  # RunID
            # ── AI outputs ──────────────────────────────────────────────
            arm_a.get("top_diagnosis",""),
            (arm_a.get("top_differentials") or ["","",""])[0] if arm_a.get("top_differentials") else "",
            (arm_a.get("top_differentials") or ["","",""])[1] if len(arm_a.get("top_differentials") or []) > 1 else "",
            arm_a.get("confidence",""),
            arm_b.get("top_diagnosis",""),
            (arm_b.get("top_differentials") or ["","",""])[0] if arm_b.get("top_differentials") else "",
            (arm_b.get("top_differentials") or ["","",""])[1] if len(arm_b.get("top_differentials") or []) > 1 else "",
            arm_b.get("confidence",""),
            str(arm_c.get("agreement","")),
            arm_d.get("final_diagnosis", (arm_d.get("debate_transcript") or {}).get("round_3",{}).get("openai_final",{}).get("top_diagnosis","")),
            ((arm_d.get("debate_transcript") or {}).get("round_3",{}).get("openai_final",{}).get("top_differentials") or ["",""])[0],
            ((arm_d.get("debate_transcript") or {}).get("round_3",{}).get("openai_final",{}).get("top_differentials") or ["",""])[1] if len(((arm_d.get("debate_transcript") or {}).get("round_3",{}).get("openai_final",{}).get("top_differentials") or [])) > 1 else "",
            arm_d.get("confidence_label",""),
            # ── Debate metrics ───────────────────────────────────────────
            str(conv.get("agreed_r1","")),
            str(conv.get("agreed_r3","")),
            str(conv.get("debate_changed_outcome","")),
            oa_c.get("revision_type",""),
            gem_c.get("revision_type",""),
            str(oa_c.get("revision_type","") == "failed" or gem_c.get("revision_type","") == "failed"),
            str(st.session_state.get("high_uncertainty_case", False)),
            # ── Reference diagnosis — researcher fills ───────────────────
            disease_code,                                                          # DiagnosisCategory
            "","","","","","",                                                    # PrimaryDiagnosis..FinalReferenceDiagnosis
            # ── Scoring — researcher fills ───────────────────────────────
            "","","",  # ArmA
            "","","",  # ArmB
            "","","",  # ArmD
            "",        # GraderNotes
        ]

        ws.append_row(row, value_input_option="USER_ENTERED")
        return True
    except Exception as e:
        st.session_state.sheets_error = str(e)
        return False


# =========================
# Models
# =========================
OPENAI_VISION_MODEL  = "gpt-4o"
OPENAI_ARBITER_MODEL = "gpt-4o-mini"
GEMINI_VISION_MODEL  = "gemini-2.5-flash"

# ── Research reproducibility metadata ────────────────────────────────────
PROMPT_VERSION       = "v1.2"   # increment when any prompt changes
ANALYSIS_LANGUAGE    = "en"     # prompts and outputs are in English
TEMPERATURE_OPENAI   = 0        # deterministic
TEMPERATURE_GEMINI   = 0.0      # deterministic
SEED_OPENAI          = 42       # fixed seed for reproducibility


# =========================
# JSON Schemas
# =========================
VISION_JSON_SCHEMA = {
    "top_diagnosis": "string",
    "top_differentials": ["string", "string", "string"],
    "key_evidence": ["string", "string", "string"],
    "confidence": "LOW|MODERATE|HIGH",
    "requested_additional_info": ["string", "string", "string"],
}

CRITIQUE_JSON_SCHEMA = {
    "agree": "true|false",
    "critique": "string — specific imaging or clinical reasoning for agreement/disagreement",
    "revised_diagnosis": "string or null if unchanged",
    "revised_confidence": "LOW|MODERATE|HIGH",
    "revision_type": "evidence_based|sycophantic|maintained|failed",
}


# =========================
# Session state
# =========================
def ss_init():
    defaults = {
        "case_id": 1,
        "patient_number": 1,
        "case_id_str": "P001-M1-K0",
        "manual_k": "K0",
        "manual_m": "M1",
        "images": [],
        "final_report": "",
        "report_history": [],
        "agreement": None,
        "confidence_label": "",
        "confidence_icon": "",
        "chat_history": [],
        "analysis_done": False,
        "uploader_key": 1,
        "last_error_type": None,
        "last_error_raw": "",
        "additional_clinical_note": "",
        "add_uploader_key": 1,
        "additional_modality": "OCT",
        # structured clinical fields
        "age": None,
        "sex": "Select",
        "laterality": "Select",          # K1: analyzed eye
        "involvement": "Select",           # K2: unilateral/bilateral
        "visual_acuity": "",               # K2: raw Snellen
        "primary_symptom": [],             # K2: multiselect
        "duration": "",                    # K2: duration text
        "systemic_diseases": [],           # K3: multiselect
        "ocular_history": "",              # K3: free text
        "medications": "",                 # K3: free text
        "family_history": "",              # K3: free text
        "additional_notes": "",            # K3: free text
        "clinical": "",
        "clinical_layer": "K0",
        "modality_set": "M1",
        "sheets_logged": None,
        "high_uncertainty_case": False,
        "openai_model_used": "",
        "openai_fingerprint": "",
        "prompt_version": PROMPT_VERSION,
        # debate log — stores full transcript for research export
        "debate_log": None,
        # arm results for research comparison
        "arm_a_result": None,   # GPT-4o solo
        "arm_b_result": None,   # Gemini solo
        "arm_c_result": None,   # Parallel no-debate
        "arm_d_result": None,   # Full debate
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

ss_init()


def reset_condition():
    """Clear images and results — keep patient info and K/M selection."""
    # Keep: patient_number, manual_k, manual_m, all clinical fields
    for k in [
        "images","final_report","report_history","agreement",
        "confidence_label","confidence_icon",
        "clinical","debate_log","sheets_logged","high_uncertainty_case",
        "arm_a_result","arm_b_result","arm_c_result","arm_d_result",
        "analysis_done","openai_model_used","openai_fingerprint",
    ]:
        st.session_state[k] = [] if k in ("images","report_history") else None
    st.session_state.analysis_done = False
    st.session_state.uploader_key = st.session_state.get("uploader_key",0) + 1
    # Update case_id_str with current K and M
    _pid3 = f"P{st.session_state.get('patient_number',1):03d}"
    _m3   = st.session_state.get("manual_m","M1").strip()[:2]
    _k3   = st.session_state.get("manual_k","K0").strip()[:2]
    st.session_state.case_id_str = f"{_pid3}-{_m3}-{_k3}"


def reset_case():
    st.session_state.case_id += 1
    for k in [
        "images", "final_report", "report_history", "agreement",
        "confidence_label", "confidence_icon", "chat_history",
        "last_error_type", "last_error_raw", "additional_clinical_note",
        "additional_modality", "clinical", "clinical_layer", "modality_set", "sheets_logged", "high_uncertainty_case", "manual_k", "manual_m", "debate_log",
        "arm_a_result", "arm_b_result", "arm_c_result", "arm_d_result",
    ]:
        st.session_state[k] = [] if k in ("images", "report_history", "chat_history") else None
    st.session_state.analysis_done = False
    st.session_state.uploader_key += 1
    # Update case_id_str for new patient
    _pid2 = f"P{st.session_state.patient_number:03d}"
    st.session_state.case_id_str = f"{_pid2}-M1-K0"
    st.session_state.add_uploader_key += 1
    st.session_state.additional_modality = "OCT"
    st.session_state.confidence_label = ""
    st.session_state.confidence_icon = ""
    st.session_state.age = None
    st.session_state.sex = "Select"
    st.session_state.laterality = "Select"
    st.session_state.involvement = "Select"
    st.session_state.systemic_diseases = []
    st.session_state.primary_symptom = []
    for f in ("visual_acuity","duration","ocular_history","medications","family_history","additional_notes","clinical","additional_clinical_note"):
        st.session_state[f] = ""


# =========================
# Helpers
# =========================
def b64_data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def safe_json_extract(text: str):
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def to_jsonable(obj):
    if obj is None: return None
    if isinstance(obj, (str, int, float, bool)): return obj
    if isinstance(obj, (bytes, bytearray)): return f"<bytes:{len(obj)}>"
    if isinstance(obj, dict): return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [to_jsonable(x) for x in obj]
    for attr in ("model_dump", "dict"):
        if hasattr(obj, attr):
            try: return to_jsonable(getattr(obj, attr)())
            except Exception: pass
    if hasattr(obj, "__dict__"):
        try: return to_jsonable(vars(obj))
        except Exception: pass
    return str(obj)


def normalize_dx(dx: str) -> str:
    """Normalize diagnosis string: remove parenthetical abbreviations, lowercase."""
    if not dx: return ""
    dx = re.sub(r"\([^)]{1,10}\)", "", dx)   # strip (DME), (RVO) etc.
    return re.sub(r"[^a-z0-9]+", " ", dx.lower()).strip()


def dx_similarity(a: str, b: str) -> bool:
    """True if two diagnosis strings are clinically equivalent."""
    a = normalize_dx(a)
    b = normalize_dx(b)
    if not a or not b: return False
    if a == b: return True
    if a in b or b in a: return True
    a_words = set(a.split()) - {"with","and","or","the","of","in","a","an","to","due","type"}
    b_words = set(b.split()) - {"with","and","or","the","of","in","a","an","to","due","type"}
    if len(a_words) >= 2 and len(b_words) >= 2:
        if len(a_words & b_words) >= min(2, min(len(a_words), len(b_words))):
            return True
    return False


def normalize_confidence(value: str) -> str:
    v = str(value or "").strip().upper()
    return v if v in {"HIGH", "MODERATE", "LOW"} else "LOW"


def overlap_top2(a: dict, b: dict) -> bool:
    """Check if two model outputs agree on the primary or top differential diagnosis."""
    a1 = normalize_dx((a or {}).get("top_diagnosis", ""))
    b1 = normalize_dx((b or {}).get("top_diagnosis", ""))
    if a1 and b1 and dx_similarity(a1, b1):
        return True
    a2 = [normalize_dx(x) for x in ((a or {}).get("top_differentials") or [])[:2]]
    b2 = [normalize_dx(x) for x in ((b or {}).get("top_differentials") or [])[:2]]
    # Check if either top dx appears in the other's differentials
    if any(dx_similarity(a1, x) for x in b2): return True
    if any(dx_similarity(b1, x) for x in a2): return True
    # Check differential overlap
    for x in a2:
        for y in b2:
            if dx_similarity(x, y): return True
    return False


def derive_confidence_badge(gem_js: dict, oa_js: dict, agreement: bool):
    gc = normalize_confidence((gem_js or {}).get("confidence", "LOW"))
    oc = normalize_confidence((oa_js  or {}).get("confidence", "LOW"))
    if agreement:
        if gc == "HIGH" and oc == "HIGH":
            return "High", "🟢"
        return "Moderate", "🟡"
    return "Low", "🔴"


def uploads_to_images(files):
    return [{"name": f.name, "mime": f.type or "image/jpeg", "data": f.getvalue()} for f in (files or [])]


def merge_images(old, new):
    return list(old or []) + list(new or [])


def get_current_assessment_label():
    n = len(st.session_state.report_history)
    return "Initial assessment" if n == 0 else f"Updated assessment {n}"


def build_clinical_summary():
    lines = []
    # K0
    if st.session_state.age: lines.append(f"Age: {st.session_state.age}")
    if st.session_state.sex not in (None, "Select"): lines.append(f"Sex: {st.session_state.sex}")
    # K1
    if st.session_state.laterality not in (None, "Select", ""):
        lines.append(f"Analyzed eye: {st.session_state.laterality}")
    # K2
    if st.session_state.get("involvement") not in (None, "Select", ""):
        lines.append(f"Involvement pattern: {st.session_state.involvement}")
    va = (st.session_state.get("visual_acuity") or "").strip()
    if va: lines.append(f"Visual acuity (analyzed eye): {va}")
    symp_list = st.session_state.get("primary_symptom", [])
    if isinstance(symp_list, list) and symp_list:
        lines.append(f"Primary symptom(s): {', '.join(symp_list)}")
    elif isinstance(symp_list, str) and symp_list:
        lines.append(f"Primary symptom: {symp_list}")
    dur = (st.session_state.get("duration") or "").strip()
    if dur: lines.append(f"Duration: {dur}")
    # K3
    sys_list = st.session_state.get("systemic_diseases", [])
    if isinstance(sys_list, list) and sys_list:
        lines.append(f"Systemic diseases: {', '.join(sys_list)}")
    elif isinstance(sys_list, str) and sys_list:
        lines.append(f"Systemic diseases: {sys_list}")
    for label, key in [
        ("Ocular history", "ocular_history"),
        ("Medications", "medications"),
        ("Family history", "family_history"),
        ("Additional notes", "additional_notes"),
    ]:
        val = (st.session_state.get(key) or "").strip()
        if val: lines.append(f"{label}: {val}")
    if st.session_state.get("clinical_layer"):
        lines.append(f"Clinical knowledge layer: {st.session_state.clinical_layer}")
    return "\n".join(lines).strip()


# =========================
# ── ROUND 1: Independent vision calls ──
# =========================
def call_openai_vision(clinical_text: str, images):
    content = [{
        "type": "text",
        "text": (
            "You are Specialist A, an expert retina subspecialist. Carefully examine ALL provided retinal images "
            "(which may include fundus photography, OCT, FAF, FA, OCTA, ICGA, or wide-field imaging) "
            "together with the clinical information.\n\n"
            "CRITICAL RULES — THESE ARE MANDATORY, NOT OPTIONAL:\n"
            "1. You MUST ALWAYS provide a top_diagnosis. NEVER return 'UNCERTAIN', 'UNKNOWN', or an empty string.\n"
            "   This is a RESEARCH PROTOCOL — uncertain cases still require your best morphological impression.\n"
            "2. If you cannot be certain, give your BEST MORPHOLOGICAL GUESS with LOW confidence.\n"
            "   Example: if you see hard exudates, write 'Exudative maculopathy — etiology unclear' NOT 'UNCERTAIN'.\n"
            "3. You MUST list at least 2 top_differentials — DIFFERENT from top_diagnosis.\n"
            "4. You MUST list at least 3 key_evidence items describing specific findings you see in the images.\n"
            "5. Always describe: fluid type (SRF/IRF/CME), exudates, hemorrhage, drusen, "
            "RPE changes, ellipsoid zone, neovascularization, vascular changes.\n"
            "6. NEVER leave top_differentials or key_evidence as empty arrays [].\n"
            "7. NEVER repeat top_diagnosis inside top_differentials.\n\n"
            "Return STRICT JSON ONLY — no markdown, no explanation — matching this schema exactly:\n"
            f"{json.dumps(VISION_JSON_SCHEMA, ensure_ascii=False)}\n\n"
            f"Clinical info:\n{clinical_text or '(none provided)'}"
        ),
    }]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": b64_data_url(img["mime"], img["data"])}})
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=TEMPERATURE_OPENAI,
            seed=SEED_OPENAI,
            max_tokens=1000,
        )
        raw = resp.choices[0].message.content or ""
        # Store model metadata for research reproducibility
        st.session_state.openai_model_used = resp.model
        st.session_state.openai_fingerprint = getattr(resp, "system_fingerprint", "")
        parsed = safe_json_extract(raw)
        # Clean up: remove top_diagnosis from differentials if present
        if parsed and parsed.get("top_differentials"):
            top_dx = normalize_dx(parsed.get("top_diagnosis",""))
            parsed["top_differentials"] = [
                d for d in parsed["top_differentials"]
                if not dx_similarity(normalize_dx(d), top_dx)
            ]
        # Fallback: if model still returned empty/uncertain, retry once with stronger nudge
        if not parsed or parsed.get("top_diagnosis", "").upper() in ("UNCERTAIN", "", "UNKNOWN"):
            nudge_content = content.copy()
            nudge_content[0] = {
                "type": "text",
                "text": (
                    "MANDATORY RETRY — You must provide a diagnosis now.\n\n"
                    "Your previous response was empty or 'UNCERTAIN'. This is NOT acceptable in this research protocol.\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Look at the image(s) carefully. Identify ALL visible abnormalities.\n"
                    "2. Even with no clinical data, you can see: hard exudates, soft exudates, hemorrhages, "
                    "fluid, drusen, pigment changes, vascular abnormalities, membrane, etc.\n"
                    "3. Based ONLY on morphological findings, name the most likely retinal condition.\n"
                    "4. If truly ambiguous, use descriptive diagnoses like:\n"
                    "   - 'Exudative maculopathy, etiology unclear'\n"
                    "   - 'Retinal vascular disease with macular involvement'\n"
                    "   - 'Macular edema, cause undetermined'\n"
                    "   NEVER use 'UNCERTAIN' or leave fields empty.\n\n"
                    "Return STRICT JSON ONLY matching this schema:\n"
                    f"{json.dumps(VISION_JSON_SCHEMA, ensure_ascii=False)}\n\n"
                    f"Clinical info:\n{clinical_text or '(none provided — base diagnosis on image morphology only)'}"
                ),
            }
            resp2 = openai_client.chat.completions.create(
                model=OPENAI_VISION_MODEL,
                messages=[{"role": "user", "content": nudge_content}],
                temperature=0.3,
                seed=43,
                max_tokens=1000,
            )
            raw = resp2.choices[0].message.content or ""
            parsed = safe_json_extract(raw)
        return raw, parsed
    except Exception as e:
        return f"OpenAI vision error: {e}", None


def call_gemini_vision(clinical_text: str, images, max_retries: int = 3):
    parts = [{
        "text": (
            "You are Specialist B, a retina specialist. Analyze the provided retinal images and clinical info.\n"
            "Return STRICT JSON ONLY matching this schema:\n"
            f"{json.dumps(VISION_JSON_SCHEMA, ensure_ascii=False)}\n\n"
            f"Clinical info:\n{clinical_text or '(none provided)'}"
        )
    }]
    for img in images:
        parts.append({"inline_data": {"mime_type": img["mime"], "data": base64.b64encode(img["data"]).decode()}})

    last_error = ""
    for attempt in range(max_retries):
        try:
            resp = gemini_client.models.generate_content(
                model=GEMINI_VISION_MODEL,
                contents=[{"role": "user", "parts": parts}],
                config={"temperature": TEMPERATURE_GEMINI, "response_mime_type": "application/json"},
            )
            raw = getattr(resp, "text", None) or ""
            parsed = safe_json_extract(raw)
            if parsed:
                return raw, parsed, None
            last_error = raw or "Gemini returned empty or invalid JSON."
        except Exception as e:
            last_error = f"Gemini vision error: {e}"
            busy = any(x in last_error for x in ["503", "UNAVAILABLE", "high demand", "busy"])
            if busy and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return last_error, None, "busy" if busy else "generic"
    return last_error, None, "busy"


# =========================
# ── ROUND 2: Cross-critique ──
# =========================
def call_openai_critique(clinical_text: str, images, gemini_opinion: dict, oa_r1: dict = None):
    """GPT-4o reads Gemini's Round 1 output and critiques it."""
    own_context = (
        f"\nYour own Round 1 assessment was:\n{json.dumps(oa_r1, ensure_ascii=False, indent=2)}\n"
        if oa_r1 else ""
    )
    crit_content = [{
        "type": "text",
        "text": (
            "You are Specialist A, an expert retina subspecialist conducting a structured peer review.\n"
            "Re-examine ALL retinal images carefully, then evaluate the anonymous peer assessment below.\n\n"
            f"Peer assessment (anonymous):\n{json.dumps(gemini_opinion, ensure_ascii=False, indent=2)}\n"
            f"{own_context}\n"
            "TASK:\n"
            "1. Look at the images again with fresh eyes.\n"
            "2. Identify specific morphological features: fluid type (SRF/IRF/CME), exudate "
            "pattern, hemorrhage distribution, vascular changes, RPE status, drusen, etc.\n"
            "3. Decide whether you AGREE or DISAGREE with your colleague's top diagnosis.\n"
            "4. Write a specific evidence-based critique referencing actual image findings.\n"
            "5. If you revise your diagnosis, cite the EXACT imaging feature that changed your mind.\n"
            "6. Do NOT simply agree because your colleague seems confident — sycophancy is a bias.\n\n"
            "Return STRICT JSON ONLY matching this schema:\n"
            f"{json.dumps(CRITIQUE_JSON_SCHEMA, ensure_ascii=False)}\n\n"
            "revision_type rules:\n"
            "- 'evidence_based': diagnosis changes AND you cite a specific imaging finding\n"
            "- 'sycophantic': diagnosis changes WITHOUT citing a new specific imaging finding\n"
            "- 'maintained': you keep your view with explicit reasoning from the images\n\n"
            f"Clinical info:\n{clinical_text or '(none provided)'}"
        ),
    }]
    for img in images:
        crit_content.append({"type": "image_url", "image_url": {"url": b64_data_url(img["mime"], img["data"])}})
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[{"role": "user", "content": crit_content}],
            temperature=0,
            seed=42,
            max_tokens=1000,
        )
        raw = resp.choices[0].message.content or ""
        return raw, safe_json_extract(raw)
    except Exception as e:
        return f"OpenAI critique error: {e}", None


def call_gemini_critique(clinical_text: str, images, openai_opinion: dict, gem_r1: dict = None, max_retries: int = 3):
    """Gemini reads GPT-4o's Round 1 output and critiques it."""
    own_context = (
        f"\nYour own Round 1 assessment was:\n{json.dumps(gem_r1, ensure_ascii=False, indent=2)}\n"
        if gem_r1 else ""
    )
    parts = [{
        "text": (
            "You are Specialist B, an expert retina subspecialist conducting a structured peer review.\n"
            "Re-examine ALL retinal images carefully, then evaluate the anonymous peer assessment below.\n\n"
            f"Peer assessment (anonymous):\n{json.dumps(openai_opinion, ensure_ascii=False, indent=2)}\n"
            f"{own_context}\n"
            "TASK:\n"
            "1. Look at the images again with fresh eyes.\n"
            "2. Identify specific morphological features: fluid type (SRF/IRF/CME), exudate "
            "pattern, hemorrhage distribution, vascular changes, RPE status, drusen, etc.\n"
            "3. Decide whether you AGREE or DISAGREE with your colleague's top diagnosis.\n"
            "4. Write a specific evidence-based critique referencing actual image findings.\n"
            "5. If you revise your diagnosis, cite the EXACT imaging feature that changed your mind.\n"
            "6. Do NOT simply agree because your colleague seems confident — sycophancy is a bias.\n\n"
            "Return STRICT JSON ONLY matching this schema:\n"
            f"{json.dumps(CRITIQUE_JSON_SCHEMA, ensure_ascii=False)}\n\n"
            "revision_type rules:\n"
            "- 'evidence_based': diagnosis changes AND you cite a specific imaging finding\n"
            "- 'sycophantic': diagnosis changes WITHOUT citing a new specific imaging finding\n"
            "- 'maintained': you keep your view with explicit reasoning from the images\n\n"
            f"Clinical info:\n{clinical_text or '(none provided)'}"
        )
    }]
    for img in images:
        parts.append({"inline_data": {"mime_type": img["mime"], "data": base64.b64encode(img["data"]).decode()}})

    last_error = ""
    for attempt in range(max_retries):
        try:
            resp = gemini_client.models.generate_content(
                model=GEMINI_VISION_MODEL,
                contents=[{"role": "user", "parts": parts}],
                config={"temperature": TEMPERATURE_GEMINI, "response_mime_type": "application/json"},
            )
            raw = getattr(resp, "text", None) or ""
            parsed = safe_json_extract(raw)

            # Validate: critique must have a non-empty critique field
            if parsed and parsed.get("critique","").strip() in ("", "No critique available.", "N/A"):
                parsed = None
                last_error = "Gemini critique was empty or placeholder — retrying."

            if parsed:
                return raw, parsed, None

            last_error = last_error or raw or "Gemini critique returned empty."
            # Short wait before retry for empty response
            if attempt < max_retries - 1:
                time.sleep(1.5)

        except Exception as e:
            last_error = f"Gemini critique error: {e}"
            busy = any(x in last_error for x in ["503", "UNAVAILABLE", "high demand", "busy"])
            if busy and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return last_error, None, "busy" if busy else "generic"

    # All retries exhausted — return best-effort fallback
    return last_error, None, "generic"


# =========================
# ── ROUND 3: Revision ──
# =========================
def _revision_prompt(own_r1: dict, critique_received: dict, clinical_text: str) -> str:
    return (
        "You are a retina subspecialist. Below is your original assessment (Round 1) "
        "and the peer critique you received (Round 2).\n\n"
        f"Your Round 1 assessment:\n{json.dumps(own_r1, ensure_ascii=False, indent=2)}\n\n"
        f"Peer critique:\n{json.dumps(critique_received, ensure_ascii=False, indent=2)}\n\n"
        "Review the images again and produce your FINAL revised assessment.\n"
        "If you change your diagnosis, explain exactly which imaging or clinical finding prompted the change.\n"
        "If you maintain your original diagnosis, explain why the critique did not change your view.\n\n"
        "Return STRICT JSON ONLY matching this schema:\n"
        f"{json.dumps(VISION_JSON_SCHEMA, ensure_ascii=False)}\n\n"
        f"Clinical info:\n{clinical_text or '(none provided)'}"
    )


def call_openai_revision(clinical_text: str, images, oa_r1: dict, gem_critique: dict):
    content = [{"type": "text", "text": _revision_prompt(oa_r1, gem_critique, clinical_text)}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": b64_data_url(img["mime"], img["data"])}})
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=TEMPERATURE_OPENAI,
            seed=SEED_OPENAI,
            max_tokens=1000,
        )
        raw = resp.choices[0].message.content or ""
        # Store model metadata for research reproducibility
        st.session_state.openai_model_used = resp.model
        st.session_state.openai_fingerprint = getattr(resp, "system_fingerprint", "")
        return raw, safe_json_extract(raw)
    except Exception as e:
        return f"OpenAI revision error: {e}", None


def call_gemini_revision(clinical_text: str, images, gem_r1: dict, oa_critique: dict, max_retries: int = 3):
    parts = [{"text": _revision_prompt(gem_r1, oa_critique, clinical_text)}]
    for img in images:
        parts.append({"inline_data": {"mime_type": img["mime"], "data": base64.b64encode(img["data"]).decode()}})

    last_error = ""
    for attempt in range(max_retries):
        try:
            resp = gemini_client.models.generate_content(
                model=GEMINI_VISION_MODEL,
                contents=[{"role": "user", "parts": parts}],
                config={"temperature": TEMPERATURE_GEMINI, "response_mime_type": "application/json"},
            )
            raw = getattr(resp, "text", None) or ""
            parsed = safe_json_extract(raw)
            if parsed:
                return raw, parsed, None
            last_error = raw or "Gemini revision returned empty."
        except Exception as e:
            last_error = f"Gemini revision error: {e}"
            busy = any(x in last_error for x in ["503", "UNAVAILABLE", "high demand", "busy"])
            if busy and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return last_error, None, "busy" if busy else "generic"
    return last_error, None, "busy"


# =========================
# ── SYNTHESIS: Arbiter report ──
# =========================
def build_final_report(clinical_text: str, debate_transcript: dict, is_update: bool):
    payload = to_jsonable({
        "clinical": clinical_text,
        "debate_transcript": debate_transcript,
        "is_update": is_update,
    })

    system = """
You are RetinaGPT Arbiter, a retina subspecialty clinical reasoning assistant.
You receive the FULL debate transcript between two anonymous AI retina specialists (Specialist A and Specialist B, 3 rounds).
Generate ONE final educational report synthesizing the debate.

FORMAT (Markdown)

**Most likely diagnosis**
- State the most probable diagnosis in **bold**
- Add confidence level (High / Moderate / Low)
- State whether the two specialists reached consensus or maintained disagreement after debate
- Brief justification referencing key debate arguments

**Diagnostic agreement status**
- State clearly: CONSENSUS / PARTIAL AGREEMENT / DISAGREEMENT
- If DISAGREEMENT or PARTIAL AGREEMENT: describe the specific point of divergence
- Example: "Specialist A favored DME; Specialist B favored CRVO — divergence on vascular etiology"

**Uncertainty assessment**
- Summarize overall diagnostic certainty
- Note if any alternative diagnosis remains plausible after debate
- Rate clinical uncertainty: LOW / MODERATE / HIGH

**Key imaging findings**
- Bullet list of the most relevant retinal imaging features discussed during debate
- Focus on morphology (RPE changes, ellipsoid zone status, SRF, IRF, PED, hemorrhage, vascular changes)

**Differential diagnosis**
- 2-4 most relevant alternatives
- Note which were debated and why each was accepted or rejected

**Management considerations**
- Evidence-based retina management
- Mention observation vs treatment when relevant
- Include follow-up strategy

**Suggested additional imaging**
- Recommend only modalities NOT already present in the case
- Examples: OCTA, FA, ICGA, FAF, wide-field imaging

**Debate summary**
- 2-3 sentences: what was agreed in Round 1, what was contested, how Round 3 resolved it
- Note revision types: evidence-based revision, maintained position, or sycophantic change

**⚠ Expert referral flag**
- If DISAGREEMENT persists after Round 3 AND/OR uncertainty is HIGH: output exactly this line:
  HIGH_UNCERTAINTY_CASE: true
- If consensus reached AND uncertainty is LOW or MODERATE: output exactly this line:
  HIGH_UNCERTAINTY_CASE: false

STYLE
- Retina subspecialty terminology
- Concise but clinically meaningful
- Morphology-first reasoning
- Do NOT mention model names (GPT, Gemini, Specialist A, Specialist B) in the clinical sections
- If this is an updated assessment, explicitly note how new imaging changed the working diagnosis
"""
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_ARBITER_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            temperature=0,
        )
        report_text = resp.choices[0].message.content or ""
        # Parse expert review flag
        flag_match = re.search(r'HIGH_UNCERTAINTY_CASE:[ \t]*(true|false)', report_text, re.IGNORECASE)
        if flag_match:
            st.session_state.high_uncertainty_case = flag_match.group(1).lower() == 'true'
        else:
            # Fallback: check agreement from debate transcript
            st.session_state.high_uncertainty_case = False
        return report_text
    except Exception as e:
        return f"Final report generation failed: {e}"


# =========================
# Chat continuation
# =========================
def chat_reply(user_text: str):
    context_pack = to_jsonable({
        "case": st.session_state.case_id,
        "clinical": st.session_state.clinical,
        "final_report": st.session_state.final_report,
        "report_history": st.session_state.report_history,
        "confidence_label": st.session_state.confidence_label,
    })
    system = """
You are RetinaGPT (educational). Continue discussion for the SAME case.
- Use the latest final report as the baseline.
- Use the structured clinical information as part of the case context.
- If the user asks what to upload next, suggest the single most useful modality and what to look for.
- Keep outputs concise and structured.
- Use retina subspecialty terminology.
"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "CASE CONTEXT:\n" + json.dumps(context_pack, ensure_ascii=False, indent=2)},
    ]
    for m in st.session_state.chat_history[-20:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_text})
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_ARBITER_MODEL, messages=messages, temperature=0,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"Chat reply failed: {e}"


# =========================
# ── MAIN ANALYSIS PIPELINE ──
# =========================
def run_analysis():
    images = st.session_state.images
    clin   = st.session_state.clinical

    st.session_state.last_error_type = None
    st.session_state.last_error_raw  = ""

    # ── Round 1: Independent ──────────────────────────────────────────────
    with st.spinner("🔬 Round 1 — Independent analysis..."):
        gem_raw, gem_r1, gem_err = call_gemini_vision(clin, images)
        oa_raw,  oa_r1           = call_openai_vision(clin, images)

    if not gem_r1:
        st.session_state.last_error_type = gem_err or "generic"
        st.session_state.last_error_raw  = gem_raw or ""
        st.session_state.analysis_done   = False
        return

    if not oa_r1:
        oa_r1 = {"top_diagnosis": "UNCERTAIN", "top_differentials": [],
                 "key_evidence": [], "confidence": "LOW", "requested_additional_info": []}

    # Save arm A and B (solo results)
    st.session_state.arm_a_result = oa_r1
    st.session_state.arm_b_result = gem_r1

    # Arm C: parallel no-debate agreement
    agree_c = overlap_top2(gem_r1, oa_r1)
    st.session_state.arm_c_result = {
        "openai_r1": oa_r1, "gemini_r1": gem_r1, "agreement": agree_c
    }

    # ── Round 2: Cross-critique ───────────────────────────────────────────
    with st.spinner("💬 Round 2 — Cross-critique..."):
        oa_crit_raw, oa_crit = call_openai_critique(clin, images, gem_r1, oa_r1=oa_r1)
        gem_crit_raw, gem_crit, _ = call_gemini_critique(clin, images, oa_r1, gem_r1=gem_r1)

    if not oa_crit:
        oa_crit = {
            "agree": True,
            "critique": "Critique generation failed — model did not return valid output.",
            "revised_diagnosis": None,
            "revised_confidence": oa_r1.get("confidence", "LOW"),
            "revision_type": "failed",
        }
    if not gem_crit:
        gem_crit = {
            "agree": True,
            "critique": "Critique generation failed — model did not return valid output.",
            "revised_diagnosis": None,
            "revised_confidence": gem_r1.get("confidence", "LOW"),
            "revision_type": "failed",
        }

    # ── Round 3: Revision ─────────────────────────────────────────────────
    with st.spinner("🔄 Round 3 — Revision..."):
        oa_rev_raw, oa_r3  = call_openai_revision(clin, images, oa_r1, gem_crit)
        gem_rev_raw, gem_r3, _ = call_gemini_revision(clin, images, gem_r1, oa_crit)

    if not oa_r3:  oa_r3  = oa_r1
    if not gem_r3: gem_r3 = gem_r1

    # ── Final agreement after debate ──────────────────────────────────────
    agree_d = overlap_top2(gem_r3, oa_r3)
    confidence_label, confidence_icon = derive_confidence_badge(gem_r3, oa_r3, agree_d)

    # ── Build debate transcript for arbiter ───────────────────────────────
    debate_transcript = {
        "round_1": {"openai": oa_r1,   "gemini": gem_r1},
        "round_2": {"openai_critique_of_gemini": oa_crit,
                    "gemini_critique_of_openai": gem_crit},
        "round_3": {"openai_final": oa_r3, "gemini_final": gem_r3},
        "convergence": {
            "agreed_r1": overlap_top2(gem_r1, oa_r1),
            "agreed_r3": agree_d,
            "debate_changed_outcome": overlap_top2(gem_r1, oa_r1) != agree_d,
        },
    }

    # ── Synthesis ─────────────────────────────────────────────────────────
    with st.spinner("📝 Synthesizing final report..."):
        is_update = len(st.session_state.report_history) > 0
        report = build_final_report(clin, debate_transcript, is_update)

    # ── Store arm D ───────────────────────────────────────────────────────
    st.session_state.arm_d_result = {
        "debate_transcript": debate_transcript,
        "final_report": report,
        "agreement": agree_d,
        "confidence_label": confidence_label,
    }

    # ── Update session state ──────────────────────────────────────────────
    label = get_current_assessment_label()
    st.session_state.debate_log       = debate_transcript
    st.session_state.final_report     = report
    st.session_state.agreement        = agree_d
    st.session_state.confidence_label = confidence_label
    st.session_state.confidence_icon  = confidence_icon
    st.session_state.analysis_done    = True
    st.session_state.report_history.append({
        "label": label,
        "report": report,
        "confidence_label": confidence_label,
        "confidence_icon": confidence_icon,
        "debate_log": debate_transcript,
    })

    # ── Auto-detect modality set (M1/M2/M3) ──────────────────────────────
    clin_text = st.session_state.clinical
    extra_mods = 0
    for kw in ["FAF","FA","OCTA","ICGA","Wide-field","wide field"]:
        if kw.lower() in clin_text.lower():
            extra_mods += 1
    if extra_mods == 0:
        st.session_state.modality_set = "M1"
    elif extra_mods == 1:
        st.session_state.modality_set = "M2"
    else:
        st.session_state.modality_set = "M3"

    # ── Log to Google Sheets ──────────────────────────────────────────────
    export_data = {
        "case_id": st.session_state.case_id,
        "clinical": st.session_state.clinical,
        "clinical_layer": st.session_state.get("clinical_layer","K1"),
        "arm_a_gpt4o_solo": to_jsonable(st.session_state.arm_a_result),
        "arm_b_gemini_solo": to_jsonable(st.session_state.arm_b_result),
        "arm_c_parallel_no_debate": to_jsonable(st.session_state.arm_c_result),
        "arm_d_debate_full": to_jsonable(st.session_state.arm_d_result),
    }
    sheets_ok = log_to_sheets(export_data)
    st.session_state.sheets_logged = sheets_ok


# =========================
# ── UI HELPERS ──
# =========================
def render_debate_expander(debate_log: dict):
    """Renders the debate transcript in a readable expander."""
    if not debate_log:
        return
    with st.expander("🧠 Debate transcript", expanded=False):
        r1 = debate_log.get("round_1", {})
        r2 = debate_log.get("round_2", {})
        r3 = debate_log.get("round_3", {})
        cv = debate_log.get("convergence", {})

        # Convergence badge
        agreed_r1 = cv.get("agreed_r1", False)
        agreed_r3 = cv.get("agreed_r3", False)
        changed   = cv.get("debate_changed_outcome", False)

        col1, col2, col3 = st.columns(3)
        with col1:
            badge1 = "agree-yes" if agreed_r1 else "agree-no"
            label1 = "✅ Agreed R1" if agreed_r1 else "❌ Disagreed R1"
            st.markdown(f'<span class="agree-badge {badge1}">{label1}</span>', unsafe_allow_html=True)
        with col2:
            badge3 = "agree-yes" if agreed_r3 else "agree-no"
            label3 = "✅ Agreed R3" if agreed_r3 else "❌ Disagreed R3"
            st.markdown(f'<span class="agree-badge {badge3}">{label3}</span>', unsafe_allow_html=True)
        with col3:
            if changed:
                st.markdown('<span class="agree-badge" style="background:#fef3c7;color:#92400e;">🔄 Debate changed outcome</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="agree-badge" style="background:#f3f4f6;color:#374151;">= No change from debate</span>', unsafe_allow_html=True)

        st.markdown("---")

        # Round 1
        st.markdown("**Round 1 — Independent analysis**")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Model A (GPT-4o)")
            oa = r1.get("openai", {})
            st.markdown(f"🔹 **{oa.get('top_diagnosis','—')}** ({oa.get('confidence','—')})")
            for ev in (oa.get("key_evidence") or []):
                st.markdown(f"- {ev}")
        with c2:
            st.caption("Model B (Gemini)")
            gm = r1.get("gemini", {})
            st.markdown(f"🔹 **{gm.get('top_diagnosis','—')}** ({gm.get('confidence','—')})")
            for ev in (gm.get("key_evidence") or []):
                st.markdown(f"- {ev}")

        st.markdown("---")

        # Round 2
        st.markdown("**Round 2 — Cross-critique**")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Model A critiques Model B")
            oa_c = r2.get("openai_critique_of_gemini", {})
            agree_icon = "✅" if oa_c.get("agree") else "❌"
            st.markdown(f"{agree_icon} **Agrees:** {oa_c.get('agree','—')}")
            st.markdown(f"_{oa_c.get('critique','—')}_")
            if oa_c.get("revised_diagnosis"):
                st.markdown(f"🔄 Revised to: **{oa_c['revised_diagnosis']}**")
            rtype = oa_c.get("revision_type","—")
            rcolor = {"evidence_based":"🟢","sycophantic":"🔴","maintained":"🔵"}.get(rtype,"⚪")
            st.markdown(f"{rcolor} {rtype}")
        with c2:
            st.caption("Model B critiques Model A")
            gm_c = r2.get("gemini_critique_of_openai", {})
            agree_icon = "✅" if gm_c.get("agree") else "❌"
            st.markdown(f"{agree_icon} **Agrees:** {gm_c.get('agree','—')}")
            st.markdown(f"_{gm_c.get('critique','—')}_")
            if gm_c.get("revised_diagnosis"):
                st.markdown(f"🔄 Revised to: **{gm_c['revised_diagnosis']}**")
            rtype = gm_c.get("revision_type","—")
            rcolor = {"evidence_based":"🟢","sycophantic":"🔴","maintained":"🔵"}.get(rtype,"⚪")
            st.markdown(f"{rcolor} {rtype}")

        st.markdown("---")

        # Round 3
        st.markdown("**Round 3 — Final positions**")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Model A final")
            oa3 = r3.get("openai_final", {})
            st.markdown(f"🔹 **{oa3.get('top_diagnosis','—')}** ({oa3.get('confidence','—')})")
        with c2:
            st.caption("Model B final")
            gm3 = r3.get("gemini_final", {})
            st.markdown(f"🔹 **{gm3.get('top_diagnosis','—')}** ({gm3.get('confidence','—')})")


# =========================
# ── UI: Clinical input ──
# =========================
st.markdown('''<div class="rgpt-section">
<div class="rgpt-section-title"><span></span>Clinical Information</div>
''', unsafe_allow_html=True)

# ── Hasta No + Case ID ───────────────────────────────────────────────────
if "patient_number" not in st.session_state:
    st.session_state.patient_number = 1

col_pid, col_cid = st.columns([1, 2])
with col_pid:
    patient_number = st.number_input(
        "Patient # ",
        min_value=1, max_value=9999,
        value=st.session_state.patient_number,
        step=1,
        help="P001, P002... formatında hasta numarası"
    )
    st.session_state.patient_number = int(patient_number)

_pid  = f"P{st.session_state.patient_number:03d}"
_mset = st.session_state.get("modality_set", "M1")
_klay = st.session_state.get("clinical_layer", "K0")
_auto_case_id = f"{_pid}-{_mset}-{_klay}"
st.session_state.case_id_str = _auto_case_id

with col_cid:
    st.markdown(
        f'''<div style="margin-top:28px;padding:8px 14px;background:#1B4F72;
        border-radius:8px;display:inline-block;">
        <span style="color:#AED6F1;font-size:0.7rem;font-weight:600;letter-spacing:1px;">CASE ID</span><br>
        <span style="color:#FFFFFF;font-size:1.1rem;font-weight:700;font-family:monospace;">{_auto_case_id}</span>
        </div>''',
        unsafe_allow_html=True
    )

# ── K Seçici — radio butonlar ────────────────────────────────────────────
k_opts = ["K0 — Image only","K1 — Basic clinical","K2 — Clinical findings","K3 — Full history"]
k_default = {"K0":0,"K1":1,"K2":2,"K3":3}.get(st.session_state.get("manual_k","K0"), 0)
k_choice = st.radio(
    "Clinical Knowledge Layer (K)",
    k_opts,
    index=k_default,
    horizontal=True,
    key="k_radio",
    help="Bu vakada hangi klinik bilgi katmanını dolduracaksınız?"
)
st.session_state.manual_k = k_choice.split("—")[0].strip()  # e.g. "K0"

# K bilgi kutusu
k_bgs   = {"K0":"#EBF5FB","K1":"#D1F2EB","K2":"#FEF9E7","K3":"#FDEDEC"}
k_bords = {"K0":"#2471A3","K1":"#0E6655","K2":"#B7950B","K3":"#A93226"}
k_texts = {"K0":"#1A5276","K1":"#0B5345","K2":"#7D6608","K3":"#7B241C"}
k_field_info = {
    "K0": "No clinical data — images only",
    "K1": "✅ Age   ✅ Sex   ✅ Analyzed eye (OD/OS)",
    "K2": "K1   +   ○ Visual acuity   ○ Symptom(s)   ○ Duration   ○ Involvement pattern",
    "K3": "K2   +   ○ Systemic diseases   ○ Ocular history   ○ Medications   ○ Family history   ○ Notes",
}
_sel_k = st.session_state.manual_k.strip()[:2]  # "K0", "K1", "K2", "K3"
st.markdown(
    f'''<div style="background:{k_bgs[_sel_k]};border-left:3px solid {k_bords[_sel_k]};
    border-radius:0 8px 8px 0;padding:8px 12px;margin:4px 0 14px 0;
    font-size:0.78rem;color:{k_texts[_sel_k]};line-height:1.7;">
    <strong>Doldurulacak alanlar:</strong>&nbsp; {k_field_info[_sel_k]}
    </div>''',
    unsafe_allow_html=True
)

# ── M Seçici — radio butonlar ─────────────────────────────────────────────
m_opts = ["M1 — Fundus only","M2 — Fundus + critical test","M3 — Fundus + 2 tests"]
m_default = {"M1":0,"M2":1,"M3":2}.get(st.session_state.get("manual_m","M1"), 0)
m_choice = st.radio(
    "Imaging Modality Set (M)",
    m_opts,
    index=m_default,
    horizontal=True,
    key="m_radio",
    help="Bu vakada kaç modalite yükleyeceksiniz?"
)
st.session_state.manual_m = m_choice.split("—")[0].strip()

m_bgs_r   = {"M1":"#EBF5FB","M2":"#D6EAF8","M3":"#AED6F1"}
m_bords_r = {"M1":"#2471A3","M2":"#1A5276","M3":"#1B4F72"}
m_texts_r = {"M1":"#1A5276","M2":"#154360","M3":"#1B4F72"}
m_field_info = {
    "M1": "✅ Color Fundus   — single modality, mandatory for all cases",
    "M2": "✅ Color Fundus   + disease-specific MOST CRITICAL test (see disease list)",
    "M3": "✅ Color Fundus   + M2 test   + SECOND complementary test (see disease list)",
}
_sel_m = st.session_state.manual_m
st.markdown(
    f'''<div style="background:{m_bgs_r[_sel_m]};border-left:3px solid {m_bords_r[_sel_m]};
    border-radius:0 8px 8px 0;padding:8px 12px;margin:4px 0 14px 0;
    font-size:0.78rem;color:{m_texts_r[_sel_m]};line-height:1.7;">
    <strong>Yüklenecek görüntüler:</strong>&nbsp; {m_field_info[_sel_m]}
    </div>''',
    unsafe_allow_html=True
)

# Manuel seçimleri modality_set olarak kaydet (case ID'ye yansısın)
st.session_state.modality_set = st.session_state.manual_m

# ── K0 Block ─────────────────────────────────────────────────────────────
st.markdown('''<div style="background:#EBF5FB;border-left:3px solid #2471A3;border-radius:0 8px 8px 0;
     padding:8px 12px;margin:14px 0 10px 0;font-size:0.75rem;font-weight:600;
     letter-spacing:0.8px;text-transform:uppercase;color:#1A5276;">
  K0 — Image Only (no clinical data)
</div>''', unsafe_allow_html=True)

st.markdown('''<div style="font-size:0.78rem;color:#5D6D7E;padding:4px 0 8px 0;">
  In K0 condition, no clinical data is entered. Only images are uploaded and analysis is initiated.
</div>''', unsafe_allow_html=True)

# Hide K1/K2/K3 blocks based on selection
_hide_k1 = _sel_k == "K0"
_hide_k2 = _sel_k in ("K0","K1")
_hide_k3 = _sel_k in ("K0","K1","K2")
if _hide_k1:
    st.markdown('''<style>
    [data-testid="stVerticalBlock"] .k1-block {display:none !important;}
    </style>''', unsafe_allow_html=True)

# ── K1 Block ─────────────────────────────────────────────────────────────
if not _hide_k1:
    st.markdown('''<div style="background:#D1F2EB;border-left:3px solid #0E6655;border-radius:0 8px 8px 0;
         padding:8px 12px;margin:14px 0 10px 0;font-size:0.75rem;font-weight:600;
         letter-spacing:0.8px;text-transform:uppercase;color:#0B5345;">
      K1 — Basic Clinical (Age + Sex + Analyzed Eye)
    </div>''', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120,
                          value=st.session_state.age or 0, step=1)
with col2:
    sex_opts = ["Select", "Male", "Female", "Other"]
    sex_map  = {"Select":"Select","Male":"Male","Female":"Female","Other":"Other"}
    cur_sex  = sex_map.get(st.session_state.sex, "Select")
    sex = st.selectbox("Sex", sex_opts,
                       index=sex_opts.index(cur_sex if cur_sex in sex_opts else "Seçiniz"))

lat_opts = ["Select", "OD (Right)", "OS (Left)"]
lat_map  = {"Select":"Select","OD (Right)":"OD (Right)","OS (Left)":"OS (Left)"}
cur_lat  = lat_map.get(st.session_state.laterality, "Select")
laterality = st.selectbox("Analyzed eye", lat_opts,
                           index=lat_opts.index(cur_lat if cur_lat in lat_opts else "Select"),
                           help="Which eye's images are being analyzed?")

# ── K2 Block ─────────────────────────────────────────────────────────────
if not _hide_k2:
    st.markdown('''<div style="background:#FEF9E7;border-left:3px solid #B7950B;border-radius:0 8px 8px 0;
         padding:8px 12px;margin:14px 0 10px 0;font-size:0.75rem;font-weight:600;
         letter-spacing:0.8px;text-transform:uppercase;color:#7D6608;">
      K2 — Clinical Findings (VA + Symptoms + Duration + Involvement)
    </div>''', unsafe_allow_html=True)

if not _hide_k2:
    col_va, col_dur = st.columns(2)
    with col_va:
        visual_acuity = st.text_input(
            "Visual acuity (Snellen)",
            value=st.session_state.get("visual_acuity",""),
            placeholder="e.g. 1.0 / 0.8 / 0.5 / 0.1 / CF / HM / LP",
            help="Raw Snellen: 1.0, 0.8, 0.5, 0.1 — CF: counting fingers, HM: hand motion, LP: light perception"
        )
    with col_dur:
        duration = st.text_input(
            "Duration / Onset",
            value=st.session_state.get("duration",""),
            placeholder="e.g. 3 days / 2 weeks / 6 months / chronic",
            help="When symptoms started or how long they have persisted"
        )
    symp_all = ["Visual loss","Metamorphopsia","Scotoma","Photopsia",
                "Floaters","Nyctalopia (night blindness)",
                "Dyschromatopsia","Peripheral visual loss",
                "Asymptomatic","Other"]
    cur_symp = st.session_state.get("primary_symptom",[])
    if isinstance(cur_symp, str): cur_symp = [cur_symp] if cur_symp else []
    primary_symptom = st.multiselect(
        "Primary symptom(s)",
        options=symp_all,
        default=[s for s in cur_symp if s in symp_all],
        help="Multiple symptoms can be selected"
    )
    inv_opts = ["Select","Unilateral","Bilateral","Unknown"]
    cur_inv = st.session_state.get("involvement","Select")
    if cur_inv not in inv_opts: cur_inv = "Select"
    involvement = st.selectbox(
        "Involvement pattern",
        inv_opts,
        index=inv_opts.index(cur_inv if cur_inv in inv_opts else "Select"),
        help="Is the disease unilateral or bilateral?"
    )
else:
    visual_acuity = st.session_state.get("visual_acuity","")
    duration = st.session_state.get("duration","")
    primary_symptom = st.session_state.get("primary_symptom",[])
    involvement = st.session_state.get("involvement","Select")

# ── K3 Block ─────────────────────────────────────────────────────────────
if not _hide_k3:
    st.markdown('''<div style="background:#FDEDEC;border-left:3px solid #A93226;border-radius:0 8px 8px 0;
         padding:8px 12px;margin:14px 0 10px 0;font-size:0.75rem;font-weight:600;
         letter-spacing:0.8px;text-transform:uppercase;color:#7B241C;">
      K3 — Full History (Systemic + Ocular + Medications + Family + Notes)
    </div>''', unsafe_allow_html=True)

if not _hide_k3:
    sys_all = ["Diabetes mellitus","Hypertension","Autoimmune disease",
               "Malignancy history","Thyroid disease","High myopia",
               "Smoking","Pregnancy","None","Other"]
    cur_sys = st.session_state.get("systemic_diseases",[])
    if isinstance(cur_sys, str): cur_sys = [cur_sys] if cur_sys else []
    systemic_diseases = st.multiselect(
        "Systemic diseases",
        options=sys_all,
        default=[s for s in cur_sys if s in sys_all],
        help="Multiple systemic diseases can be selected"
    )
    col_k3a, col_k3b = st.columns(2)
    with col_k3a:
        ocular_history = st.text_input(
            "Ocular history",
            value=st.session_state.get("ocular_history",""),
            placeholder="e.g. laser, anti-VEGF injection, vitrectomy, cataract surgery..."
        )
        family_history = st.text_input(
            "Family history",
            value=st.session_state.get("family_history",""),
            placeholder="e.g. family history of AMD, RP, glaucoma, DM..."
        )
    with col_k3b:
        medications = st.text_input(
            "Medications",
            value=st.session_state.get("medications",""),
            placeholder="e.g. metformin, warfarin, chloroquine, corticosteroids..."
        )
        additional_notes = st.text_input(
            "Additional notes",
            value=st.session_state.get("additional_notes",""),
            placeholder="trauma history, gestational week, systemic treatment, other..."
        )
else:
    systemic_diseases = st.session_state.get("systemic_diseases",[])
    ocular_history = st.session_state.get("ocular_history","")
    medications = st.session_state.get("medications","")
    family_history = st.session_state.get("family_history","")
    additional_notes = st.session_state.get("additional_notes","")

# ── Update session state ──────────────────────────────────────────────────
st.session_state.age                = int(age)
st.session_state.sex = sex
st.session_state.laterality = laterality if laterality != "Select" else ""
st.session_state.visual_acuity      = visual_acuity or ""
st.session_state.primary_symptom    = primary_symptom
st.session_state.duration           = duration or ""
st.session_state.involvement        = involvement if involvement != "Select" else ""
st.session_state.systemic_diseases  = systemic_diseases
st.session_state.ocular_history     = ocular_history or ""
st.session_state.medications        = medications or ""
st.session_state.family_history     = family_history or ""
st.session_state.additional_notes   = additional_notes or ""

# ── Clinical layer detection ──────────────────────────────────────────────
def detect_clinical_layer():
    k3 = any([
        bool(st.session_state.get("systemic_diseases",[])),
        bool((st.session_state.get("ocular_history") or "").strip()),
        bool((st.session_state.get("medications") or "").strip()),
        bool((st.session_state.get("family_history") or "").strip()),
        bool((st.session_state.get("additional_notes") or "").strip()),
    ])
    k2 = any([
        bool((st.session_state.get("visual_acuity") or "").strip()),
        bool(st.session_state.get("primary_symptom",[])),
        bool((st.session_state.get("duration") or "").strip()),
        st.session_state.get("involvement","") not in ("","Select"),
    ])
    k1 = st.session_state.laterality not in ("", "Select")
    if k3: return "K3"
    if k2: return "K2"
    if k1: return "K1"
    return "K0"

st.session_state.clinical_layer = detect_clinical_layer()

# Layer indicator
layer_colors = {"K0":"#EBF5FB","K1":"#D1F2EB","K2":"#FEF9E7","K3":"#FDEDEC"}
layer_text   = {"K0":"#1A5276","K1":"#0B5345","K2":"#7D6608","K3":"#7B241C"}
cur_layer = st.session_state.clinical_layer
st.markdown(
    f'<div style="margin-top:8px;display:inline-block;padding:4px 12px;'
    f'border-radius:999px;background:{layer_colors[cur_layer]};'
    f'color:{layer_text[cur_layer]};font-size:0.75rem;font-weight:600;">'
    f'Clinical layer: {cur_layer}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ── UI: Image upload + modality tagging ──
# =========================
st.markdown('''<div class="rgpt-section">
<div class="rgpt-section-title"><span></span>Retinal Imaging</div>
''', unsafe_allow_html=True)

MODALITY_OPTIONS = ["OCT", "Fundus photo", "FAF", "FA", "OCTA", "ICGA", "Wide-field", "Other"]
MODALITY_COLORS  = {
    "OCT":         "#0ea5e9",
    "Fundus photo":"#10b981",
    "FAF":         "#f59e0b",
    "FA":          "#8b5cf6",
    "OCTA":        "#ec4899",
    "ICGA":        "#6366f1",
    "Wide-field":  "#14b8a6",
    "Other":       "#94a3b8",
}

uploads = st.file_uploader(
    "Upload images — OCT, fundus, FAF, FA, OCTA, ICGA, wide-field",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
)

# Per-image modality selector
image_modalities = {}
if uploads:
    st.markdown('<div style="margin-top:10px;">', unsafe_allow_html=True)
    for i, f in enumerate(uploads):
        col_img, col_sel = st.columns([1, 3])
        with col_img:
            st.image(f, use_container_width=True)
        with col_sel:
            mod_key = f"mod_{st.session_state.uploader_key}_{i}_{f.name}"
            # Guess modality from filename
            fname_lower = f.name.lower()
            default_mod = "OCT"
            for guess, keywords in [
                ("OCT",          ["oct"]),
                ("Fundus photo", ["fundus","cfp","color","photo"]),
                ("FAF",          ["faf","autofluor"]),
                ("FA",           ["fa_","_fa","fluore","angio"]),
                ("OCTA",         ["octa","angiograph"]),
                ("ICGA",         ["icga","indocyan"]),
                ("Wide-field",   ["wide","optos","clarus"]),
            ]:
                if any(k in fname_lower for k in keywords):
                    default_mod = guess
                    break
            sel = st.selectbox(
                f.name[:35] + ("…" if len(f.name) > 35 else ""),
                MODALITY_OPTIONS,
                index=MODALITY_OPTIONS.index(default_mod),
                key=mod_key,
                label_visibility="visible"
            )
            image_modalities[f.name] = sel
            dot_color = MODALITY_COLORS.get(sel, "#94a3b8")
            st.markdown(
                f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                f'background:{dot_color};margin-right:5px;vertical-align:middle;"></span>'
                f'<span style="font-size:0.75rem;color:{dot_color};font-weight:600;">{sel}</span>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="margin-top:8px;">', unsafe_allow_html=True)
st.markdown('<span class="small-note">* K0: images only. K1+: Age and Sex required.</span>', unsafe_allow_html=True)
analyze = st.button("🔬  Analyze  —  3-Round Debate", type="primary", use_container_width=True)

if analyze:
    missing = []
    # Read K level directly from radio widget value (most current)
    _k_raw = st.session_state.get("k_radio", "K0 — Image only")
    _k = _k_raw.split("—")[0].strip()
    if _k not in ("K0", "K0 ") :
        if sex == "Select": missing.append("Sex")
        if age == 0: missing.append("Age")
    if missing:
        st.error("Please complete required fields: " + ", ".join(missing))
    elif not uploads:
        st.error("Please upload at least one retinal image.")
    else:
        # Build clinical summary with modality info
        st.session_state.clinical = build_clinical_summary()
        # Append modality context to clinical text
        if image_modalities:
            mod_lines = [f"  - {fname}: {mod}" for fname, mod in image_modalities.items()]
            st.session_state.clinical += "\n\nImaging modalities provided:\n" + "\n".join(mod_lines)
        st.session_state.images = uploads_to_images(uploads)
        run_analysis()
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# ── UI: Error states ──
# =========================
if st.session_state.last_error_type:
    st.markdown('<div class="rgpt-section">', unsafe_allow_html=True)
    msg = "System is busy — please retry in a few seconds." \
          if st.session_state.last_error_type == "busy" \
          else "Analysis could not be completed. Please retry."
    st.warning(msg)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔁 Retry", use_container_width=True):
            if st.session_state.images:
                run_analysis(); st.rerun()
    with c2:
        if st.button("🆕 New case", use_container_width=True):
            reset_case(); st.rerun()
    with st.expander("Technical details"):
        st.code(st.session_state.last_error_raw or "No details available.")
    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# ── UI: Report output ──
# =========================
if st.session_state.report_history:
    conf_label = st.session_state.confidence_label or "—"
    conf_icon  = st.session_state.confidence_icon  or "🟡"
    conf_class = {"High":"conf-high","Moderate":"conf-mod","Low":"conf-low"}.get(conf_label,"conf-mod")

    st.markdown(f'''<div class="rgpt-section">
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
  <div class="rgpt-section-title" style="margin:0"><span></span>Diagnostic Report</div>
  <div>
    <span class="pill {conf_class}">{conf_icon} {conf_label} confidence</span>
    <span class="pill pill-muted">Case #{st.session_state.case_id}</span>
  </div>
</div>
''', unsafe_allow_html=True)

    for idx, item in enumerate(st.session_state.report_history):
        c_label = item.get("confidence_label","Moderate")
        c_icon  = item.get("confidence_icon","🟡")
        c_cls   = {"High":"conf-high","Moderate":"conf-mod","Low":"conf-low"}.get(c_label,"conf-mod")

        st.markdown(f'''<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
  <span style="font-weight:600;font-size:0.9rem;color:#1e293b;">{item["label"]}</span>
  <span class="pill {c_cls}" style="font-size:0.72rem;">{c_icon} {c_label}</span>
</div>''', unsafe_allow_html=True)

        if item.get("debate_log"):
            render_debate_expander(item["debate_log"])

        st.markdown('<div class="report-box">', unsafe_allow_html=True)
        st.markdown(item["report"])
        st.markdown('</div>', unsafe_allow_html=True)

        if idx < len(st.session_state.report_history) - 1:
            st.markdown('<div class="rgpt-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="rgpt-divider" style="margin-top:14px;"></div>', unsafe_allow_html=True)

    # ── Expert review flag ───────────────────────────────────────────────
    if st.session_state.get("high_uncertainty_case") is True:
        st.markdown(
            '''<div style="display:flex;align-items:center;gap:8px;padding:8px 14px;
            border-radius:8px;background:#FEF9E7;border:1px solid #F39C12;margin-bottom:10px;">
            <span style="font-size:1rem;">⚡</span>
            <div>
            <div style="font-weight:700;color:#7D6608;font-size:0.8rem;">YÜKSEK BELİRSİZLİK — debate_unresolved: TRUE</div>
            <div style="color:#7D6608;font-size:0.75rem;">R3 sonrası tanısal uyumsuzluk devam ediyor veya güven düzeyi düşük.</div>
            </div></div>''',
            unsafe_allow_html=True
        )
    elif st.session_state.get("high_uncertainty_case") is False and st.session_state.get("analysis_done"):
        st.markdown(
            '''<div style="display:inline-flex;align-items:center;gap:6px;padding:4px 12px;
            border-radius:999px;background:#D5F5E3;color:#1A7A4A;font-size:0.75rem;
            font-weight:600;margin-bottom:10px;">✓ Tanısal konsensüs — debate_unresolved: FALSE</div>''',
            unsafe_allow_html=True
        )

    # Sheets status indicator
    if st.session_state.get("sheets_logged") is True:
        st.markdown(
            '<div style="display:inline-flex;align-items:center;gap:6px;'
            'padding:4px 12px;border-radius:999px;background:#D1F2EB;'
            'color:#0B5345;font-size:0.75rem;font-weight:600;margin-bottom:10px;">'
            "&#x2705; Google Sheets kaydedildi</div>",
            unsafe_allow_html=True
        )
    elif st.session_state.get("sheets_logged") is False:
        st.markdown(
            '<div style="display:inline-flex;align-items:center;gap:6px;'
            'padding:4px 12px;border-radius:999px;background:#FDEBD0;'
            'color:#784212;font-size:0.75rem;font-weight:600;margin-bottom:10px;">'
            "&#x26A0; Sheets: save failed — download JSON</div>",
            unsafe_allow_html=True
        )

    btn_c1, btn_c2, btn_c3 = st.columns(3)
    with btn_c1:
        if st.session_state.debate_log:
            export_data = {
                "case_id": st.session_state.case_id,
                "clinical": st.session_state.clinical,
                "clinical_layer": st.session_state.get("clinical_layer","K1"),
                "modality_set": st.session_state.get("modality_set","M1"),
                "arm_a_gpt4o_solo": to_jsonable(st.session_state.arm_a_result),
                "arm_b_gemini_solo": to_jsonable(st.session_state.arm_b_result),
                "arm_c_parallel_no_debate": to_jsonable(st.session_state.arm_c_result),
                "arm_d_debate_full": to_jsonable(st.session_state.arm_d_result),
            }
            st.download_button(
                label="📥 Export JSON",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=f"retinagpt_case_{st.session_state.case_id}.json",
                mime="application/json",
                use_container_width=True,
            )
    with btn_c2:
        if st.button("🔄 New condition (same patient)", use_container_width=True,
                     help="Keeps patient info and K/M selection — clears images and results"):
            reset_condition()
            st.rerun()
    with btn_c3:
        if st.button("🆕 New patient", use_container_width=True,
                     help="Clears everything — next patient number"):
            st.session_state.patient_number = st.session_state.get("patient_number",1) + 1
            reset_case()
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# ── UI: Add more images ──
# =========================
if st.session_state.report_history:
    st.markdown('''<div class="rgpt-section">
<div class="rgpt-section-title"><span></span>Add Additional Imaging</div>
<p style="font-size:0.8rem;color:#64748b;margin:0 0 12px 0;">Upload more modalities for the same case to refine the assessment.</p>
''', unsafe_allow_html=True)

    modality_options = ["OCT","Fundus photo","FAF","FA","OCTA","ICGA","Wide-field imaging","Other"]
    cur_mod = st.session_state.additional_modality
    extra_modality = st.selectbox("Imaging modality", modality_options,
                                   index=modality_options.index(cur_mod if cur_mod in modality_options else "Other"),
                                   key="extra_modality_box")
    st.session_state.additional_modality = extra_modality

    extra_note = st.text_area("Clinical note",
                               placeholder="e.g. FA shows leakage at fovea, OCTA reveals type 1 MNV...",
                               value=st.session_state.additional_clinical_note,
                               key="additional_note_box", height=72)
    st.session_state.additional_clinical_note = extra_note or ""

    extra_uploads = st.file_uploader("Upload additional images",
                                      type=["jpg","jpeg","png","webp"],
                                      accept_multiple_files=True,
                                      key=f"add_uploader_{st.session_state.add_uploader_key}")

    if extra_uploads:
        cols = st.columns(3)
        for i, f in enumerate(extra_uploads[:3]):
            with cols[i % 3]:
                st.image(f, caption=f.name[:20], use_container_width=True)
        if len(extra_uploads) > 3:
            st.caption(f"+ {len(extra_uploads) - 3} more image(s).")

    if st.button("🔄 Update Diagnosis", use_container_width=True):
        if not extra_uploads and not st.session_state.additional_clinical_note.strip():
            st.warning("Please upload at least one image or add a clinical note.")
        else:
            if extra_uploads:
                st.session_state.images = merge_images(
                    st.session_state.images, uploads_to_images(extra_uploads)
                )
            update_note = f"Newly added imaging modality: {st.session_state.additional_modality}"
            if st.session_state.additional_clinical_note.strip():
                update_note += "\n" + st.session_state.additional_clinical_note.strip()
            if st.session_state.clinical.strip():
                st.session_state.clinical = st.session_state.clinical.strip() + "\n\nAdditional update:\n" + update_note
            else:
                st.session_state.clinical = update_note
            st.session_state.additional_clinical_note = ""
            st.session_state.additional_modality      = "OCT"
            st.session_state.add_uploader_key        += 1
            run_analysis()
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# ── UI: Follow-up chat ──
# =========================
if st.session_state.report_history:
    st.markdown('''<div class="rgpt-section">
<div class="rgpt-section-title"><span></span>Follow-up Discussion</div>
''', unsafe_allow_html=True)

    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_msg = st.chat_input("Ask about this case — management, imaging, differentials…")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                ans = chat_reply(user_msg)
                st.write(ans)
        st.session_state.chat_history.append({"role": "assistant", "content": ans})

    st.markdown('</div>', unsafe_allow_html=True)
