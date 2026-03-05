import os
import json
import base64
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from openai import OpenAI

# Optional RAG deps (only if you use local FAISS index)
try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    faiss = None
    np = None

# Optional image preprocessing deps
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None


# -----------------------------
# CONFIG
# -----------------------------
APP_TITLE = "RetinaGPT v2"
APP_SUBTITLE = "Academic + Decision Support (Educational only)"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RAG_DIR = os.getenv("RAG_DIR", "data")  # expects data/index.faiss and data/meta.json
MAX_RAG_HITS = int(os.getenv("MAX_RAG_HITS", "6"))

st.set_page_config(page_title=APP_TITLE, page_icon="👁️", layout="wide")


# -----------------------------
# SYSTEM PROMPT (v2) - robust + albinism trigger
# -----------------------------
SYSTEM_PROMPT = """
You are RetinaGPT v2, a retina subspecialty educational imaging discussion and decision-support system.

IMPORTANT CONTEXT
- The user will upload DE-IDENTIFIED ophthalmic images (retinal fundus/OCT/FAF/angiography).
- These are NOT photos of faces/people. Do NOT attempt to identify any person.
- Your task is to describe imaging morphology and provide an EDUCATIONAL differential diagnosis.
- Do NOT provide patient-specific treatment instructions. Not medical advice.

PURPOSE
Provide structured academic discussion of retinal imaging findings and differential diagnostic reasoning
for educational purposes only.

STYLE
- Formal medical English, objective, concise.
- Use retina subspecialty terminology (e.g., EZ/ellipsoid zone, RPE, SHRM, PED, hypertransmission, etc.).
- Avoid over-commitment for rare/atypical patterns; describe morphology first, then offer a ranked differential.

REFERENCE KNOWLEDGE (RAG)
If “REFERENCE CARDS” are provided, you MUST treat them as the primary factual source.
- Use them to refine discriminators, pitfalls, work-up, and management (high level).
- If imaging suggests a different pattern than retrieved cards, explicitly state discrepancy and explain why.
- Do not invent facts not supported by imaging/metadata/reference cards.

GLOBAL SAFETY
- Educational purposes only.
- No individualized treatment plans, dosing, or urgent step-by-step instructions.
- For potentially vision-threatening patterns, suggest prompt clinical evaluation in general terms.

OUTPUT REQUIREMENTS (IMPORTANT)
You MUST respond in TWO blocks:
(A) A JSON block inside a Markdown code fence labeled json (STRICT JSON, no trailing commas)
(B) A human-readable report in the specified format.

(A) JSON schema:
{
  "case_summary": {
    "age": "string or null",
    "sex": "string or null",
    "symptoms": "string or null",
    "duration": "string or null",
    "laterality": "string or null",
    "history": "string or null"
  },
  "modalities_detected": ["Fundus","OCT","FAF","FA","ICGA","OCTA"],
  "missing_modalities_suggested": ["OCT", "FAF", ...],
  "patterns": [
    {"name": "string", "confidence": 0.0},
    {"name": "string", "confidence": 0.0}
  ],
  "feature_checklist": {
    "subretinal_fluid": "PRESENT|ABSENT|UNCERTAIN",
    "intraretinal_fluid": "PRESENT|ABSENT|UNCERTAIN",
    "hemorrhage_exudation": "PRESENT|ABSENT|UNCERTAIN",
    "inner_retinal
