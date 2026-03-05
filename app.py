# ... (kodun baş kısmı aynı kalıyor)

# ----------------------------
# GEMINI VISION (güncellenmiş - 2025/2026 SDK uyumlu)
# ----------------------------

def gemini_part_text(text: str):
    return types.Part(text=text)


def gemini_part_bytes(data: bytes, mime_type: str):
    return types.Part(
        inline_data=types.Blob(
            mime_type=mime_type,
            data=data
        )
    )


def call_gemini_vision():
    if not (use_gemini and gemini_client):
        return None, None

    prompt = """
You are a retina specialist.
Task:
1) Describe morphology first.
2) Output STRICT JSON ONLY (no markdown, no extra text).
Return JSON keys:
- modalities_detected: list[str]
- image_quality: one of ["POOR","FAIR","GOOD","EXCELLENT"]
- key_findings: list[str]
- top_diagnoses: list[{"name": str, "confidence": float, "for": list[str], "against": list[str]}]
"""

    parts = [gemini_part_text(prompt)]

    for f in uploads:
        parts.append(gemini_part_bytes(f.getvalue(), f.type))

    # Güncel kullanım: Content objesi ile
    contents = [types.Content(role="user", parts=parts)]

    # Generation config (JSON zorlamak için)
    generation_config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json"
    )

    # Call Gemini
    try:
        resp = gemini_client.models.generate_content(
            model=gemini_model,
            contents=contents,
            generation_config=generation_config
        )

        # Response extraction (en yaygın iki yapıya karşı güvenli)
        if hasattr(resp, "text") and resp.text:
            raw = resp.text
        else:
            try:
                raw = resp.candidates[0].content.parts[0].text
            except (AttributeError, IndexError, KeyError):
                raw = str(resp)  # fallback

        js = parse_json_loose(raw)
        return raw, js

    except Exception as e:
        error_msg = f"Gemini call failed: {str(e)}"
        return json.dumps({"error": error_msg})[:2000], None


# ----------------------------
# UI RUN (burası da aynı kalıyor)
# ----------------------------
if run:
    if not uploads:
        st.error("Please upload at least one image.")
        st.stop()

    with st.spinner("Running OpenAI vision..."):
        openai_raw, openai_json = call_openai_vision()

    with st.spinner("Running Gemini vision..."):
        gemini_raw, gemini_json = call_gemini_vision()

    with st.spinner("Generating final report..."):
        final_report = build_final_report(openai_json, gemini_json)

    st.subheader("Final Report")
    st.write(final_report)

    st.subheader("Debug")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### OpenAI raw")
        st.code(openai_raw or "")
        st.markdown("### OpenAI JSON")
        st.json(openai_json)
    with col2:
        st.markdown("### Gemini raw")
        st.code(gemini_raw or "")
        st.markdown("### Gemini JSON")
        st.json(gemini_json)
