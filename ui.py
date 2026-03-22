import os
import tempfile

import streamlit as st

from main import extract_text_from_image, extract_text_from_video, process_post


st.set_page_config(page_title="Vernacular Fact Checker", page_icon="🧾", layout="centered")
st.title("🧾 Vernacular Fact Checker")
st.caption("Paste text or upload image/video to check claim status.")

input_mode = st.radio("Choose input type", ["Text", "Image", "Video"], horizontal=True)


def render_result(result: dict) -> None:
    verification = result.get("verification", {})
    st.subheader("Result")
    st.write(f"**Verdict:** {verification.get('verdict', 'N/A')}")
    st.write(f"**Confidence:** {verification.get('confidence', 'N/A')}")
    st.write(f"**Evidence Source:** {verification.get('evidence_source', 'N/A')}")
    st.write(f"**Claim:** {result.get('claim', '')}")
    with st.expander("Details"):
        st.json(result)


if input_mode == "Text":
    text = st.text_area("Paste news text", height=180)
    if st.button("Check Text", type="primary"):
        if not text.strip():
            st.error("Please enter text.")
        else:
            try:
                render_result(process_post(text))
            except Exception as exc:
                st.error(str(exc))

elif input_mode == "Image":
    file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
    if st.button("Check Image", type="primary"):
        if file is None:
            st.error("Please upload an image.")
        else:
            suffix = os.path.splitext(file.name)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                temp.write(file.read())
                temp_path = temp.name
            try:
                text = extract_text_from_image(temp_path)
                render_result(process_post(text))
            except Exception as exc:
                st.error(str(exc))
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

else:
    file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
    if st.button("Check Video", type="primary"):
        if file is None:
            st.error("Please upload a video.")
        else:
            suffix = os.path.splitext(file.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                temp.write(file.read())
                temp_path = temp.name
            try:
                text = extract_text_from_video(temp_path)
                result = process_post(text)
                result["input_video"] = file.name
                render_result(result)
            except Exception as exc:
                st.error(str(exc))
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
