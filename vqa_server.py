import io
import json
import streamlit as st
import os, sys
import logging

logging.basicConfig(level=logging.DEBUG)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labelizer import (
    get_vqa_prompts,
    process_image,
    is_url_of_image_file,
    cleanup_string,
)

ACCEPTED_FILE_TYPES = ["png", "jpg", "jpeg"]
DOCUMENT_OPTIONS = ["Please Select", "URL", "File Upload"]

# Page title
st.set_page_config(page_title="VQA Demo")
st.title("VQA Demo")

# options = ['Please Select','File Upload','URL']
selected_option = st.selectbox("Select Document Type", DOCUMENT_OPTIONS)

url_text = None
uploaded_file = None

if selected_option == "URL":
    url_text = st.text_input("Enter your url:", placeholder="Please provide a URL.")
elif selected_option == "File Upload":
    uploaded_file = st.file_uploader("Upload an image", type=ACCEPTED_FILE_TYPES)

prompts = st.text_area(
    "Enter your prompt(s):",
    placeholder="Please provide a prompt.",
    height=600,
    value=json.dumps(get_vqa_prompts(), indent=2),
)


with st.form("vqa", clear_on_submit=True):
    submitted = st.form_submit_button(
        "Submit", disabled=not (uploaded_file or url_text)
    )
    if submitted:
        input = (
            url_text
            if is_url_of_image_file(url_text)
            else io.BytesIO(uploaded_file.getvalue())
        )
        if not input:
            st.error("Please provide a URL or upload an image.")
            st.stop()

        (keywords, caption) = process_image(input, json.loads(prompts.strip()))
        output = cleanup_string(caption + (f" ({keywords})" if keywords else ""))
        # st.write(response)
        st.text_area("Response", value=output, height=400, max_chars=None, key=None)
