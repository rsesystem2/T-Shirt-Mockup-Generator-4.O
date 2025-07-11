import streamlit as st
from PIL import Image
import numpy as np
import zipfile
import io
import cv2
import os

st.set_page_config(page_title="Shirt Mockup Generator", layout="centered")
st.title("👕 Shirt Mockup Generator with Batching")

st.markdown("""
Upload multiple design PNGs and shirt templates.  
Preview placement and generate mockups in batches.
""")

# --- Sidebar Controls ---
plain_padding_ratio = st.sidebar.slider("Padding Ratio – Plain Shirt", 0.1, 1.0, 0.45, 0.05)
model_padding_ratio = st.sidebar.slider("Padding Ratio – Model Shirt", 0.1, 1.0, 0.35, 0.05)
plain_offset_pct = st.sidebar.slider("Vertical Offset – Plain Shirt (%)", -50, 100, -7, 1)
model_offset_pct = st.sidebar.slider("Vertical Offset – Model Shirt (%)", -50, 100, 3, 1)

# --- Session Setup ---
if "zip_files_output" not in st.session_state:
    st.session_state.zip_files_output = {}
if "design_names" not in st.session_state:
    st.session_state.design_names = {}

# --- Upload Section ---
design_files = st.file_uploader("📌 Upload Design Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
shirt_files = st.file_uploader("🎨 Upload Shirt Templates", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# --- Clear Button ---
if st.button("🔄 Start Over (Clear Generated Mockups)"):
    for key in ["design_files", "design_names", "zip_files_output"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- Design Naming ---
if design_files:
    st.markdown("### ✏️ Name Each Design")
    for i, file in enumerate(design_files):
        default_name = os.path.splitext(file.name)[0]
        custom_name = st.text_input(
            f"Name for Design {i+1} ({file.name})", 
            value=st.session_state.design_names.get(file.name, default_name),
            key=f"name_input_{i}_{file.name}"
        )
        st.session_state.design_names[file.name] = custom_name

# --- Batch Controls ---
if design_files:
    st.markdown("### 📦 Batch Processing Control")
    total_designs = len(design_files)
    batch_start = st.number_input("Start from Design #", min_value=1, max_value=total_designs, value=1)
    batch_end = st.number_input("End at Design #", min_value=batch_start, max_value=total_designs, value=min(batch_start + 19, total_designs))
    selected_batch = design_files[batch_start - 1: batch_end]

# --- Bounding Box Detection ---
def get_shirt_bbox(pil_image):
    img_cv = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest)
    return None

# --- Live Preview ---
if design_files and shirt_files:
    st.markdown("### 👀 Live Preview")
    selected_design = st.selectbox("Select a Design", design_files, format_func=lambda x: x.name)
    selected_shirt = st.selectbox("Select a Shirt Template", shirt_files, format_func=lambda x: x.name)

    try:
        selected_design.seek(0)
        design = Image.open(selected_design).convert("RGBA")
        selected_shirt.seek(0)
        shirt = Image.open(selected_shirt).convert("RGBA")

        is_model = "model" in selected_shirt.name.lower()
        offset_pct = model_offset_pct if is_model else plain_offset_pct
        padding_ratio = model_padding_ratio if is_model else plain_padding_ratio

        bbox = get_shirt_bbox(shirt)
        if bbox:
            sx, sy, sw, sh = bbox
            scale = min(sw / design.width, sh / design.height, 1.0) * padding_ratio
            new_width = int(design.width * scale)
            new_height = int(design.height * scale)
            resized_design = design.resize((new_width, new_height))
            y_offset = int(sh * offset_pct / 100)
            x = sx + (sw - new_width) // 2
            y = sy + y_offset
        else:
            resized_design = design
            x = (shirt.width - design.width) // 2
            y = (shirt.height - design.height) // 2

        preview = shirt.copy()
        preview.paste(resized_design, (x, y), resized_design)
        st.image(preview, caption="📸 Live Mockup Preview", use_column_width=True)
    except Exception as e:
        st.error(f"⚠️ Preview failed: {e}")

# --- Generate Mockups ---
if st.button("🚀 Generate Mockups for Selected Batch"):
    if not (selected_batch and shirt_files):
        st.warning("Upload at least one design and one shirt template.")
    else:
        master_zip = io.BytesIO()
        with zipfile.ZipFile(master_zip, "w", zipfile.ZIP_DEFLATED) as master_zipf:
            for design_file in selected_batch:
                graphic_name = st.session_state.design_names.get(design_file.name, "graphic")
                design_file.seek(0)
                design = Image.open(design_file).convert("RGBA")

                inner_zip_buffer = io.BytesIO()
                with zipfile.ZipFile(inner_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for shirt_file in shirt_files:
                        color_name = os.path.splitext(shirt_file.name)[0]
                        shirt_file.seek(0)
                        shirt = Image.open(shirt_file).convert("RGBA")

                        is_model = "model" in shirt_file.name.lower()
                        offset_pct = model_offset_pct if is_model else plain_offset_pct
                        padding_ratio = model_padding_ratio if is_model else plain_padding_ratio

                        bbox = get_shirt_bbox(shirt)
                        if bbox:
                            sx, sy, sw, sh = bbox
                            scale = min(sw / design.width, sh / design.height, 1.0) * padding_ratio
                            new_width = int(design.width * scale)
                            new_height = int(design.height * scale)
                            resized_design = design.resize((new_width, new_height))
                            y_offset = int(sh * offset_pct / 100)
                            x = sx + (sw - new_width) // 2
                            y = sy + y_offset
                        else:
                            resized_design = design
                            x = (shirt.width - design.width) // 2
                            y = (shirt.height - design.height) // 2

                        shirt_copy = shirt.copy()
                        shirt_copy.paste(resized_design, (x, y), resized_design)

                        output_name = f"{graphic_name}_{color_name}_tee.png"
                        img_byte_arr = io.BytesIO()
                        shirt_copy.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        zipf.writestr(output_name, img_byte_arr.getvalue())

                inner_zip_buffer.seek(0)
                master_zipf.writestr(f"{graphic_name}.zip", inner_zip_buffer.read())

        master_zip.seek(0)
        st.download_button(
            label="📦 Download All Mockups (Grouped by Design)",
            data=master_zip,
            file_name="all_mockups_by_design.zip",
            mime="application/zip"
        )
