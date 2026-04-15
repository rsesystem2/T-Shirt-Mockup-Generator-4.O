import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import zipfile
import io
import cv2
import os

st.set_page_config(page_title="Pro Shirt Mockup Generator", layout="centered")
st.title("👕 Pro Shirt Mockup Generator (Realistic Blending)")

st.markdown("""
Upload designs and templates. This version uses **Texture Mapping** and **Multiply Blending** to make designs follow the fabric's natural shadows and folds.
""")

# --- Sidebar: Placement Controls ---
st.sidebar.header("📍 Placement Settings")
plain_padding_ratio = st.sidebar.slider("Padding Ratio – Plain Shirt", 0.1, 1.0, 0.45, 0.05)
model_padding_ratio = st.sidebar.slider("Padding Ratio – Model Shirt", 0.1, 1.0, 0.35, 0.05)
plain_offset_pct = st.sidebar.slider("Vertical Offset – Plain Shirt (%)", -50, 100, 23, 1)
model_offset_pct = st.sidebar.slider("Vertical Offset – Model Shirt (%)", -50, 100, 38, 1)

# --- Sidebar: Realism Controls ---
st.sidebar.header("✨ Realism Settings")
ink_opacity = st.sidebar.slider("Ink Opacity (Breathability)", 0.5, 1.0, 0.92, 0.01)
shadow_intensity = st.sidebar.slider("Texture/Shadow Depth", 0.0, 2.0, 1.1, 0.1)
blur_edges = st.sidebar.checkbox("Subtle Edge Softening (Realistic Ink)", value=True)

# --- Session Setup ---
if "design_names" not in st.session_state:
    st.session_state.design_names = {}

# --- Helper: Realistic Blending Engine ---
def apply_realistic_blending(shirt_bg, design_img, x, y, size):
    # 1. Resize design
    design_res = design_img.resize(size, Image.Resampling.LANCZOS)
    
    # 2. Create the design layer
    design_layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
    design_layer.paste(design_res, (x, y), design_res)

    # 3. Analyze shirt brightness to prevent "fading" on black
    shirt_stat = ImageEnhance.Brightness(shirt_bg.convert("L")).enhance(1.0)
    avg_brightness = np.array(shirt_stat).mean()
    
    # 4. Create Texture/Shadow Map
    # If the shirt is dark, we reduce the intensity of the 'Multiply' 
    # so it doesn't kill the design colors.
    shirt_gray = shirt_bg.convert("L")
    if avg_brightness < 100:  # Dark shirt logic
        # On dark shirts, we want to extract the highlights/folds
        texture_map = ImageEnhance.Contrast(shirt_gray).enhance(shadow_intensity * 0.5)
    else:
        texture_map = ImageEnhance.Contrast(shirt_gray).enhance(shadow_intensity)
    
    texture_map_rgba = texture_map.convert("RGBA")

    # 5. The "Punchy" Blend
    # First, put the design on normally so colors stay 100%
    base_composite = Image.alpha_composite(shirt_bg.convert("RGBA"), design_layer)
    
    # Second, subtly multiply the shirt texture OVER the design area only
    # This keeps the design vibrant while adding fabric folds
    shadowed_design = ImageChops.multiply(design_layer, texture_map_rgba)
    
    # Blend the shadowed version with the clean version based on ink opacity
    final_design_layer = Image.blend(design_layer, shadowed_design, ink_opacity)

    return Image.alpha_composite(shirt_bg.convert("RGBA"), final_design_layer)

# --- Helper: Bounding Box ---
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

# --- Upload Section ---
design_files = st.file_uploader("📌 Upload Design PNGs", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
shirt_files = st.file_uploader("🎨 Upload Shirt Templates", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("🔄 Start Over"):
    st.session_state.design_names = {}
    st.rerun()

# --- Design Naming ---
if design_files:
    st.markdown("### ✏️ Name Your Designs")
    for i, file in enumerate(design_files):
        default_name = os.path.splitext(file.name)[0]
        st.session_state.design_names[file.name] = st.text_input(
            f"Name for {file.name}", value=default_name, key=f"name_{i}"
        )

# --- Batch Controls ---
if design_files:
    total_designs = len(design_files)
    batch_start = st.number_input("Start Design #", 1, total_designs, 1)
    batch_end = st.number_input("End Design #", 1, total_designs, min(20, total_designs))
    selected_batch = design_files[batch_start-1:batch_end]

# --- Live Preview ---
if design_files and shirt_files:
    st.markdown("---")
    st.markdown("### 👀 Realistic Preview")
    sel_design = st.selectbox("Preview Design", design_files, format_func=lambda x: x.name)
    sel_shirt = st.selectbox("Preview Template", shirt_files, format_func=lambda x: x.name)

    try:
        sel_design.seek(0); design_img = Image.open(sel_design).convert("RGBA")
        sel_shirt.seek(0); shirt_img = Image.open(sel_shirt).convert("RGBA")

        is_model = "model" in sel_shirt.name.lower()
        offset_pct = model_offset_pct if is_model else plain_offset_pct
        pad = model_padding_ratio if is_model else plain_padding_ratio

        bbox = get_shirt_bbox(shirt_img)
        if bbox:
            sx, sy, sw, sh = bbox
            scale = min(sw/design_img.width, sh/design_img.height, 1.0) * pad
            nw, nh = int(design_img.width * scale), int(design_img.height * scale)
            x, y = sx + (sw - nw)//2, sy + int(sh * offset_pct/100)
        else:
            nw, nh = design_img.size
            x, y = (shirt_img.width - nw)//2, (shirt_img.height - nh)//2

        preview = apply_realistic_blending(shirt_img, design_img, x, y, (nw, nh))
        st.image(preview, caption="Final Realistic Output (Simulated JPG)", use_container_width=True)
    except Exception as e:
        st.error(f"Preview Error: {e}")

# --- Batch Process ---
if st.button("🚀 Generate Realistic Batch"):
    if not (selected_batch and shirt_files):
        st.warning("Upload designs and templates first.")
    else:
        master_zip = io.BytesIO()
        with zipfile.ZipFile(master_zip, "w", zipfile.ZIP_DEFLATED) as master_zipf:
            progress_bar = st.progress(0)
            
            for idx, d_file in enumerate(selected_batch):
                g_name = st.session_state.design_names.get(d_file.name, "design")
                d_file.seek(0); d_img = Image.open(d_file).convert("RGBA")
                
                inner_zip_buffer = io.BytesIO()
                with zipfile.ZipFile(inner_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for s_file in shirt_files:
                        c_name = os.path.splitext(s_file.name)[0]
                        s_file.seek(0); s_img = Image.open(s_file).convert("RGBA")

                        is_m = "model" in s_file.name.lower()
                        off = model_offset_pct if is_m else plain_offset_pct
                        p_ratio = model_padding_ratio if is_m else plain_padding_ratio

                        bbox = get_shirt_bbox(s_img)
                        if bbox:
                            sx, sy, sw, sh = bbox
                            sc = min(sw/d_img.width, sh/d_img.height, 1.0) * p_ratio
                            nw, nh = int(d_img.width*sc), int(d_img.height*sc)
                            px, py = sx + (sw-nw)//2, sy + int(sh*off/100)
                        else:
                            nw, nh = d_img.size
                            px, py = (s_img.width-nw)//2, (s_img.height-nh)//2

                        # Apply Realistic Logic
                        final_img = apply_realistic_blending(s_img, d_img, px, py, (nw, nh))
                        
                        # Save to JPG
                        img_io = io.BytesIO()
                        final_img.convert("RGB").save(img_io, format='JPEG', quality=90, optimize=True)
                        zipf.writestr(f"{g_name}_{c_name}.jpg", img_io.getvalue())
                
                inner_zip_buffer.seek(0)
                master_zipf.writestr(f"{g_name}.zip", inner_zip_buffer.read())
                progress_bar.progress((idx + 1) / len(selected_batch))

        master_zip.seek(0)
        st.download_button("📦 Download All Mockups", master_zip, "mockups_realistic.zip", "application/zip")
