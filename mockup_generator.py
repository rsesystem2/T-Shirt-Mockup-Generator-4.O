import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import zipfile
import io
import cv2
import os

st.set_page_config(page_title="Premium Mockup Generator", layout="centered")
st.title("👕 Premium Shirt Mockup Generator")

st.markdown("""
**Pro Edition:** Uses OpenCV Displacement Mapping to warp designs around wrinkles. 
Best for dark shirts to keep colors vibrant while maintaining high realism.
""")

# --- Sidebar: Logic Controls ---
st.sidebar.header("📍 Placement Settings")
plain_padding_ratio = st.sidebar.slider("Padding Ratio – Plain", 0.1, 1.0, 0.45, 0.05)
model_padding_ratio = st.sidebar.slider("Padding Ratio – Model", 0.1, 1.0, 0.35, 0.05)
plain_offset_pct = st.sidebar.slider("Vertical Offset – Plain (%)", -50, 100, 23, 1)
model_offset_pct = st.sidebar.slider("Vertical Offset – Model (%)", -50, 100, 38, 1)

st.sidebar.header("✨ Premium Realism")
warp_intensity = st.sidebar.slider("Warp/Displacement Strength", 0.0, 5.0, 1.5, 0.5)
ink_vibrancy = st.sidebar.slider("Ink Vibrancy", 0.5, 1.5, 1.1, 0.05)
texture_depth = st.sidebar.slider("Fabric Texture Depth", 0.1, 1.0, 0.4, 0.05)

# --- Helper: Premium Displacement & Blending Engine ---
def apply_premium_mockup(shirt_bg, design_img, x, y, size):
    # 1. Resize Design
    design_res = design_img.resize(size, Image.Resampling.LANCZOS)
    
    # 2. DISPLACEMENT MAPPING (Warping design to wrinkles)
    # Convert shirt area to grayscale for analysis
    shirt_np = np.array(shirt_bg.convert("RGB"))
    roi = shirt_np[y:y+size[1], x:x+size[0]]
    
    # Ensure ROI and design match (handling edge cases)
    if roi.shape[0] != size[1] or roi.shape[1] != size[0]:
        # Fallback if bbox is slightly off-canvas
        return Image.alpha_composite(shirt_bg.convert("RGBA"), Image.new("RGBA", shirt_bg.size))

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    # Use Sobel gradients to find "slopes" of fabric folds
    grad_x = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=5)
    
    # Map the design pixels based on the shirt's physical folds
    design_np = np.array(design_res)
    h, w = design_np.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    
    map_x = (xx + grad_x * (warp_intensity * 0.1)).astype(np.float32)
    map_y = (yy + grad_y * (warp_intensity * 0.1)).astype(np.float32)
    
    warped_design = cv2.remap(design_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    design_final = Image.fromarray(warped_design).filter(ImageFilter.GaussianBlur(0.3))

    # 3. ADVANCED BLENDING (Keeping the blue vibrant)
    # Create the full design layer
    design_layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
    design_layer.paste(design_final, (x, y), design_final)
    
    # Enhance the vibrant colors of the original graphic
    design_layer = ImageEnhance.Color(design_layer).enhance(ink_vibrancy)

    # 4. SELECTIVE SHADOWS (Linear Burn style)
    # We only apply the darkest shirt wrinkles to the design
    shirt_gray_full = shirt_bg.convert("L")
    texture_mask = ImageEnhance.Contrast(shirt_gray_full).enhance(2.0)
    
    # Composite the vibrant design onto the shirt
    combined = Image.alpha_composite(shirt_bg.convert("RGBA"), design_layer)
    
    # Final touch: Gently blend the fabric texture back in so it's not a 'sticker'
    # but don't let it fade the blue!
    shadow_map = ImageChops.multiply(design_layer, texture_mask.convert("RGBA"))
    result = Image.blend(combined, Image.alpha_composite(combined, shadow_map), texture_depth)

    return result

# --- Helper: Bounding Box ---
def get_shirt_bbox(pil_image):
    img_cv = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.boundingRect(max(contours, key=cv2.contourArea)) if contours else None

# --- Upload & State Management ---
design_files = st.file_uploader("📌 Upload Design PNGs", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
shirt_files = st.file_uploader("🎨 Upload Shirt Templates", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if "design_names" not in st.session_state:
    st.session_state.design_names = {}

# --- Naming & Batching ---
if design_files:
    for i, file in enumerate(design_files):
        if file.name not in st.session_state.design_names:
            st.session_state.design_names[file.name] = os.path.splitext(file.name)[0]
    
    total = len(design_files)
    batch_start = st.number_input("Start #", 1, total, 1)
    batch_end = st.number_input("End #", 1, total, min(20, total))
    selected_batch = design_files[batch_start-1:batch_end]

# --- Preview ---
if design_files and shirt_files:
    st.markdown("---")
    sel_design = st.selectbox("Preview Design", design_files, format_func=lambda x: x.name)
    sel_shirt = st.selectbox("Preview Template", shirt_files, format_func=lambda x: x.name)

    try:
        sel_design.seek(0); d_img = Image.open(sel_design).convert("RGBA")
        sel_shirt.seek(0); s_img = Image.open(sel_shirt).convert("RGBA")

        is_m = "model" in sel_shirt.name.lower()
        off = model_offset_pct if is_m else plain_offset_pct
        pad = model_padding_ratio if is_m else plain_padding_ratio

        bbox = get_shirt_bbox(s_img)
        if bbox:
            sx, sy, sw, sh = bbox
            sc = min(sw/d_img.width, sh/d_img.height, 1.0) * pad
            nw, nh = int(d_img.width*sc), int(d_img.height*sc)
            px, py = sx + (sw-nw)//2, sy + int(sh*off/100)
            
            preview = apply_premium_mockup(s_img, d_img, px, py, (nw, nh))
            st.image(preview, caption="Premium Vibrant Mockup", use_container_width=True)
        else:
            st.warning("Could not detect shirt area. Check background color.")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Batch Generate ---
if st.button("🚀 Generate Premium Batch"):
    if selected_batch and shirt_files:
        master_zip = io.BytesIO()
        with zipfile.ZipFile(master_zip, "w", zipfile.ZIP_DEFLATED) as master_zipf:
            progress = st.progress(0)
            for idx, d_file in enumerate(selected_batch):
                d_file.seek(0); d_img = Image.open(d_file).convert("RGBA")
                inner_zip = io.BytesIO()
                with zipfile.ZipFile(inner_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    for s_file in shirt_files:
                        s_file.seek(0); s_img = Image.open(s_file).convert("RGBA")
                        bbox = get_shirt_bbox(s_img)
                        if bbox:
                            is_m = "model" in s_file.name.lower()
                            off = model_offset_pct if is_m else plain_offset_pct
                            sc = min(bbox[2]/d_img.width, bbox[3]/d_img.height, 1.0) * (model_padding_ratio if is_m else plain_padding_ratio)
                            nw, nh = int(d_img.width*sc), int(d_img.height*sc)
                            px, py = bbox[0] + (bbox[2]-nw)//2, bbox[1] + int(bbox[3]*off/100)
                            
                            res = apply_premium_mockup(s_img, d_img, px, py, (nw, nh))
                            img_io = io.BytesIO()
                            res.convert("RGB").save(img_io, format='JPEG', quality=95)
                            zf.writestr(f"{s_file.name}_{d_file.name}.jpg", img_io.getvalue())
                inner_zip.seek(0)
                master_zipf.writestr(f"{d_file.name}.zip", inner_zip.read())
                progress.progress((idx+1)/len(selected_batch))
        
        st.download_button("📦 Download Zip", master_zip.getvalue(), "premium_mockups.zip")
