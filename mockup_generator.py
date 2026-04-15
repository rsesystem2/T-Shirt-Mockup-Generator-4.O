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
**Pro Edition:** Features OpenCV Displacement Mapping. 
* **Warp Intensity:** Bends the design to follow shirt folds.
* **Fabric Texture:** Embosses cotton grain onto the ink.
* **Safety:** Set both to **0** to see your original 100% vibrant design.
""")

# --- Sidebar: Placement Controls ---
st.sidebar.header("📍 Placement Settings")
plain_padding_ratio = st.sidebar.slider("Padding Ratio – Plain", 0.1, 1.0, 0.45, 0.05)
model_padding_ratio = st.sidebar.slider("Padding Ratio – Model", 0.1, 1.0, 0.35, 0.05)
plain_offset_pct = st.sidebar.slider("Vertical Offset – Plain (%)", -50, 100, 23, 1)
model_offset_pct = st.sidebar.slider("Vertical Offset – Model (%)", -50, 100, 38, 1)

st.sidebar.header("✨ Premium Realism")
# Setting these to 0 will bypass all effects for 100% original look
warp_intensity = st.sidebar.slider("Warp/Displacement Strength", 0.0, 5.0, 1.5, 0.5)
texture_depth = st.sidebar.slider("Fabric Texture Depth", 0.0, 1.0, 0.3, 0.05)
ink_vibrancy = st.sidebar.slider("Ink Vibrancy (Saturation)", 0.5, 1.5, 1.0, 0.05)

# --- Helper: Premium Displacement & Blending Engine ---
def apply_premium_mockup(shirt_bg, design_img, x, y, size):
    # 1. Resize Design with High Quality
    design_res = design_img.resize(size, Image.Resampling.LANCZOS)
    
    # --- SAFETY SWITCH: If intensity is 0, return the clean original ---
    if warp_intensity == 0 and texture_depth == 0:
        design_layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
        design_layer.paste(design_res, (x, y), design_res)
        if ink_vibrancy != 1.0:
            design_layer = ImageEnhance.Color(design_layer).enhance(ink_vibrancy)
        return Image.alpha_composite(shirt_bg.convert("RGBA"), design_layer)

    # 2. DISPLACEMENT (Warping design to wrinkles)
    if warp_intensity > 0:
        shirt_np = np.array(shirt_bg.convert("RGB"))
        # Safeguard ROI coordinates
        y_end, x_end = min(y + size[1], shirt_np.shape[0]), min(x + size[2] if len(size)>2 else x + size[0], shirt_np.shape[1])
        roi = shirt_np[y:y+size[1], x:x+size[0]]
        
        if roi.shape[0] == size[1] and roi.shape[1] == size[0]:
            roi_pil = Image.fromarray(roi)
            # Boost brightness/contrast just to find hidden wrinkles on black fabric
            texture_finder = ImageEnhance.Brightness(roi_pil.convert("L")).enhance(2.5)
            texture_finder = ImageEnhance.Contrast(texture_finder).enhance(3.0)
            roi_gray = np.array(texture_finder)
            
            # Use Sobel to find slopes of fabric
            grad_x = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
            
            design_np = np.array(design_res)
            h, w = design_np.shape[:2]
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            
            map_x = (xx + grad_x * (warp_intensity * 0.05)).astype(np.float32)
            map_y = (yy + grad_y * (warp_intensity * 0.05)).astype(np.float32)
            
            warped_design_np = cv2.remap(design_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
            design_final = Image.fromarray(warped_design_np)
        else:
            design_final = design_res
    else:
        design_final = design_res

    # 3. Create Layer & Apply Vibrancy
    design_layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
    design_layer.paste(design_final, (x, y), design_final)
    if ink_vibrancy != 1.0:
        design_layer = ImageEnhance.Color(design_layer).enhance(ink_vibrancy)

    # 4. COMPOSITE
    combined = Image.alpha_composite(shirt_bg.convert("RGBA"), design_layer)
    
    # 5. SELECTIVE TEXTURE (Texture Depth)
    if texture_depth > 0:
        # Re-detect lighting texture to overlay as a subtle shadow
        shirt_gray_full = shirt_bg.convert("L")
        # Use a high-contrast version of the shirt as a 'shadow map'
        tex_map = ImageEnhance.Contrast(shirt_gray_full).enhance(2.0).convert("RGBA")
        
        # Multiply only the design layer with the texture map
        shadow_mask = ImageChops.multiply(design_layer, tex_map)
        
        # Blend the 'textured' version with the 'clean' version
        result = Image.blend(combined, Image.alpha_composite(combined, shadow_mask), texture_depth * 0.4)
        return result
    
    return combined

# --- Helper: Bounding Box ---
def get_shirt_bbox(pil_image):
    img_cv = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.boundingRect(max(contours, key=cv2.contourArea)) if contours else None

# --- Upload & State Management ---
design_files = st.file_uploader("📌 Upload Design Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
shirt_files = st.file_uploader("🎨 Upload Shirt Templates", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if "design_names" not in st.session_state:
    st.session_state.design_names = {}

if design_files:
    st.markdown("### ✏️ Design Names")
    for i, file in enumerate(design_files):
        st.session_state.design_names[file.name] = st.text_input(
            f"Label for {file.name}", value=os.path.splitext(file.name)[0], key=f"label_{i}"
        )
    
    total = len(design_files)
    batch_start = st.number_input("Batch Start", 1, total, 1)
    batch_end = st.number_input("Batch End", 1, total, min(20, total))
    selected_batch = design_files[batch_start-1:batch_end]

# --- Preview Section ---
if design_files and shirt_files:
    st.markdown("---")
    sel_design = st.selectbox("Design to Preview", design_files, format_func=lambda x: x.name)
    sel_shirt = st.selectbox("Shirt to Preview", shirt_files, format_func=lambda x: x.name)

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
            
            preview_img = apply_premium_mockup(s_img, d_img, px, py, (nw, nh))
            st.image(preview_img, caption="Final Mockup (Adjust Sliders to see Realism)", use_container_width=True)
        else:
            st.warning("Shirt area not detected. Ensure background is light/white.")
    except Exception as e:
        st.error(f"Render Error: {e}")

# --- Batch Processing ---
if st.button("🚀 Process Batch"):
    if selected_batch and shirt_files:
        master_zip = io.BytesIO()
        with zipfile.ZipFile(master_zip, "w", zipfile.ZIP_DEFLATED) as master_zf:
            pb = st.progress(0)
            for idx, df in enumerate(selected_batch):
                df.seek(0); d_img = Image.open(df).convert("RGBA")
                inner_zip = io.BytesIO()
                with zipfile.ZipFile(inner_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    for sf in shirt_files:
                        sf.seek(0); s_img = Image.open(sf).convert("RGBA")
                        bbox = get_shirt_bbox(s_img)
                        if bbox:
                            is_m = "model" in sf.name.lower()
                            off = model_offset_pct if is_m else plain_offset_pct
                            sc = min(bbox[2]/d_img.width, bbox[3]/d_img.height, 1.0) * (model_padding_ratio if is_m else plain_padding_ratio)
                            nw, nh = int(d_img.width*sc), int(d_img.height*sc)
                            px, py = bbox[0] + (bbox[2]-nw)//2, bbox[1] + int(bbox[3]*off/100)
                            
                            res = apply_premium_mockup(s_img, d_img, px, py, (nw, nh))
                            img_io = io.BytesIO()
                            res.convert("RGB").save(img_io, format='JPEG', quality=95)
                            zf.writestr(f"{sf.name}_{df.name}.jpg", img_io.getvalue())
                
                inner_zip.seek(0)
                master_zf.writestr(f"{st.session_state.design_names.get(df.name, 'graphic')}.zip", inner_zip.read())
                pb.progress((idx+1)/len(selected_batch))
        
        st.download_button("📦 Download Results", master_zip.getvalue(), "mockups.zip")
