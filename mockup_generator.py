import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import zipfile
import io
import cv2
import os

st.set_page_config(page_title="Gwest Dept Pro Mockup", layout="centered")
st.title("👕 Premium Pro-Grade Mockup Generator")

st.markdown("""
**Professional Blending Engine:** Designed to match high-end tools like Mockey.AI.
* **Warping:** Real physical bending around fabric folds.
* **Global Lighting:** Design reacts to highlights and shadows.
* **Vibrancy:** Maintains 100% color accuracy for electric blues and skin tones.
""")

# --- Sidebar: Logic Controls ---
st.sidebar.header("📍 Placement Settings")
plain_padding_ratio = st.sidebar.slider("Padding Ratio – Plain", 0.1, 1.0, 0.45, 0.05)
model_padding_ratio = st.sidebar.slider("Padding Ratio – Model", 0.1, 1.0, 0.35, 0.05)
plain_offset_pct = st.sidebar.slider("Vertical Offset – Plain (%)", -50, 100, 23, 1)
model_offset_pct = st.sidebar.slider("Vertical Offset – Model (%)", -50, 100, 38, 1)

st.sidebar.header("✨ Mockey-Style Realism")
warp_intensity = st.sidebar.slider("Warp/Fold Strength", 0.0, 5.0, 1.8, 0.2)
texture_vibrancy = st.sidebar.slider("Fabric Lighting Depth", 0.0, 1.0, 0.35, 0.05)
ink_saturation = st.sidebar.slider("Color Saturation", 0.5, 1.5, 1.05, 0.05)

# --- Helper: The Premium Logic ---
def apply_mockey_premium_blend(shirt_bg, design_img, x, y, size):
    # 1. High-Precision Resizing
    design_res = design_img.resize(size, Image.Resampling.LANCZOS)
    
    # 2. CREATE BILATERAL HEIGHT MAP
    # This prevents the 'shattered' look and follows big wrinkles only
    shirt_np = np.array(shirt_bg.convert("RGB"))
    roi = shirt_np[y:y+size[1], x:x+size[0]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    # Smooth out noise but keep wrinkle edges sharp
    smooth_map = cv2.bilateralFilter(roi_gray, 11, 60, 60)
    
    # 3. ADVANCED DISPLACEMENT (Warping)
    if warp_intensity > 0:
        grad_x = cv2.Sobel(smooth_map, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smooth_map, cv2.CV_32F, 0, 1, ksize=3)
        
        design_np = np.array(design_res)
        h, w = design_np.shape[:2]
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Calculate displacement based on shirt physics
        map_x = (xx + grad_x * (warp_intensity * 0.04)).astype(np.float32)
        map_y = (yy + grad_y * (warp_intensity * 0.04)).astype(np.float32)
        
        # Warp with CUBIC for sharpness
        warped_np = cv2.remap(design_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        design_final = Image.fromarray(warped_np).filter(ImageFilter.SHARPEN)
    else:
        design_final = design_res

    # 4. COLOR & SATURATION PROTECTOR
    design_layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
    design_layer.paste(design_final, (x, y), design_final)
    design_layer = ImageEnhance.Color(design_layer).enhance(ink_saturation)

    # 5. SELECTIVE SOFT-LIGHT BLENDING (The 'Premium' Step)
    # This makes the design look 'inside' the cotton, not on top.
    combined = Image.alpha_composite(shirt_bg.convert("RGBA"), design_layer)
    
    if texture_vibrancy > 0:
        # Generate Lighting Map (Highlights + Shadows)
        lighting_map = shirt_bg.convert("L")
        lighting_map = ImageEnhance.Contrast(lighting_map).enhance(2.0)
        
        # 'Soft Light' blend maintains vibrant blue while adding fabric texture
        texture_overlay = ImageChops.soft_light(combined, design_layer.convert("RGBA"))
        
        # Blend the result based on the texture slider
        result = Image.blend(combined, texture_overlay, texture_vibrancy)
        
        # Final subtle Multiply for deep creases ONLY
        crease_mask = ImageEnhance.Contrast(lighting_map).enhance(2.0).convert("RGBA")
        shadow_final = ImageChops.multiply(design_layer, crease_mask)
        result = Image.blend(result, Image.alpha_composite(result, shadow_final), texture_vibrancy * 0.3)
        
        return result
    
    return combined

# --- Utilities: Bounding Box & Batching ---
def get_shirt_bbox(pil_image):
    img_cv = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (7, 7), 0), 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.boundingRect(max(contours, key=cv2.contourArea)) if contours else None

# --- Main Interface ---
design_files = st.file_uploader("📌 Upload Graphics", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
shirt_files = st.file_uploader("🎨 Upload Templates", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if design_files and shirt_files:
    if "labels" not in st.session_state: st.session_state.labels = {}
    for f in design_files: st.session_state.labels[f.name] = os.path.splitext(f.name)[0]
    
    sel_d = st.selectbox("Design Preview", design_files, format_func=lambda x: x.name)
    sel_s = st.selectbox("Shirt Preview", shirt_files, format_func=lambda x: x.name)

    try:
        sel_d.seek(0); d_img = Image.open(sel_d).convert("RGBA")
        sel_s.seek(0); s_img = Image.open(sel_s).convert("RGBA")

        is_m = "model" in sel_s.name.lower()
        off = model_offset_pct if is_m else plain_offset_pct
        pad = model_padding_ratio if is_m else plain_padding_ratio

        bbox = get_shirt_bbox(s_img)
        if bbox:
            sc = min(bbox[2]/d_img.width, bbox[3]/d_img.height, 1.0) * pad
            nw, nh = int(d_img.width*sc), int(d_img.height*sc)
            px, py = bbox[0] + (bbox[2]-nw)//2, bbox[1] + int(bbox[3]*off/100)
            
            # RUN PREMIUM RENDER
            preview = apply_mockey_premium_blend(s_img, d_img, px, py, (nw, nh))
            st.image(preview, caption="Premium Rendered Output", use_container_width=True)
            
            if st.button("🚀 Export All (Premium Batch)"):
                master_zip = io.BytesIO()
                with zipfile.ZipFile(master_zip, "w") as mzf:
                    for df in design_files:
                        df.seek(0); di = Image.open(df).convert("RGBA")
                        inner = io.BytesIO()
                        with zipfile.ZipFile(inner, "w") as zf:
                            for sf in shirt_files:
                                sf.seek(0); si = Image.open(sf).convert("RGBA")
                                b = get_shirt_bbox(si)
                                if b:
                                    sc_b = min(b[2]/di.width, b[3]/di.height, 1.0) * (model_padding_ratio if "model" in sf.name.lower() else plain_padding_ratio)
                                    nw_b, nh_b = int(di.width*sc_b), int(di.height*sc_b)
                                    px_b, py_b = b[0] + (b[2]-nw_b)//2, b[1] + int(b[3]*off/100)
                                    res = apply_mockey_premium_blend(si, di, px_b, py_b, (nw_b, nh_b))
                                    buf = io.BytesIO(); res.convert("RGB").save(buf, format="JPEG", quality=95)
                                    zf.writestr(f"{sf.name}.jpg", buf.getvalue())
                        inner.seek(0); mzf.writestr(f"{st.session_state.labels[df.name]}.zip", inner.read())
                st.download_button("📂 Download Premium Mockups", master_zip.getvalue(), "premium_export.zip")
    except Exception as e:
        st.error(f"Render Error: {e}")
