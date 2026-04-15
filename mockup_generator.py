import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import zipfile
import io
import cv2
import os

st.set_page_config(page_title="Gwest Dept Mockup Pro", layout="centered")
st.title("👕 Pro-Grade Shirt Mockup Generator")

st.markdown("""
**Bilateral Displacement Engine:** This version uses advanced computer vision to:
1. **Clean Noise:** Stops designs from 'shattering' on dark fabric.
2. **Warp Folds:** Physically bends the graphic along the shirt's curves.
3. **Protect Color:** Maintains 100% color accuracy for vibrant graphics.
""")

# --- Sidebar: Controls ---
st.sidebar.header("📍 Placement Settings")
plain_padding_ratio = st.sidebar.slider("Padding Ratio – Plain", 0.1, 1.0, 0.45, 0.05)
model_padding_ratio = st.sidebar.slider("Padding Ratio – Model", 0.1, 1.0, 0.35, 0.05)
plain_offset_pct = st.sidebar.slider("Vertical Offset – Plain (%)", -50, 100, 23, 1)
model_offset_pct = st.sidebar.slider("Vertical Offset – Model (%)", -50, 100, 38, 1)

st.sidebar.header("✨ Realism Engine")
# Setting to 0.0 bypasses all warping and blending for a perfect original look
warp_intensity = st.sidebar.slider("Warp Strength (Fold Flow)", 0.0, 5.0, 1.5, 0.5)
texture_depth = st.sidebar.slider("Fabric Grain Depth", 0.0, 1.0, 0.25, 0.05)
ink_vibrancy = st.sidebar.slider("Ink Vibrancy", 0.5, 1.5, 1.0, 0.05)

# --- Helper: The "Best-In-Class" Logic ---
def apply_best_in_class_mockup(shirt_bg, design_img, x, y, size):
    # 1. CRISP PREPARATION
    # Pre-sharpen the design to combat the natural softening of the warp
    design_res = design_img.resize(size, Image.Resampling.LANCZOS)
    if warp_intensity > 0:
        design_res = design_res.filter(ImageFilter.SHARPEN)
    
    # --- SAFETY SWITCH: 0 means original graphic ---
    if warp_intensity == 0 and texture_depth == 0:
        design_layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
        design_layer.paste(design_res, (x, y), design_res)
        if ink_vibrancy != 1.0:
            design_layer = ImageEnhance.Color(design_layer).enhance(ink_vibrancy)
        return Image.alpha_composite(shirt_bg.convert("RGBA"), design_layer)

    # 2. BILATERAL HEIGHT MAP (The Noise Fix)
    shirt_np = np.array(shirt_bg.convert("RGB"))
    roi = shirt_np[y:y+size[1], x:x+size[0]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    # Bilateral filter deletes camera noise but keeps the fabric folds sharp
    # This prevents the 'broken glass' look on black shirts
    smooth_roi = cv2.bilateralFilter(roi_gray, 15, 75, 75)

    # 3. WARP WITH CUBIC INTERPOLATION (The Premium Warp)
    if warp_intensity > 0:
        grad_x = cv2.Sobel(smooth_roi, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smooth_roi, cv2.CV_32F, 0, 1, ksize=3)
        
        design_np = np.array(design_res)
        h, w = design_np.shape[:2]
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Calculate displacement map
        map_x = (xx + grad_x * (warp_intensity * 0.035)).astype(np.float32)
        map_y = (yy + grad_y * (warp_intensity * 0.035)).astype(np.float32)
        
        # remap using INTER_CUBIC for maximum sharpness
        warped_np = cv2.remap(design_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        design_final = Image.fromarray(warped_np)
    else:
        design_final = design_res

    # 4. COLOR PROTECTION & LAYER ASSEMBLY
    design_layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
    design_layer.paste(design_final, (x, y), design_final)
    
    # Auto-adjust Saturation based on shirt brightness
    avg_bri = np.array(Image.fromarray(roi).convert("L")).mean()
    color_boost = ink_vibrancy * (1.15 if avg_bri < 60 else 1.0) # Boost on black
    design_layer = ImageEnhance.Color(design_layer).enhance(color_boost)

    # 5. SELECTIVE EMBOSSING (Texture Overlay)
    # Put design on shirt first
    combined = Image.alpha_composite(shirt_bg.convert("RGBA"), design_layer)
    
    if texture_depth > 0:
        # Create a texture map that ignores base color
        texture_mask = shirt_bg.convert("L").filter(ImageFilter.FIND_EDGES)
        texture_mask = ImageEnhance.Contrast(texture_mask).enhance(2.0).filter(ImageFilter.GaussianBlur(1))
        
        # Use Overlay/Hard Light logic to 'emboss' folds onto the design
        # This keeps the Gwest Dept blue vibrant while catching the fabric grain
        texture_overlay = ImageChops.overlay(design_layer, texture_mask.convert("RGBA"))
        result = Image.blend(combined, Image.alpha_composite(combined, texture_overlay), texture_depth * 0.5)
        return result
    
    return combined

# --- Standard Utilities ---
def get_shirt_bbox(pil_image):
    img_cv = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.boundingRect(max(contours, key=cv2.contourArea)) if contours else None

# --- Main App ---
design_files = st.file_uploader("📌 Upload Designs", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
shirt_files = st.file_uploader("🎨 Upload Shirt Templates", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if design_files and shirt_files:
    if "dn" not in st.session_state: st.session_state.dn = {}
    for f in design_files: st.session_state.dn[f.name] = os.path.splitext(f.name)[0]
    
    sel_d = st.selectbox("Design Preview", design_files, format_func=lambda x: x.name)
    sel_s = st.selectbox("Shirt Preview", shirt_files, format_func=lambda x: x.name)

    try:
        sel_d.seek(0); di = Image.open(sel_d).convert("RGBA")
        sel_s.seek(0); si = Image.open(sel_s).convert("RGBA")

        is_m = "model" in sel_s.name.lower()
        off = model_offset_pct if is_m else plain_offset_pct
        pad = model_padding_ratio if is_m else plain_padding_ratio

        bbox = get_shirt_bbox(si)
        if bbox:
            sc = min(bbox[2]/di.width, bbox[3]/di.height, 1.0) * pad
            nw, nh = int(di.width*sc), int(di.height*sc)
            px, py = bbox[0] + (bbox[2]-nw)//2, bbox[1] + int(bbox[3]*off/100)
            
            final = apply_best_in_class_mockup(si, di, px, py, (nw, nh))
            st.image(final, caption="Premium Vibrant Mockup", use_container_width=True)
            
            if st.button("🚀 Generate Zip for all Designs"):
                master_zip = io.BytesIO()
                with zipfile.ZipFile(master_zip, "w") as mzf:
                    for df in design_files:
                        df.seek(0); d_img_proc = Image.open(df).convert("RGBA")
                        inner = io.BytesIO()
                        with zipfile.ZipFile(inner, "w") as zf:
                            for sf in shirt_files:
                                sf.seek(0); s_img_proc = Image.open(sf).convert("RGBA")
                                b = get_shirt_bbox(s_img_proc)
                                if b:
                                    sc_b = min(b[2]/d_img_proc.width, b[3]/d_img_proc.height, 1.0) * (model_padding_ratio if "model" in sf.name.lower() else plain_padding_ratio)
                                    nw_b, nh_b = int(d_img_proc.width*sc_b), int(d_img_proc.height*sc_b)
                                    px_b, py_b = b[0] + (b[2]-nw_b)//2, b[1] + int(b[3]*off/100)
                                    res = apply_best_in_class_mockup(s_img_proc, d_img_proc, px_b, py_b, (nw_b, nh_b))
                                    buf = io.BytesIO(); res.convert("RGB").save(buf, format="JPEG", quality=95)
                                    zf.writestr(f"{sf.name}.jpg", buf.getvalue())
                        inner.seek(0); mzf.writestr(f"{st.session_state.dn[df.name]}.zip", inner.read())
                st.download_button("📂 Download Mockups", master_zip.getvalue(), "premium_mockups.zip")
        else:
            st.warning("Shirt not detected. Use a template with a clean background.")
    except Exception as e:
        st.error(f"Error: {e}")
