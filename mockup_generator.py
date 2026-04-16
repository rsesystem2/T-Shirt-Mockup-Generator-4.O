import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import zipfile
import io
import cv2
import os

st.set_page_config(page_title="Premium Mockup Studio", layout="wide")
st.title("👕 Premium Pro-Grade Mockup Generator")

st.markdown("""
**Professional Blending Engine:** Designed for high-fidelity, sellable results.
* **Bilateral Warp:** Physical pixel displacement along fabric folds.
* **High-Pass Texture:** Embosses cotton grain onto the ink for a 'printed' feel.
* **Color Guard:** Automatically protects ink vibrancy on black and white garments.
""")

# --- Sidebar: Advanced Settings ---
st.sidebar.header("📍 Master Controls")
p_pad = st.sidebar.slider("Padding - Plain", 0.1, 1.0, 0.45, 0.05)
m_pad = st.sidebar.slider("Padding - Model", 0.1, 1.0, 0.35, 0.05)
p_off = st.sidebar.slider("Vertical Offset - Plain (%)", -50, 100, 23, 1)
m_off = st.sidebar.slider("Vertical Offset - Model (%)", -50, 100, 38, 1)

st.sidebar.header("✨ Realism Engine")
warp_str = st.sidebar.slider("Warp/Fold Intensity", 0.0, 5.0, 1.8, 0.1)
grain_depth = st.sidebar.slider("Fabric Grain Depth", 0.0, 1.0, 0.3, 0.05)
ink_pop = st.sidebar.slider("Ink Saturation Boost", 0.5, 2.0, 1.05, 0.05)

# --- The Rendering Engine ---
def apply_premium_render(shirt_bg, design_img, x, y, size):
    # 1. High-Res Preparation & Sharpness
    design_res = design_img.resize(size, Image.Resampling.LANCZOS)
    if warp_str > 0:
        design_res = design_res.filter(ImageFilter.SHARPEN)

    # --- SAFETY: Zero settings return original ---
    if warp_str == 0 and grain_depth == 0:
        layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
        layer.paste(design_res, (x, y), design_res)
        return Image.alpha_composite(shirt_bg.convert("RGBA"), layer)

    # 2. BILATERAL DISPLACEMENT (The Professional Secret)
    shirt_np = np.array(shirt_bg.convert("RGB"))
    roi = shirt_np[y:y+size[1], x:x+size[0]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    # Bilateral Filter: Deletes photo noise but keeps big wrinkles
    smooth_roi = cv2.bilateralFilter(roi_gray, 15, 75, 75)
    
    if warp_str > 0:
        # Sobel Gradients for pixel-shifting
        gx = cv2.Sobel(smooth_roi, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth_roi, cv2.CV_32F, 0, 1, ksize=3)
        
        design_np = np.array(design_res)
        h, w = design_np.shape[:2]
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Warp with INTER_CUBIC for best text clarity
        mx = (xx + gx * (warp_str * 0.035)).astype(np.float32)
        my = (yy + gy * (warp_str * 0.035)).astype(np.float32)
        
        warped_np = cv2.remap(design_np, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        design_final = Image.fromarray(warped_np)
    else:
        design_final = design_res

    # 3. COLOR GUARD & LAYER ASSEMBLY
    layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
    layer.paste(design_final, (x, y), design_final)
    
    # Auto-Vibrancy for Darker Shirts
    avg_lume = np.array(Image.fromarray(roi).convert("L")).mean()
    lume_boost = ink_pop * (1.15 if avg_lume < 70 else 1.0)
    layer = ImageEnhance.Color(layer).enhance(lume_boost)

    # 4. SELECTIVE FABRIC EMBOSSING
    combined = Image.alpha_composite(shirt_bg.convert("RGBA"), layer)
    
    if grain_depth > 0:
        # Generate lighting overlay
        tex_mask = shirt_bg.convert("L").filter(ImageFilter.FIND_EDGES)
        tex_mask = ImageEnhance.Contrast(tex_mask).enhance(2.0).filter(ImageFilter.GaussianBlur(1))
        
        # Overlay blend allows design to catch light like real ink
        tex_overlay = ImageChops.overlay(layer, tex_mask.convert("RGBA"))
        result = Image.blend(combined, Image.alpha_composite(combined, tex_overlay), grain_depth * 0.5)
        
        # Subtle hard-shadow pass for deep creases
        if avg_lume < 100:
            shadow_pass = ImageChops.multiply(layer, ImageEnhance.Contrast(shirt_bg.convert("L")).enhance(1.5).convert("RGBA"))
            result = Image.blend(result, Image.alpha_composite(result, shadow_pass), grain_depth * 0.2)
            
        return result
    
    return combined

# --- Standard Utilities ---
def get_shirt_bbox(pil_image):
    img_cv = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (7, 7), 0), 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.boundingRect(max(contours, key=cv2.contourArea)) if contours else None

# --- Main App Logic ---
col1, col2 = st.columns([1, 1])

with col1:
    d_files = st.file_uploader("📌 Upload Graphics", type=["png", "jpg"], accept_multiple_files=True)
with col2:
    s_files = st.file_uploader("🎨 Upload Templates", type=["png", "jpg"], accept_multiple_files=True)

if d_files and s_files:
    if "ln" not in st.session_state: st.session_state.ln = {}
    for f in d_files: st.session_state.ln[f.name] = os.path.splitext(f.name)[0]
    
    s_d = st.selectbox("Current Graphic", d_files, format_func=lambda x: x.name)
    s_s = st.selectbox("Current Template", s_files, format_func=lambda x: x.name)

    try:
        s_d.seek(0); di = Image.open(s_d).convert("RGBA")
        s_s.seek(0); si = Image.open(s_s).convert("RGBA")

        is_m = "model" in s_s.name.lower()
        off = m_off if is_m else p_off
        pad = m_pad if is_m else p_pad

        bbox = get_shirt_bbox(si)
        if bbox:
            sc = min(bbox[2]/di.width, bbox[3]/di.height, 1.0) * pad
            nw, nh = int(di.width*sc), int(di.height*sc)
            px, py = bbox[0] + (bbox[2]-nw)//2, bbox[1] + int(bbox[3]*off/100)
            
            final_img = apply_premium_render(si, di, px, py, (nw, nh))
            st.image(final_img, caption="Professional High-Fidelity Render", use_container_width=True)
            
            # --- Batch Export ---
            if st.button("🚀 Batch Export Sellable Zip"):
                master_zip = io.BytesIO()
                with zipfile.ZipFile(master_zip, "w") as mzf:
                    for df in d_files:
                        df.seek(0); cur_d = Image.open(df).convert("RGBA")
                        inner = io.BytesIO()
                        with zipfile.ZipFile(inner, "w") as zf:
                            for sf in s_files:
                                sf.seek(0); cur_s = Image.open(sf).convert("RGBA")
                                b = get_shirt_bbox(cur_s)
                                if b:
                                    sc_b = min(b[2]/cur_d.width, b[3]/cur_d.height, 1.0) * (m_pad if "model" in sf.name.lower() else p_pad)
                                    nw_b, nh_b = int(cur_d.width*sc_b), int(cur_d.height*sc_b)
                                    px_b, py_b = b[0] + (b[2]-nw_b)//2, b[1] + int(b[3]*off/100)
                                    res = apply_premium_render(cur_s, cur_d, px_b, py_b, (nw_b, nh_b))
                                    buf = io.BytesIO(); res.convert("RGB").save(buf, format="JPEG", quality=98)
                                    zf.writestr(f"{sf.name}.jpg", buf.getvalue())
                        inner.seek(0); mzf.writestr(f"{st.session_state.ln[df.name]}.zip", inner.read())
                st.download_button("📂 Download All Premium Mockups", master_zip.getvalue(), "premium_pro_export.zip")
        else:
            st.warning("Detection failed. Please use a clean template background.")
    except Exception as e:
        st.error(f"Error: {e}")
