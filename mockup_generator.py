import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import zipfile
import io
import cv2
import os

st.set_page_config(page_title="Premium POD Mockup Engine", layout="wide")
st.title("👕 Premium POD Mockup Engine (Color-Safe)")

st.markdown("""
**Professional Grade Rendering:**
* **Zero Color Shift:** Uses Soft-Light blending to protect original graphic hex codes.
* **Subtle-Fold Warp:** Tuned to detect wrinkles even on smooth, light-colored shirts.
* **Safety Switch:** Set realism sliders to **0** for a raw, unprocessed original file paste.
""")

# --- Sidebar: Master Controls ---
st.sidebar.header("📍 Master Controls")
p_pad = st.sidebar.slider("Padding - Plain", 0.1, 1.0, 0.45, 0.05)
m_pad = st.sidebar.slider("Padding - Model", 0.1, 1.0, 0.35, 0.05)
p_off = st.sidebar.slider("Vertical Offset - Plain (%)", -50, 100, 23, 1)
m_off = st.sidebar.slider("Vertical Offset - Model (%)", -50, 100, 38, 1)

st.sidebar.header("✨ Realism Engine (Color-Safe)")
warp_str = st.sidebar.slider("Warp/Fold Intensity", 0.0, 5.0, 1.8, 0.1)
grain_depth = st.sidebar.slider("Fabric Grain Depth", 0.0, 1.0, 0.25, 0.05)
# Note: Saturation is set to 1.0 by default to ensure no color change.
ink_vibrancy = st.sidebar.slider("Ink Vibrancy (1.0 = Original)", 0.5, 1.5, 1.0, 0.05)

# --- The "Color-Safe" Premium Engine ---
def apply_premium_render(shirt_bg, design_img, x, y, size):
    # 1. High-Res Preparation
    design_res = design_img.resize(size, Image.Resampling.LANCZOS)
    
    # --- HARD ZERO CHECK: Bypass all processing if sliders are 0 ---
    if warp_str == 0 and grain_depth == 0 and ink_vibrancy == 1.0:
        layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
        layer.paste(design_res, (x, y), design_res)
        return Image.alpha_composite(shirt_bg.convert("RGBA"), layer)

    # 2. HD-BILATERAL WARP (Tuned for smooth & dark shirts)
    shirt_np = np.array(shirt_bg.convert("RGB"))
    roi = shirt_np[y:y+size[1], x:x+size[0]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    # Bilateral smoothing: tight sigma to see subtle folds on green/white shirts
    smooth_roi = cv2.bilateralFilter(roi_gray, 9, 30, 75)
    
    if warp_str > 0:
        # Sobel Kernel 5 for smoother bending on high-res templates
        gx = cv2.Sobel(smooth_roi, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(smooth_roi, cv2.CV_32F, 0, 1, ksize=5)
        
        design_np = np.array(design_res)
        h, w = design_np.shape[:2]
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Warp logic - map intensity tuned for realistic displacement
        mx = (xx + gx * (warp_str * 0.05)).astype(np.float32)
        my = (yy + gy * (warp_str * 0.05)).astype(np.float32)
        
        warped_np = cv2.remap(design_np, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        design_final = Image.fromarray(warped_np).filter(ImageFilter.SHARPEN)
    else:
        design_final = design_res

    # 3. COLOR-SAFE ASSEMBLY
    layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
    layer.paste(design_final, (x, y), design_final)
    
    # Optional Vibrancy (Leave at 1.0 for exact original color)
    if ink_vibrancy != 1.0:
        layer = ImageEnhance.Color(layer).enhance(ink_vibrancy)

    # Put graphic down FIRST (protects the base color)
    combined = Image.alpha_composite(shirt_bg.convert("RGBA"), layer)

    # 4. SELECTIVE LIGHTING PASS (Adds folds without changing Hue)
    if grain_depth > 0:
        # Create a lighting map (Luminance only)
        l_map = shirt_bg.convert("L")
        
        # Use Soft-Light blending: it brightens peaks and darkens valleys 
        # but is mathematically incapable of shifting the 'Hue' of your blue.
        textured_overlay = ImageChops.soft_light(combined, layer)
        
        # Blend based on user depth
        result = Image.blend(combined, textured_overlay, grain_depth)
        
        # Optional deep crease darkening (Subtle multiply)
        if np.array(l_map).mean() < 128: # If shirt is dark
            crease_pass = ImageChops.multiply(layer, ImageEnhance.Contrast(l_map).enhance(2.0).convert("RGBA"))
            result = Image.blend(result, Image.alpha_composite(result, crease_pass), grain_depth * 0.2)
            
        return result
    
    return combined

# --- Utilities ---
def get_shirt_bbox(pil_image):
    img_cv = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (7, 7), 0), 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.boundingRect(max(contours, key=cv2.contourArea)) if contours else None

# --- Main Logic ---
d_files = st.file_uploader("📌 POD Graphics (PNG)", type=["png"], accept_multiple_files=True)
s_files = st.file_uploader("🎨 Templates", type=["png", "jpg"], accept_multiple_files=True)

if d_files and s_files:
    if "ln" not in st.session_state: st.session_state.ln = {}
    for f in d_files: st.session_state.ln[f.name] = os.path.splitext(f.name)[0]
    
    sd = st.selectbox("Active Graphic", d_files, format_func=lambda x: x.name)
    ss = st.selectbox("Active Template", s_files, format_func=lambda x: x.name)

    try:
        sd.seek(0); di = Image.open(sd).convert("RGBA")
        ss.seek(0); si = Image.open(ss).convert("RGBA")

        is_m = "model" in ss.name.lower()
        off = m_off if is_m else p_off
        pad = m_pad if is_m else p_pad

        bbox = get_shirt_bbox(si)
        if bbox:
            sc = min(bbox[2]/di.width, bbox[3]/di.height, 1.0) * pad
            nw, nh = int(di.width*sc), int(di.height*sc)
            px, py = bbox[0] + (bbox[2]-nw)//2, bbox[1] + int(bbox[3]*off/100)
            
            # FINAL RENDER
            final_img = apply_premium_render(si, di, px, py, (nw, nh))
            st.image(final_img, caption="Premium Render (Original Color Protected)", use_container_width=True)
            
            if st.button("🚀 Export Premium Zip"):
                m_zip = io.BytesIO()
                with zipfile.ZipFile(m_zip, "w") as mzf:
                    for df in d_files:
                        df.seek(0); c_di = Image.open(df).convert("RGBA")
                        inner = io.BytesIO()
                        with zipfile.ZipFile(inner, "w") as zf:
                            for sf in s_files:
                                sf.seek(0); c_si = Image.open(sf).convert("RGBA")
                                b = get_shirt_bbox(c_si)
                                if b:
                                    sc_b = min(b[2]/c_di.width, b[3]/c_di.height, 1.0) * (m_pad if "model" in sf.name.lower() else p_pad)
                                    nw_b, nh_b = int(c_di.width*sc_b), int(c_di.height*sc_b)
                                    px_b, py_b = b[0] + (b[2]-nw_b)//2, b[1] + int(b[3]*off/100)
                                    res = apply_premium_render(c_si, c_di, px_b, py_b, (nw_b, nh_b))
                                    buf = io.BytesIO(); res.convert("RGB").save(buf, format="JPEG", quality=98)
                                    zf.writestr(f"{sf.name}.jpg", buf.getvalue())
                        inner.seek(0); mzf.writestr(f"{st.session_state.ln[df.name]}.zip", inner.read())
                st.download_button("📂 Download Professional Mockups", m_zip.getvalue(), "pro_mockups.zip")
    except Exception as e:
        st.error(f"Error: {e}")
