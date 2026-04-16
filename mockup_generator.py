import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import zipfile
import io
import cv2
import os

st.set_page_config(page_title="Premium Mockup Studio", layout="wide")
st.title("👕 Premium Pro-Grade Mockup Generator")

# --- Sidebar: Master Controls ---
st.sidebar.header("📍 Master Controls")
p_pad = st.sidebar.slider("Padding - Plain", 0.1, 1.0, 0.45, 0.05)
m_pad = st.sidebar.slider("Padding - Model", 0.1, 1.0, 0.35, 0.05)
p_off = st.sidebar.slider("Vertical Offset - Plain (%)", -50, 100, 32, 1)
m_off = st.sidebar.slider("Vertical Offset - Model (%)", -50, 100, 38, 1)

st.sidebar.header("✨ Realism Engine (Anti-Shatter)")
warp_str = st.sidebar.slider("Warp/Fold Intensity", 0.0, 5.0, 1.8, 0.1)
grain_depth = st.sidebar.slider("Fabric Grain Depth", 0.0, 1.0, 0.3, 0.05)
ink_vibrancy = st.sidebar.slider("Ink Vibrancy (1.0 = Original)", 0.5, 1.5, 1.0, 0.05)

# --- The "Anti-Shatter" Rendering Engine ---
def apply_premium_render(shirt_bg, design_img, x, y, size):
    # 1. High-Res Prep
    design_res = design_img.resize(size, Image.Resampling.LANCZOS)
    
    # --- SAFETY SWITCH ---
    if warp_str == 0 and grain_depth == 0:
        layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
        layer.paste(design_res, (x, y), design_res)
        return Image.alpha_composite(shirt_bg.convert("RGBA"), layer)

    # 2. HD-BILATERAL WARP (Stops the shattered effect)
    shirt_np = np.array(shirt_bg.convert("RGB"))
    roi = shirt_np[y:y+size[1], x:x+size[0]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    # Bilateral filter deletes photo noise but keeps big wrinkles sharp
    # This is the secret to stopping the pixelated distortion
    smooth_roi = cv2.bilateralFilter(roi_gray, 15, 75, 75)
    
    if warp_str > 0:
        # Calculate gradients based on the SMOOTHED roi
        gx = cv2.Sobel(smooth_roi, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth_roi, cv2.CV_32F, 0, 1, ksize=3)
        
        design_np = np.array(design_res)
        h, w = design_np.shape[:2]
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Shift pixels smoothly
        map_x = (xx + gx * (warp_str * 0.035)).astype(np.float32)
        map_y = (yy + gy * (warp_str * 0.035)).astype(np.float32)
        
        # Using INTER_CUBIC for premium text clarity
        warped_np = cv2.remap(design_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        design_final = Image.fromarray(warped_np)
    else:
        design_final = design_res

    # 3. ASSEMBLY & COLOR PROTECT
    layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
    layer.paste(design_final, (x, y), design_final)
    
    if ink_vibrancy != 1.0:
        layer = ImageEnhance.Color(layer).enhance(ink_vibrancy)

    combined = Image.alpha_composite(shirt_bg.convert("RGBA"), layer)

    # 4. SELECTIVE FABRIC LIGHTING (Soft-Light)
    if grain_depth > 0:
        # Soft-Light blend lets the texture push through without shifting the blue color
        textured_overlay = ImageChops.soft_light(combined, layer.convert("RGBA"))
        result = Image.blend(combined, textured_overlay, grain_depth)
        return result
    
    return combined

# --- Utilities ---
def get_shirt_bbox(pil_image):
    img_cv = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (7, 7), 0), 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.boundingRect(max(contours, key=cv2.contourArea)) if contours else None

# --- Main App ---
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
        bbox = get_shirt_bbox(si)

        if bbox:
            off = m_off if "model" in ss.name.lower() else p_off
            pad = m_pad if "model" in ss.name.lower() else p_pad
            sc = min(bbox[2]/di.width, bbox[3]/di.height, 1.0) * pad
            nw, nh = int(di.width*sc), int(di.height*sc)
            px, py = bbox[0] + (bbox[2]-nw)//2, bbox[1] + int(bbox[3]*off/100)
            
            # FINAL PREMIUM RENDER
            final_img = apply_premium_render(si, di, px, py, (nw, nh))
            st.image(final_img, caption="Professional Clean Render", use_container_width=True)
            
            if st.button("🚀 Export All Mockups"):
                m_zip = io.BytesIO()
                with zipfile.ZipFile(m_zip, "w") as mzf:
                    for df in d_files:
                        df.seek(0); c_di = Image.open(df).convert("RGBA")
                        inner = io.BytesIO()
                        with zipfile.ZipFile(inner, "w") as zf:
                            for sf in shirt_files:
                                sf.seek(0); c_si = Image.open(sf).convert("RGBA")
                                b = get_shirt_bbox(c_si)
                                if b:
                                    sc_b = min(b[2]/c_di.width, b[3]/c_di.height, 1.0) * pad
                                    nw_b, nh_b = int(c_di.width*sc_b), int(c_di.height*sc_b)
                                    px_b, py_b = b[0] + (b[2]-nw_b)//2, b[1] + int(b[3]*off/100)
                                    res = apply_premium_render(c_si, c_di, px_b, py_b, (nw_b, nh_b))
                                    buf = io.BytesIO(); res.convert("RGB").save(buf, format="JPEG", quality=98)
                                    zf.writestr(f"{sf.name}.jpg", buf.getvalue())
                        inner.seek(0); mzf.writestr(f"{st.session_state.ln[df.name]}.zip", inner.read())
                st.download_button("📂 Download Premium Exports", m_zip.getvalue(), "premium_pro_mockups.zip")
    except Exception as e:
        st.error(f"Error: {e}")
