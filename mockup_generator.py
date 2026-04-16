import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import zipfile
import io
import cv2
import os

st.set_page_config(page_title="Premium Mockup Studio", layout="wide")
st.title("👕 Premium Pro-Grade Mockup Generator")

# --- Sidebar: Professional Realism Settings ---
st.sidebar.header("✨ Realism Engine")
st.sidebar.markdown("Adjust these for the 'Mockey.ai' look.")
warp_strength = st.sidebar.slider("Warp/Fold Intensity", 0.0, 5.0, 1.8, 0.1)
lighting_depth = st.sidebar.slider("Fabric Lighting Depth", 0.0, 1.0, 0.35, 0.05)
ink_vibrancy = st.sidebar.slider("Ink Vibrancy Boost", 0.5, 1.5, 1.05, 0.05)

st.sidebar.header("📍 Placement Controls")
p_pad = st.sidebar.slider("Padding - Plain", 0.1, 1.0, 0.45, 0.05)
m_pad = st.sidebar.slider("Padding - Model", 0.1, 1.0, 0.35, 0.05)
p_off = st.sidebar.slider("Vertical Offset - Plain (%)", -50, 100, 23, 1)
m_off = st.sidebar.slider("Vertical Offset - Model (%)", -50, 100, 38, 1)

# --- The Premium Engine ---
def apply_premium_render(shirt_bg, design_img, x, y, size):
    # 1. High-Res Prep with Anti-Blur
    design_res = design_img.resize(size, Image.Resampling.LANCZOS)
    if warp_strength > 0:
        design_res = design_res.filter(ImageFilter.SHARPEN)

    # --- SAFETY: Zero settings return 100% original color ---
    if warp_strength == 0 and lighting_depth == 0:
        layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
        layer.paste(design_res, (x, y), design_res)
        return Image.alpha_composite(shirt_bg.convert("RGBA"), layer)

    # 2. CREATE BILATERAL HEIGHT MAP (Professional Standard)
    shirt_np = np.array(shirt_bg.convert("RGB"))
    roi = shirt_np[y:y+size[1], x:x+size[0]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    # Bilateral smoothing: Keeps fabric folds sharp but deletes pixel noise
    smooth_map = cv2.bilateralFilter(roi_gray, 15, 75, 75)

    # 3. PHYSICAL WARPING (Displacement)
    if warp_strength > 0:
        grad_x = cv2.Sobel(smooth_map, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smooth_map, cv2.CV_32F, 0, 1, ksize=3)
        
        design_np = np.array(design_res)
        h, w = design_np.shape[:2]
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Shift pixels along the 'slopes' of the fabric folds
        map_x = (xx + grad_x * (warp_strength * 0.04)).astype(np.float32)
        map_y = (yy + grad_y * (warp_strength * 0.04)).astype(np.float32)
        
        # Remap using INTER_CUBIC for premium sharpness
        warped_np = cv2.remap(design_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        design_final = Image.fromarray(warped_np)
    else:
        design_final = design_res

    # 4. COLOR GUARD & ASSEMBLY
    layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
    layer.paste(design_final, (x, y), design_final)
    
    # Dynamic Saturation: Auto-boosts colors on dark shirts to prevent fading
    avg_lume = np.array(Image.fromarray(roi).convert("L")).mean()
    lume_boost = ink_vibrancy * (1.15 if avg_lume < 65 else 1.0)
    layer = ImageEnhance.Color(layer).enhance(lume_boost)

    # 5. SELECTIVE BLENDING (Soft-Light + Overlay)
    combined = Image.alpha_composite(shirt_bg.convert("RGBA"), layer)
    
    if lighting_depth > 0:
        # Generate Lighting Map from shirt's brightness
        lighting_map = shirt_bg.convert("L")
        lighting_map = ImageEnhance.Contrast(lighting_map).enhance(2.0).convert("RGBA")
        
        # 'Soft Light' blend maintains vibrant colors while adding fabric grain
        textured_overlay = ImageChops.soft_light(combined, layer)
        result = Image.blend(combined, textured_overlay, lighting_depth)
        
        # Add subtle Hard Shadows for deep creases only
        crease_mask = ImageEnhance.Contrast(lighting_map).enhance(1.5).convert("RGBA")
        shadow_final = ImageChops.multiply(layer, crease_mask)
        result = Image.blend(result, Image.alpha_composite(result, shadow_final), lighting_depth * 0.3)
        
        return result
    
    return combined

# --- Standard Utilities ---
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
    
    sel_d = st.selectbox("Current Graphic", d_files, format_func=lambda x: x.name)
    sel_s = st.selectbox("Current Template", s_files, format_func=lambda x: x.name)

    try:
        sel_d.seek(0); d_img = Image.open(sel_d).convert("RGBA")
        sel_s.seek(0); s_img = Image.open(sel_s).convert("RGBA")

        is_m = "model" in sel_s.name.lower()
        off = m_off if is_m else p_off
        pad = m_pad if is_m else p_pad

        bbox = get_shirt_bbox(s_img)
        if bbox:
            sc = min(bbox[2]/d_img.width, bbox[3]/d_img.height, 1.0) * pad
            nw, nh = int(d_img.width*sc), int(d_img.height*sc)
            px, py = bbox[0] + (bbox[2]-nw)//2, bbox[1] + int(bbox[3]*off/100)
            
            # PREVIEW RENDER
            render = apply_premium_render(s_img, d_img, px, py, (nw, nh))
            st.image(render, caption="Final Premium Rendered Mockup", use_container_width=True)
            
            # BATCH EXPORT
            if st.button("🚀 Export All (Premium Pro-Zip)"):
                m_zip = io.BytesIO()
                with zipfile.ZipFile(m_zip, "w") as mzf:
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
                st.download_button("📂 Download Professional Export", m_zip.getvalue(), "premium_pro_mockups.zip")
        else:
            st.warning("Detection failed. Please use a clean template background.")
    except Exception as e:
        st.error(f"Render Error: {e}")
