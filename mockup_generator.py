import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import zipfile
import io
import cv2
import os

st.set_page_config(page_title="Pro Mockup Placer", layout="wide")
st.title("👕 Pro Mockup Placer")

st.markdown("""
This tool focuses on **Placement Realism**. It uses **Displacement Mapping** to warp your graphic 
along the physical folds of the shirt template.
""")

# --- Sidebar: The Realism Engine ---
st.sidebar.header("✨ Realism Controls")
warp_str = st.sidebar.slider("Warp Intensity (Bends the design)", 0.0, 5.0, 2.0, 0.1)
lighting_blend = st.sidebar.slider("Fabric Lighting (Shadows/Highlights)", 0.0, 1.0, 0.3, 0.05)

st.sidebar.header("📍 Placement Controls")
p_pad = st.sidebar.slider("Padding - Plain", 0.1, 1.0, 0.45, 0.05)
m_pad = st.sidebar.slider("Padding - Model", 0.1, 1.0, 0.35, 0.05)
vert_off = st.sidebar.slider("Vertical Offset (%)", -50, 100, 32, 1)

# --- The Displacement Logic ---
def apply_mockup_placer(shirt_bg, design_img, x, y, size):
    # 1. High-Res Resize
    design_res = design_img.resize(size, Image.Resampling.LANCZOS)
    
    # --- HARD ZERO: If Warp is 0, just paste it ---
    if warp_str == 0 and lighting_blend == 0:
        layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
        layer.paste(design_res, (x, y), design_res)
        return Image.alpha_composite(shirt_bg.convert("RGBA"), layer)

    # 2. CREATE DISPLACEMENT MAP
    # We analyze the shirt area to find where the fabric 'dips'
    shirt_np = np.array(shirt_bg.convert("RGB"))
    roi = shirt_np[y:y+size[1], x:x+size[0]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    # Bilateral Filter: Deletes photo noise but keeps big folds sharp
    # This prevents the graphic from 'shattering' or pixelating
    smooth_roi = cv2.bilateralFilter(roi_gray, 15, 75, 75)

    # 3. WARP PIXELS
    if warp_str > 0:
        gx = cv2.Sobel(smooth_roi, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth_roi, cv2.CV_32F, 0, 1, ksize=3)
        
        design_np = np.array(design_res)
        h, w = design_np.shape[:2]
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # This physically moves the pixels based on the shirt folds
        mx = (xx + gx * (warp_str * 0.04)).astype(np.float32)
        my = (yy + gy * (warp_str * 0.04)).astype(np.float32)
        
        warped_np = cv2.remap(design_np, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        design_final = Image.fromarray(warped_np)
    else:
        design_final = design_res

    # 4. COMPOSITE
    layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
    layer.paste(design_final, (x, y), design_final)
    combined = Image.alpha_composite(shirt_bg.convert("RGBA"), layer)

    # 5. LIGHTING OVERLAY (The 'Printed' look)
    if lighting_blend > 0:
        # Use Soft-Light to add fabric texture without changing the graphic color
        textured = ImageChops.soft_light(combined, layer.convert("RGBA"))
        return Image.blend(combined, textured, lighting_blend)
    
    return combined

# --- Utilities ---
def get_shirt_bbox(pil_image):
    img_cv = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (7, 7), 0), 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.boundingRect(max(contours, key=cv2.contourArea)) if contours else None

# --- Main App ---
d_files = st.file_uploader("📌 Upload Graphics", type=["png"], accept_multiple_files=True)
s_files = st.file_uploader("🎨 Upload Templates", type=["png", "jpg"], accept_multiple_files=True)

if d_files and s_files:
    # Handle naming for the download
    if "dn" not in st.session_state: st.session_state.dn = {}
    for f in d_files: st.session_state.dn[f.name] = os.path.splitext(f.name)[0]

    sel_d = st.selectbox("Select Graphic", d_files, format_func=lambda x: x.name)
    sel_s = st.selectbox("Select Template", s_files, format_func=lambda x: x.name)

    try:
        sel_d.seek(0); d_img = Image.open(sel_d).convert("RGBA")
        sel_s.seek(0); s_img = Image.open(sel_s).convert("RGBA")
        
        bbox = get_shirt_bbox(s_img)
        if bbox:
            pad = m_pad if "model" in sel_s.name.lower() else p_pad
            sc = min(bbox[2]/d_img.width, bbox[3]/d_img.height, 1.0) * pad
            nw, nh = int(d_img.width*sc), int(d_img.height*sc)
            px, py = bbox[0] + (bbox[2]-nw)//2, bbox[1] + int(bbox[3]*vert_off/100)
            
            # FINAL RENDER
            result = apply_mockup_placer(s_img, d_img, px, py, (nw, nh))
            st.image(result, caption="Final Placed Mockup", use_container_width=True)
            
            # BATCH EXPORT
            if st.button("🚀 Export All (Premium Zip)"):
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
                                    sc_b = min(b[2]/c_di.width, b[3]/c_di.height, 1.0) * pad
                                    nw_b, nh_b = int(c_di.width*sc_b), int(c_di.height*sc_b)
                                    px_b, py_b = b[0] + (b[2]-nw_b)//2, b[1] + int(b[3]*vert_off/100)
                                    res = apply_mockup_placer(c_si, c_di, px_b, py_b, (nw_b, nh_b))
                                    buf = io.BytesIO(); res.convert("RGB").save(buf, format="JPEG", quality=98)
                                    zf.writestr(f"{sf.name}.jpg", buf.getvalue())
                        inner.seek(0); mzf.writestr(f"{st.session_state.dn[df.name]}.zip", inner.read())
                st.download_button("📂 Download Zip", m_zip.getvalue(), "mockups.zip")
        else:
            st.warning("Could not detect shirt area. Ensure the template background is plain.")
    except Exception as e:
        st.error(f"Error: {e}")
    
