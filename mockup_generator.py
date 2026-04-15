import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import zipfile
import io
import cv2
import os

st.set_page_config(page_title="Universal Mockup Generator", layout="centered")
st.title("👕 Universal Premium Mockup Generator")

st.markdown("""
**Dynamic Engine:** Automatically detects shirt color to apply the best blending math.
* **Warping:** Physical displacement along fabric folds.
* **Vibrancy:** Keeps your 'Gwest Dept' blues punchy on any fabric.
""")

# --- Sidebar: Controls ---
st.sidebar.header("📍 Placement Settings")
plain_padding_ratio = st.sidebar.slider("Padding Ratio – Plain", 0.1, 1.0, 0.45, 0.05)
model_padding_ratio = st.sidebar.slider("Padding Ratio – Model", 0.1, 1.0, 0.35, 0.05)
plain_offset_pct = st.sidebar.slider("Vertical Offset – Plain (%)", -50, 100, 23, 1)
model_offset_pct = st.sidebar.slider("Vertical Offset – Model (%)", -50, 100, 38, 1)

st.sidebar.header("✨ Dynamic Realism")
warp_intensity = st.sidebar.slider("Warp Strength (Fold Following)", 0.0, 5.0, 1.5, 0.5)
texture_depth = st.sidebar.slider("Fabric Texture Depth", 0.0, 1.0, 0.25, 0.05)
ink_vibrancy = st.sidebar.slider("Ink Vibrancy", 0.5, 1.5, 1.0, 0.05)

# --- Helper: Universal Dynamic Engine ---
def apply_universal_mockup(shirt_bg, design_img, x, y, size):
    # 1. High-Quality Resize & Sharpness (Prevents Blur)
    design_res = design_img.resize(size, Image.Resampling.LANCZOS)
    if warp_intensity > 0:
        design_res = design_res.filter(ImageFilter.SHARPEN)
    
    # 2. Safety Check
    if warp_intensity == 0 and texture_depth == 0:
        design_layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
        design_layer.paste(design_res, (x, y), design_res)
        return Image.alpha_composite(shirt_bg.convert("RGBA"), design_layer)

    # 3. Analyze Shirt Luminance
    shirt_np = np.array(shirt_bg.convert("RGB"))
    roi = shirt_np[y:y+size[1], x:x+size[0]]
    roi_pil = Image.fromarray(roi)
    avg_brightness = np.array(roi_pil.convert("L")).mean()

    # 4. Extract Texture Based on Color
    if avg_brightness > 200: # White/Very Light
        tex_finder = ImageEnhance.Contrast(roi_pil.convert("L")).enhance(2.5)
        tex_finder = ImageEnhance.Brightness(tex_finder).enhance(0.8)
    elif avg_brightness < 50: # Black/Very Dark
        tex_finder = ImageEnhance.Brightness(roi_pil.convert("L")).enhance(3.0)
        tex_finder = ImageEnhance.Contrast(tex_finder).enhance(3.5)
    else: # Mid-tones (Red, Blue, Green, etc.)
        tex_finder = ImageEnhance.Contrast(roi_pil.convert("L")).enhance(2.0)
    
    roi_gray = np.array(tex_finder)

    # 5. Anti-Blur Displacement (Warping)
    if warp_intensity > 0:
        grad_x = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
        
        design_np = np.array(design_res)
        h, w = design_np.shape[:2]
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Warp with INTER_CUBIC to maintain edge sharpness
        map_x = (xx + grad_x * (warp_intensity * 0.04)).astype(np.float32)
        map_y = (yy + grad_y * (warp_intensity * 0.04)).astype(np.float32)
        
        warped_np = cv2.remap(design_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        design_final = Image.fromarray(warped_np)
    else:
        design_final = design_res

    # 6. Blending Layer
    design_layer = Image.new("RGBA", shirt_bg.size, (0, 0, 0, 0))
    design_layer.paste(design_final, (x, y), design_final)
    design_layer = ImageEnhance.Color(design_layer).enhance(ink_vibrancy)

    # 7. Final Composite Math
    combined = Image.alpha_composite(shirt_bg.convert("RGBA"), design_layer)
    
    if texture_depth > 0:
        # Generate the texture overlay
        shirt_tex_full = shirt_bg.convert("L")
        if avg_brightness < 128:
            # For dark: Hard shadows
            tex_map = ImageEnhance.Contrast(shirt_tex_full).enhance(2.0).convert("RGBA")
        else:
            # For light: Subtle folds
            tex_map = ImageEnhance.Brightness(shirt_tex_full).enhance(0.9).convert("RGBA")
            
        shadow_mask = ImageChops.multiply(design_layer, tex_map)
        # Use lower multiplier for depth to keep the colors from getting "dirty"
        result = Image.blend(combined, Image.alpha_composite(combined, shadow_mask), texture_depth * 0.35)
        return result
    
    return combined

# --- Standard Utilities (Bounding Box, Uploads, Batching) ---
def get_shirt_bbox(pil_image):
    img_cv = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.boundingRect(max(contours, key=cv2.contourArea)) if contours else None

design_files = st.file_uploader("📌 Upload Design PNGs", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
shirt_files = st.file_uploader("🎨 Upload Shirt Templates", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if design_files and shirt_files:
    # Logic for batching and naming
    if "d_names" not in st.session_state: st.session_state.d_names = {}
    for f in design_files: st.session_state.d_names[f.name] = os.path.splitext(f.name)[0]
    
    selected_design = st.selectbox("Preview Design", design_files, format_func=lambda x: x.name)
    selected_shirt = st.selectbox("Preview Shirt", shirt_files, format_func=lambda x: x.name)

    try:
        selected_design.seek(0); d_img = Image.open(selected_design).convert("RGBA")
        selected_shirt.seek(0); s_img = Image.open(selected_shirt).convert("RGBA")

        is_m = "model" in selected_shirt.name.lower()
        off = model_offset_pct if is_m else plain_offset_pct
        pad = model_padding_ratio if is_m else plain_padding_ratio

        bbox = get_shirt_bbox(s_img)
        if bbox:
            sc = min(bbox[2]/d_img.width, bbox[3]/d_img.height, 1.0) * pad
            nw, nh = int(d_img.width*sc), int(d_img.height*sc)
            px, py = bbox[0] + (bbox[2]-nw)//2, bbox[1] + int(bbox[3]*off/100)
            
            final_preview = apply_universal_mockup(s_img, d_img, px, py, (nw, nh))
            st.image(final_preview, caption="Dynamic Realistic Output", use_container_width=True)
            
            # Batch Button
            if st.button("🚀 Process & Download Zip"):
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
                                    s_calc = min(b[2]/di.width, b[3]/di.height, 1.0) * (model_padding_ratio if "model" in sf.name.lower() else plain_padding_ratio)
                                    nw_b, nh_b = int(di.width*s_calc), int(di.height*s_calc)
                                    px_b, py_b = b[0] + (b[2]-nw_b)//2, b[1] + int(b[3]*off/100)
                                    res = apply_universal_mockup(si, di, px_b, py_b, (nw_b, nh_b))
                                    buf = io.BytesIO(); res.convert("RGB").save(buf, format="JPEG", quality=95)
                                    zf.writestr(f"{sf.name}.jpg", buf.getvalue())
                        inner.seek(0); mzf.writestr(f"{st.session_state.d_names[df.name]}.zip", inner.read())
                st.download_button("📂 Download All", master_zip.getvalue(), "mockups.zip")
        else:
            st.warning("Detection failed. Check template background.")
    except Exception as e:
        st.error(f"Error: {e}")
