# python libraries

import streamlit as st
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw, ImageFont
from rembg import remove
import io
import base64
import cv2
import numpy as np
import os




# --------------------------
# Page Setup & Custom CSS
# --------------------------
st.set_page_config(
    page_title="🎨 AI Image Studio | Background Removal & Filters",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Vibrant, modern CSS with gradient accents
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&family=Montserrat:wght@400;600;800&display=swap');
    
    * {{
        font-family: 'Poppins', sans-serif;
        color: #4A2B82; /* Deep Plum */
    }}
    
    :root {{
        --primary: #6C63FF;
        --secondary: #FF6584;
        --accent: #00BFA6;
        --dark: #2A2D43;
        --light: #F7F9FC;
    }}
    
    body {{
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
    }}
    
    .header {{
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, var(--primary) 0%, #7b68ee 100%);
        color: white;
        border-radius: 0 0 30px 30px;
        margin: -1rem -1rem 2.5rem -1rem;
        box-shadow: 0 15px 30px rgba(108, 99, 255, 0.25);
    }}
    
    .feature-card {{
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 18px;
        padding: 1.8rem;
        margin: 1.2rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        border: none;
        transition: all 0.4s ease;
    }}
    
    .feature-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(108, 99, 255, 0.2);
    }}
    
    .success-badge {{
        background: linear-gradient(135deg, var(--accent) 0%, #00ccb1 100%);
        color: white;
        padding: 0.8rem 1.8rem;
        border-radius: 50px;
        font-weight: 700;
        margin: 2rem auto;
        width: fit-content;
        box-shadow: 0 8px 20px rgba(0, 191, 166, 0.35);
        font-size: 1.1rem;
    }}
    
    .download-btn {{
        background: linear-gradient(135deg, var(--secondary) 0%, #ff7e9a 100%);
        color: white !important;
        padding: 16px 32px;
        border-radius: 14px;
        text-decoration: none;
        font-weight: 700;
        display: block;
        text-align: center;
        margin: 2rem auto;
        width: 90%;
        font-size: 1.15rem;
        box-shadow: 0 8px 20px rgba(255, 101, 132, 0.35);
        transition: all 0.4s ease;
        border: none;
        letter-spacing: 0.5px;
    }}
    
    .download-btn:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 25px rgba(255, 101, 132, 0.45);
        color: white !important;
    }}
    
    .upload-area {{
        background: rgba(255, 255, 255, 0.85);
        border: 2px dashed #c2d0ea;
        border-radius: 24px;
        padding: 3.5rem 2rem;
        text-align: center;
        margin: 2.5rem 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }}
    
    .upload-area:hover {{
        border-color: var(--primary);
        background-color: rgba(250, 251, 255, 0.95);
        box-shadow: 0 10px 25px rgba(108, 99, 255, 0.1);
    }}
    
    .image-container {{
        background: url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23d8e2f3' fill-opacity='0.3' fill-rule='evenodd'%3E%3Cpath d='M0 40L40 0H20L0 20M40 40V20L20 40'/%3E%3C/g%3E%3C/svg%3E");
        border-radius: 18px;
        padding: 25px;
        display: flex;
        justify-content: center;
    }}
    
    .comparison-container {{
        display: flex;
        justify-content: space-around;
        gap: 2.5rem;
        margin: 2.5rem 0;
        flex-wrap: wrap;
    }}
    
    .comparison-box {{
        flex: 1;
        min-width: 320px;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        transition: all 0.4s ease;
        backdrop-filter: blur(5px);
    }}
    
    .comparison-box:hover {{
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.12);
    }}
    
    .comparison-header {{
        background: linear-gradient(135deg, var(--primary) 0%, #7b68ee 100%);
        color: white;
        padding: 1.2rem;
        text-align: center;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }}
    
    footer {{
        text-align: center;
        margin-top: 4rem;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.95rem;
        border-top: 1px solid rgba(233, 236, 239, 0.5);
        background: rgba(255, 255, 255, 0.85);
        border-radius: 24px 24px 0 0;
        backdrop-filter: blur(5px);
    }}
    
    .pulse {{
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    
    .tab-content {{
        padding: 2rem 0;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        padding: 1rem 2rem;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin: 0 5px !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--primary) 0%, #7b68ee 100%) !important;
        color: white !important;
        box-shadow: 0 8px 20px rgba(108, 99, 255, 0.25);
        transform: translateY(-2px);
    }}
    
    .filter-card {{
        background: rgba(255, 255, 255, 0.85);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.06);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }}
    
    .filter-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(108, 99, 255, 0.15);
    }}
    
    .feature-icon {{
        font-size: 2.8rem;
        margin-bottom: 1.2rem;
        background: linear-gradient(135deg, var(--primary) 0%, #7b68ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 10px rgba(108, 99, 255, 0.15);
    }}
    
    .color-option {{
        width: 55px;
        height: 55px;
        border-radius: 50%;
        cursor: pointer;
        border: 3px solid transparent;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }}
    
    .color-option:hover {{
        transform: scale(1.12);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }}
    
    .color-option.selected {{
        border: 4px solid #2A2D43;
        transform: scale(1.18);
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }}
    
    .slider-container {{
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 16px;
        margin: 1.5rem 0;
        backdrop-filter: blur(5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
    }}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Helper Functions
# --------------------------

@st.cache_resource
def get_rembg_model():
    """Cache the rembg model to avoid reloading it on every rerun."""
    # rembg's remove function loads the model implicitly.
    # We'll just call it once with a dummy image to ensure the model is loaded.
    try:
        dummy_img = Image.new("RGB", (10, 10))
        remove(dummy_img)
        st.success("Background removal model loaded! ✅")
    except Exception as e:
        st.error(f"Failed to load background removal model: {e}. Please check your internet connection and try again.")
    return True # Return a dummy value to indicate success/failure of loading

def pil_to_cv2(img_pil):
    """Converts a PIL Image to an OpenCV image (BGR)."""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv2):
    """Converts an OpenCV image (BGR) to a PIL Image."""
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

def apply_sepia(img_cv2):
    """Applies a sepia tone filter to an OpenCV image."""
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img_cv2, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def apply_filters(img_cv2, filter_name):
    """Applies various image filters using OpenCV."""
    if filter_name == "Grayscale":
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2BGR) # Convert back to BGR for consistency
    elif filter_name == "Sepia":
        img_cv2 = apply_sepia(img_cv2)
    elif filter_name == "Blur":
        img_cv2 = cv2.GaussianBlur(img_cv2, (25, 25), 0)
    elif filter_name == "Canny Edge":
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        img_cv2 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif filter_name == "Pencil Sketch":
        # pencilSketch returns gray and color sketch, we use color sketch
        _, img_cv2 = cv2.pencilSketch(img_cv2, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    elif filter_name == "Invert Colors":
        img_cv2 = cv2.bitwise_not(img_cv2)
    elif filter_name == "Oil Painting":
        # Ensure opencv-contrib-python is installed for xphoto module
        try:
            img_cv2 = cv2.xphoto.oilPainting(img_cv2, 7, 1)
        except AttributeError:
            st.warning("Oil Painting filter requires `opencv-contrib-python`. Please install it (`pip install opencv-contrib-python`) for this feature.")
    elif filter_name == "Emboss":
        kernel = np.array([[0, -1, -1],
                           [1, 0, -1],
                           [1, 1, 0]])
        img_cv2 = cv2.filter2D(img_cv2, -1, kernel)
    elif filter_name == "Sharpen":
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        img_cv2 = cv2.filter2D(img_cv2, -1, kernel)
    return img_cv2

@st.cache_data
def remove_background(_image_bytes):
    """Removes the background from an image using rembg. Cached for performance."""
    try:
        input_image = Image.open(io.BytesIO(_image_bytes))
        output_image = remove(input_image)
        return output_image
    except Exception as e:
        st.error(f"❌ Error during background removal: {str(e)}")
        return None

def get_image_download_link(img, filename, text):
    """Generates a base64 encoded download link for a PIL Image."""
    buffered = io.BytesIO()
    img_format = "PNG" if img.mode == 'RGBA' else "JPEG"
    img.save(buffered, format=img_format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/{img_format.lower()};base64,{img_str}" download="{filename}" class="download-btn pulse">{text}</a>'

def add_colored_background(image, color_hex):
    """Adds a solid color background to an RGBA image."""
    if color_hex is None:
        return image
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
        
    background = Image.new('RGBA', image.size, color_hex)
    composite = Image.alpha_composite(background, image)
    return composite.convert('RGB')

@st.cache_data
def auto_resize(img_pil, max_dim=1200):
    """Resizes an image if its largest dimension exceeds max_dim, maintaining aspect ratio."""
    width, height = img_pil.size
    if max(width, height) > max_dim:
        ratio = max_dim / float(max(width, height))
        new_size = (int(width * ratio), int(height * ratio))
        # Use Image.Resampling.LANCZOS for high-quality downsampling
        return img_pil.resize(new_size, Image.Resampling.LANCZOS)
    return img_pil

def add_watermark(image, text, position, opacity=0.5):
    """Adds a text watermark to an image."""
    if not text:
        return image

    watermark = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)
    
    # Try to load a truetype font, fall back to default
    font_size = min(image.size) // 20
    try:
        # Prioritize common system fonts, or provide a path if bundled
        font_path = "arial.ttf" if os.path.exists("arial.ttf") else "/System/Library/Fonts/Supplemental/Arial.ttf" # Example for macOS
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        try:
            # Another common font if Arial isn't found
            font_path = "FreeSans.ttf" # Often found on Linux systems, or can be bundled
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default() # Fallback to a basic bitmap font
            st.warning("Could not load TrueType font for watermark. Using default font.")

    # Get text bounding box using textbbox
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top
    
    # Position handling
    margin = 20
    if position == "Bottom Right":
        x = image.width - text_width - margin
        y = image.height - text_height - margin
    elif position == "Top Left":
        x = margin
        y = margin
    elif position == "Top Right":
        x = image.width - text_width - margin
        y = margin
    elif position == "Bottom Left":
        x = margin
        y = image.height - text_height - margin
    else:  # Center
        x = (image.width - text_width) // 2
        y = (image.height - text_height) // 2
    
    # Draw text with opacity
    fill_color = (255, 255, 255, int(255 * opacity))
    draw.text((x, y), text, font=font, fill=fill_color)
    
    # Composite with original image
    return Image.alpha_composite(image.convert("RGBA"), watermark).convert("RGB")

def apply_rotation(image, rotation):
    """Applies rotation to a PIL Image."""
    if rotation == 90:
        return image.transpose(Image.Transpose.ROTATE_90)
    elif rotation == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    elif rotation == 270:
        return image.transpose(Image.Transpose.ROTATE_270)
    return image

@st.cache_data
def enhance_image(_img_pil, brightness=1.0, contrast=1.0, saturation=1.0):
    """Applies brightness, contrast, and saturation enhancements to a PIL Image."""
    enhancer = ImageEnhance.Brightness(_img_pil)
    _img_pil = enhancer.enhance(brightness)
    
    enhancer = ImageEnhance.Contrast(_img_pil)
    _img_pil = enhancer.enhance(contrast)
    
    enhancer = ImageEnhance.Color(_img_pil)
    return enhancer.enhance(saturation)

# --------------------------
# UI Components as Functions
# --------------------------

def render_header():
    """Renders the main application header."""
    st.markdown("""
    <div class="header">
        <h1 style="font-size: 3.2rem; margin-bottom: 0.8rem; font-family: 'Montserrat', sans-serif;">✨ AI IMAGE STUDIO</h1>
        <p style="font-size: 1.4rem; opacity: 0.95; max-width: 800px; margin: 0 auto;">
            Remove backgrounds & apply professional filters with AI
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_background_remover_tab():
    """Renders the Background Remover tab content."""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>Remove Backgrounds Instantly</h2>
        <p style="font-size: 1.1rem;">Upload any image to automatically remove its background in seconds</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for color selection
    if 'selected_color' not in st.session_state:
        st.session_state.selected_color = None
        
    # Define color options
    COLOR_OPTIONS = {
        "Transparent": None,
        "Pure White": "#FFFFFF",
        "Sky Blue": "#87CEEB",
        "Lime Green": "#32CD32",
        "Sunshine Yellow": "#FFD700",
        "Coral Pink": "#FF7F50",
        "Lavender": "#E6E6FA"
    }
    
    uploaded_file = st.file_uploader(
        "📤 Upload your image",
        type=["png", "jpg", "jpeg"],
        key="bg_uploader"
    )
    
    if uploaded_file is None:
        st.markdown("""
        <div class="upload-area">
            <div style="font-size: 5rem; color: #6C63FF; margin-bottom: 1.5rem;">📁</div>
            <h3 style="color: #2A2D43; margin-bottom: 0.5rem;">Drag & Drop Your Image Here</h3>
            <p style="color: #6c757d; font-size: 1.1rem;">or click to browse files</p>
            <p style="font-size: 0.95rem; color: #6c757d; margin-top: 1.5rem;">Supports JPG, PNG (Max 10MB)</p>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display original image
        st.subheader("Your Image Preview")
        original_image = Image.open(uploaded_file)
        original_image = auto_resize(original_image)
        st.image(original_image, use_container_width=True)
        
        # Process button
        if st.button("✨ REMOVE BACKGROUND NOW", 
                     type="primary", 
                     use_container_width=True,
                     help="Click to process your image with AI"):
            
            with st.spinner("🧠 AI is working its magic..."):
                # Pass bytes to cached function
                result_image = remove_background(uploaded_file.getvalue())
                
            if result_image:
                st.balloons()
                st.markdown('<div class="success-badge">✅ BACKGROUND REMOVED SUCCESSFULLY!</div>', unsafe_allow_html=True)
                
                # Store result in session state
                st.session_state.bg_result_image = result_image
                
                # Display comparison
                st.markdown("## Before & After")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="comparison-box">
                        <div class="comparison-header">Original</div>
                        <div style="padding: 1.8rem; display: flex; justify-content: center;">
                            <img src="data:image/png;base64,{orig}" width="100%" style="max-width: 100%; border-radius: 12px;">
                        </div>
                    </div>
                    """.format(orig=base64.b64encode(io.BytesIO(uploaded_file.getvalue()).getvalue()).decode()), 
                    unsafe_allow_html=True)
                
                with col2:
                    buffered = io.BytesIO()
                    result_image.save(buffered, format="PNG")
                    result_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    st.markdown(f"""
                    <div class="comparison-box">
                        <div class="comparison-header" style="background: linear-gradient(135deg, var(--accent) 0%, #00ccb1 100%);">Result</div>
                        <div class="image-container">
                            <img src="data:image/png;base64,{result_str}" width="100%" style="max-width: 100%; border-radius: 12px;">
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Download button for transparent version
                st.markdown(get_image_download_link(
                    result_image,
                    f"no_bg_{uploaded_file.name.split('.')[0]}.png",
                    "⬇ DOWNLOAD TRANSPARENT PNG"
                ), unsafe_allow_html=True)
                
                # Reset selected color
                st.session_state.selected_color = None
    
    # Color options and preview
    if 'bg_result_image' in st.session_state and st.session_state.bg_result_image:
        st.markdown("---")
        st.markdown("## 🎨 Choose a Background Color")
        
        # Create color options
        st.markdown('<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 25px;">', unsafe_allow_html=True)
        
        # Streamlit columns break the flexbox layout, so we'll render buttons and CSS manually
        color_buttons_html = ""
        for idx, (color_name, hex_code) in enumerate(COLOR_OPTIONS.items()):
            if hex_code:
                color_style = f"background-color: {hex_code};"
            else:
                # Transparent option - checkerboard pattern
                color_style = """
                    background: 
                        linear-gradient(45deg, #ccc 25%, transparent 25%), 
                        linear-gradient(-45deg, #ccc 25%, transparent 25%),
                        linear-gradient(45deg, transparent 75%, #ccc 75%),
                        linear-gradient(-45deg, transparent 75%, #ccc 75%);
                    background-size: 10px 10px;
                    background-position: 0 0, 0 5px, 5px -5px, -5px 0px;
                """ 
            selected = "selected" if st.session_state.selected_color == hex_code else ""
            
            # Embed a hidden button to trigger state change
            color_buttons_html += f"""
            <div style="text-align: center;">
                <button class="color-option {selected}" 
                        style="{color_style}"
                        onclick="window.parent.document.querySelector('button[data-testid=\"stButton:btn-{idx}\"]').click();">
                </button>
                <div style="text-align: center; margin-top: 8px; font-size: 0.85rem; font-weight: 500;">{color_name}</div>
            </div>
            """
            # Streamlit button to capture the click and update state
            if st.button(color_name, key=f"btn-{idx}", help=color_name, use_container_width=False):
                st.session_state.selected_color = hex_code
                st.rerun() # Rerun to apply color immediately

        st.markdown(color_buttons_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Custom color picker
        st.markdown("---")
        st.subheader("Or Pick a Custom Color")
        custom_color = st.color_picker("Choose a custom background color", "#FFFFFF")
        if custom_color != "#FFFFFF" and st.session_state.selected_color != custom_color:
            if st.button("Apply Custom Color", use_container_width=True):
                st.session_state.selected_color = custom_color
                st.rerun() # Rerun to apply custom color

        # Preview of selected color
        if st.session_state.selected_color is not None:
            st.markdown("### 🖼️ Preview with Selected Background")
            
            colored_image = add_colored_background(st.session_state.bg_result_image, st.session_state.selected_color)
            st.image(colored_image, use_container_width=True)
            
            color_name_for_download = [name for name, code in COLOR_OPTIONS.items() if code == st.session_state.selected_color]
            color_name_for_download = color_name_for_download[0] if color_name_for_download else "Custom"
            
            st.markdown(get_image_download_link(
                colored_image,
                f"colored_bg_{color_name_for_download.lower().replace(' ', '_')}_{uploaded_file.name.split('.')[0]}.png",
                f"⬇ DOWNLOAD WITH {color_name_for_download.upper()} BACKGROUND"
            ), unsafe_allow_html=True)
        else:
            st.info("👆 Select a background color or pick a custom one to preview")
        
        # Reset option
        st.markdown("---")
        if st.button("🔄 PROCESS ANOTHER IMAGE", 
                     use_container_width=True,
                     type="secondary"):
            # Clear session state specific to this tab
            for key in ['bg_result_image', 'selected_color', 'bg_uploader']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun() # Use st.rerun instead of experimental_rerun

def render_image_filters_tab():
    """Renders the Image Filters tab content."""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>Transform Your Images</h2>
        <p style="font-size: 1.1rem;">Apply professional filters and enhancements to your photos</p>
    </div>
    """, unsafe_allow_html=True)
    
    FILTER_OPTIONS = {
        "Basic": ["None", "Grayscale", "Sepia", "Blur", "Sharpen"],
        "Artistic": ["Pencil Sketch", "Oil Painting", "Emboss"],
        "Special Effects": ["Canny Edge", "Invert Colors"]
    }
    
    # Initialize session state for filter tab specific variables
    if 'filter_image_raw' not in st.session_state:
        st.session_state.filter_image_raw = None
    if 'filtered_image' not in st.session_state:
        st.session_state.filtered_image = None
    
    uploaded_file = st.file_uploader(
        "📤 Upload your image", 
        type=["jpg", "jpeg", "png"],
        key="filter_uploader"
    )
    
    if uploaded_file:
        # Load and process image
        try:
            # Only load and resize if a new file is uploaded or not in session state
            if st.session_state.filter_image_raw is None or st.session_state.filter_image_raw != uploaded_file.getvalue():
                image = Image.open(uploaded_file).convert("RGB")
                image = auto_resize(image)
                st.session_state.filter_image_raw = uploaded_file.getvalue() # Store raw bytes to check for changes
                st.session_state.processed_filter_image = image # Store processed PIL image
            
            image_to_process = st.session_state.processed_filter_image
            
            st.subheader("Original Image")
            st.image(image_to_process, use_container_width=True)
            
            with st.sidebar:
                st.header("⚙️ Filter Settings")
                
                filter_category = st.selectbox("Filter Category", list(FILTER_OPTIONS.keys()), key="filter_cat")
                filter_selected = st.selectbox("Select Filter", FILTER_OPTIONS[filter_category], key="filter_sel")
                
                rotation = st.selectbox("Rotate Image", [0, 90, 180, 270], key="rotation_sel")
                
                st.subheader("🎚 Adjustments")
                brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1, key="brightness_sl")
                contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1, key="contrast_sl")
                saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1, key="saturation_sl")
                
                st.subheader("💧 Watermark")
                watermark_text = st.text_input("Watermark Text", "", key="watermark_text_in")
                watermark_position = st.selectbox("Position", 
                                                 ["Bottom Right", "Top Left", "Top Right", "Bottom Left", "Center"], 
                                                 key="watermark_pos")
                watermark_opacity = st.slider("Opacity", 0.0, 1.0, 0.5, 0.1, key="watermark_opacity_sl") if watermark_text else 0.0
                
                if st.button("Reset All Settings", use_container_width=True, key="filter_reset_btn"):
                    for key in ['filter_image_raw', 'filtered_image', 'processed_filter_image', 'filter_cat', 'filter_sel', 
                                'rotation_sel', 'brightness_sl', 'contrast_sl', 'saturation_sl', 
                                'watermark_text_in', 'watermark_pos', 'watermark_opacity_sl']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            
            # Apply all transformations in order
            current_image = image_to_process
            current_image = apply_rotation(current_image, rotation)
            current_image = enhance_image(current_image, brightness, contrast, saturation)
            
            if filter_selected != "None":
                cv_image = pil_to_cv2(current_image)
                filtered_cv = apply_filters(cv_image, filter_selected)
                final_image = cv2_to_pil(filtered_cv)
            else:
                final_image = current_image
                
            if watermark_text:
                final_image = add_watermark(final_image, watermark_text, watermark_position, watermark_opacity)
            
            st.session_state.filtered_image = final_image
            
            st.subheader("✨ Processed Image")
            st.image(final_image, use_container_width=True)
            
            st.markdown("### 📥 Download Your Image")
            buf = io.BytesIO()
            final_image.save(buf, format="PNG")
            
            st.download_button(
                "⬇️ Download as PNG", 
                data=buf.getvalue(),
                file_name="filtered_image.png", 
                mime="image/png",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.session_state.filter_image_raw = None # Clear potentially corrupted state
    else:
        st.info("👆 Please upload an image to get started with filters and enhancements.")
        st.session_state.filtered_image = None
        st.session_state.filter_image_raw = None

def render_about_tab():
    """Renders the About tab content."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h2>✨ About AI Image Studio</h2>
            <p>AI Image Studio combines powerful background removal with professional-grade image filters in one seamless application.</p>
            <p>Powered by advanced AI algorithms, our tool can process your images in seconds while maintaining the highest quality results.</p>
            <p>All processing happens locally in your browser - your images are never uploaded to any server.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🧹 Background Removal</h3>
            <p>• Remove backgrounds automatically with AI</p>
            <p>• Choose from multiple background colors</p>
            <p>• Download transparent PNGs instantly</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🎨 Image Filters</h3>
            <p>• Apply professional filters and effects</p>
            <p>• Adjust brightness, contrast, and saturation</p>
            <p>• Add custom watermarks to your images</p>
            <p>• Rotate images to any orientation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h2>🌟 Key Features</h2>
            
            <div style="display: flex; align-items: center; margin: 1.5rem 0;">
                <div style="font-size: 2.5rem; margin-right: 1rem;">⚡</div>
                <div>
                    <h3 style="margin: 0;">Ultra-Fast Processing</h3>
                    <p style="margin: 0;">Process images in seconds using optimized AI algorithms</p>
                </div>
            </div>
            
            <div style="display: flex; align-items: center; margin: 1.5rem 0;">
                <div style="font-size: 2.5rem; margin-right: 1rem;">🔒</div>
                <div>
                    <h3 style="margin: 0;">Privacy First</h3>
                    <p style="margin: 0;">Your images never leave your device - all processing is local</p>
                </div>
            </div>
            
            <div style="display: flex; align-items: center; margin: 1.5rem 0;">
                <div style="font-size: 2.5rem; margin-right: 1rem;">🎨</div>
                <div>
                    <h3 style="margin: 0;">Professional Results</h3>
                    <p style="margin: 0;">Get studio-quality results with our advanced processing</p>
                </div>
            </div>
            
            <div style="display: flex; align-items: center; margin: 1.5rem 0;">
                <div style="font-size: 2.5rem; margin-right: 1rem;">💻</div>
                <div>
                    <h3 style="margin: 0;">No Installation</h3>
                    <p style="margin: 0;">Works directly in your browser - no downloads required</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_footer():
    """Renders the application footer."""
    st.markdown("""
    <footer>
        <div style="display: flex; justify-content: center; gap: 25px; margin-bottom: 1.5rem; flex-wrap: wrap;">
            <span style="display: flex; align-items: center; gap: 5px;">✉️ itsvikassharma007@gmail.com</span>
            <span style="display: flex; align-items: center; gap: 5px;">🐦 <a href="https://x.com/ItsVikasXd" target="_blank" style="color: #6c757d; text-decoration: none;">https://x.com/ItsVikasXd</a></span>
            <span style="display: flex; align-items: center; gap: 5px;">💻 <a href="https://github.com/Its-Vikas-xd" target="_blank" style="color: #6c757d; text-decoration: none;">https://github.com/Its-Vikas-xd</a></span>
        </div>
        <p>© 2025 AI Image Studio | All Rights Reserved</p>
        <p style="font-size: 0.9rem; margin-top: 1.5rem; color: #6c757d;">Powered by Streamlit, rembg, OpenCV, and advanced AI technology</p>
    </footer>
    """, unsafe_allow_html=True)

# --------------------------
# Main Application Logic
# --------------------------
if __name__ == "__main__":
    render_header()

    # Pre-load rembg model once at startup
    get_rembg_model()

    tab_bg, tab_filters, tab_about = st.tabs([
        "🧹 Background Remover", 
        "🎨 Image Filters", 
        "ℹ️ About"
    ])

    with tab_bg:
        render_background_remover_tab()

    with tab_filters:
        render_image_filters_tab()

    with tab_about:
        render_about_tab()

    render_footer()