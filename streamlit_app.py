import streamlit as st
import cv2
import numpy as np
import base64
from measurement import BodyMeasurement

# Page Config
st.set_page_config(page_title="TailorAI - Body Measurement", page_icon="📏")

# Initialize logic (cached to avoid reloading model)
@st.cache_resource
def get_measurer():
    return BodyMeasurement()

measurer = get_measurer()

st.title("TailorAI 📏 Body Measurement")
st.markdown("Upload a photo or use your webcam to get estimated body measurements.")

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    height = st.number_input("Your Height (cm)", min_value=50, max_value=300, value=170, step=1)
    st.info("Please ensure your full body is visible in the photo for accurate measurements.")

# Tabs for input method
tab1, tab2 = st.tabs(["📸 Camera", "📁 Upload"])

def process_and_display(image_bytes):
    if image_bytes is None:
        return

    with st.spinner("Processing..."):
        try:
            # Process image
            # measurement.py accepts bytes
            results = measurer.process_image(image_bytes, height)
            
            if "error" in results:
                st.error(f"Error: {results['error']}")
                return

            # Decode and display result image
            img_bytes = base64.b64decode(results['image_base64'])
            nparr = np.frombuffer(img_bytes, np.uint8)
            processed_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            
            st.image(processed_img_rgb, caption="Processed Image with Measurements", use_column_width=True)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Shoulder Width", f"{results['shoulder_width_cm']} cm")
            with col2:
                st.metric("Hip Width", f"{results['hip_width_cm']} cm")
            with col3:
                st.metric("Arm Length", f"{results['arm_length_cm']} cm")
                
            with st.expander("Debug Details"):
                st.json(results)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

with tab1:
    st.header("Use Webcam")
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        process_and_display(camera_image.getvalue())

with tab2:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        process_and_display(uploaded_file.getvalue())
