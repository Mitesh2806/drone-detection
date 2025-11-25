import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import pandas as pd
# --- Page Configuration ---
st.set_page_config(
    page_title="YOLOv8 Drone Detection",
    page_icon="🚀",
    layout="wide"
)

# --- Model Loading ---
MODEL_PATH = '/best.pt'

# We cache the model loading to make the app faster.
@st.cache_resource
def load_model(model_path):
    """Loads the YOLOv8 model from the specified path."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.error("Please make sure the path is correct and the model file exists.")
        return None
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model(MODEL_PATH)

# --- Page Title ---
st.title("🚀 YOLOv8 Object Detection")
st.write("Upload an image and the model will detect objects.")

# --- Inference ---
if model is None:
    st.error("Model could not be loaded. App cannot run.")
else:
    # 1. HOW THE USER UPLOADS AN IMAGE
    uploaded_file = st.file_uploader(
        "Upload an image (jpg, png, jpeg)", 
        type=["jpg", "png", "jpeg"]
    )
    
    # Add a confidence slider
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    if uploaded_file is not None:
        try:
            # Open the uploaded image
            image = Image.open(uploaded_file)
            
            # Convert to numpy array
            image_np = np.array(image)
            
            st.subheader("Original Image")
            st.image(image, caption="Your uploaded image")
            
            with st.spinner("Running detection..."):
                # 2. HOW TO GET PREDICTIONS
                results = model(image_np, conf=confidence_threshold)
                
                # Draw the bounding boxes on the image
                annotated_image_bgr = results[0].plot()
                
                # Convert BGR (OpenCV format) to RGB (PIL/Streamlit format)
                annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
                
                st.subheader("Detection Results")
                st.image(annotated_image_rgb, caption="Image with detected objects")
                
                # Show the raw detection data
                st.subheader("Detected Objects Data")
                detections = results[0].boxes.data.cpu().numpy()
                if len(detections) > 0:
                    df_detections = pd.DataFrame(detections, columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class_id'])
                    # Get class names from the model
                    df_detections['class_name'] = df_detections['class_id'].apply(lambda x: model.names[int(x)])
                    st.dataframe(df_detections)
                else:
                    st.info("No objects detected with the current confidence threshold.")
                    
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")