"""
streamlit_app.py
Streamlit Web Interface for Surface Crack Detection - UPDATED
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import time
from gradcam_utils import get_gradcam_heatmap, overlay_gradcam
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Surface Crack Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-size: 1.2rem;
    }
    .crack-detected {
        background-color: #ffcccc;
        border: 2px solid #ff4444;
    }
    .no-crack {
        background-color: #ccffcc;
        border: 2px solid #44ff44;
    }
    .confidence-high {
        color: #00aa00;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffaa00;
        font-weight: bold;
    }
    .confidence-low {
        color: #ff4444;
        font-weight: bold;
    }
    .image-info {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    """Load the trained model with caching"""
    try:
        # Correct paths based on your structure
        BASE_PATH = "D:/DeepLearningModels/Model1(SurfaceCrack)"
        best_model_path = os.path.join(BASE_PATH, "models/best_model.h5")
        final_model_path = os.path.join(BASE_PATH, "models/final_model.h5")
        
        if os.path.exists(best_model_path):
            model = load_model(best_model_path)
            st.sidebar.success("✅ Best model loaded!")
            return model
        elif os.path.exists(final_model_path):
            model = load_model(final_model_path)
            st.sidebar.success("✅ Final model loaded!")
            return model
        else:
            st.sidebar.error("❌ No trained model found. Please train the model first.")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

def preprocess_image(image, target_size=(227, 227)):
    """
    Preprocess the uploaded image for prediction
    Handles both RGB and grayscale images
    """
    # Resize image
    image = image.resize(target_size)
    
    # Convert to RGB if needed (handle grayscale, RGBA, etc.)
    if image.mode != 'RGB':
        st.info(f"🔄 Converting {image.mode} image to RGB")
        image = image.convert('RGB')
    
    # Convert to array and preprocess
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1]
    
    return img_array

def plot_prediction(image, prediction, class_names):
    """Create a visualization of the prediction"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Show original image
    ax1.imshow(image)
    ax1.set_title('Uploaded Image')
    ax1.axis('off')
    
    # Show prediction probabilities
    colors = ['#44ff44', '#ff4444']  # Green for no crack, Red for crack
    bars = ax2.barh(class_names, prediction[0], color=colors)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Confidence')
    ax2.set_title('Prediction Confidence')
    
    # Add value labels on bars
    for bar, value in zip(bars, prediction[0]):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def get_confidence_level(confidence):
    """Get CSS class based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">🔍 Surface Crack Detection</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This AI model detects surface cracks in images using a Convolutional Neural Network (CNN).
    Upload an image of a concrete surface, wall, or any surface to check for cracks.
    """)
    
    # Sidebar for information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **How to use:**
        1. Upload an image (JPG, PNG, JPEG)
        2. The AI will analyze it
        3. View the results and confidence
        
        **Model Info:**
        - CNN Architecture
        - Trained on 40,000+ images
        - Binary classification: Crack vs No Crack
        - Handles both color and grayscale images
        """)
        
        st.header("📊 Model Status")
        model = load_trained_model()
        
        if model:
            st.success("Model: ✅ Loaded")
            st.info(f"Input shape: {model.input_shape[1:3]}")
            st.info(f"Expected channels: 3 (RGB)")
        else:
            st.error("Model: ❌ Not Found")
            st.info("Please train the model using train_model.py first")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of a surface to check for cracks"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Image info with better formatting
            st.markdown(f"""
            <div class="image-info">
            <strong>Image Details:</strong><br>
            • Dimensions: {image.size[0]}×{image.size[1]} pixels<br>
            • Format: {uploaded_file.type}<br>
            • Mode: {image.mode}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.header("🔍 Analysis Results")
        
        if uploaded_file is not None and model is not None:
            with st.spinner('🔄 Analyzing image for cracks...'):
                # Add a small delay to show the spinner
                time.sleep(1)
                
                try:
                    # Preprocess and predict
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image, verbose=0)
                    
                    # Get results
                    class_names = ['No Crack', 'Crack Detected']
                    predicted_class = np.argmax(prediction[0])
                    confidence = prediction[0][predicted_class]
                    
                    # Display results
                    st.subheader("Prediction Result")
                    
                    if predicted_class == 1:  # Crack Detected
                        st.markdown(
                            f'<div class="prediction-box crack-detected">'
                            f'<h3>🚨 CRACK DETECTED</h3>'
                            f'<p>Confidence: <span class="{get_confidence_level(confidence)}">{confidence:.3f}</span></p>'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
                        st.warning("⚠️ This surface may require inspection!")
                    else:  # No Crack
                        st.markdown(
                            f'<div class="prediction-box no-crack">'
                            f'<h3>✅ NO CRACK DETECTED</h3>'
                            f'<p>Confidence: <span class="{get_confidence_level(confidence)}">{confidence:.3f}</span></p>'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
                        st.success("🎉 Surface appears to be in good condition!")
                    
                    # Confidence visualization
                    st.subheader("Confidence Levels")
                    col_crack, col_no_crack = st.columns(2)
                    
                    with col_crack:
                        crack_confidence = prediction[0][1]
                        confidence_class = get_confidence_level(crack_confidence)
                        st.metric("Crack Confidence", f"{crack_confidence:.3f}")
                    
                    with col_no_crack:
                        no_crack_confidence = prediction[0][0]
                        confidence_class = get_confidence_level(no_crack_confidence)
                        st.metric("No Crack Confidence", f"{no_crack_confidence:.3f}")
                    
                    # Detailed probabilities
                    st.subheader("Detailed Analysis")
                    fig = plot_prediction(image, prediction, class_names)
                    st.pyplot(fig)
                    
                    # Additional info
                    with st.expander("📋 Technical Details"):
                        st.write(f"**Model Input Shape:** {processed_image.shape}")
                        st.write(f"**Prediction Array:** {prediction}")
                        st.write(f"**Class Probabilities:**")
                        st.json({class_names[i]: float(prediction[0][i]) for i in range(len(class_names))})
                        
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.info("Please try with a different image or check the model files.")
        
        elif uploaded_file is not None and model is None:
            st.error("❌ Please train the model first using train_model.py")
            st.info("The model files (best_model.h5 or final_model.h5) were not found.")
        
        else:
            st.info("👆 Upload an image to get started")
            
            # Sample images or instructions
            st.subheader("📝 Tips for Best Results:")
            st.markdown("""
            - Use clear, well-lit images
            - Focus on the surface area
            - Avoid blurry or dark images
            - Image will be resized to 227×227 pixels
            - Supported formats: JPG, PNG, JPEG
            - Works with both color and grayscale images
            """)
            
            # Example results preview
            with st.expander("🎯 What to Expect"):
                st.markdown("""
                **Example Results:**
                - ✅ **No Crack**: High confidence in green box
                - 🚨 **Crack Detected**: High confidence in red box
                - 📊 **Confidence Scores**: Detailed probability breakdown
                - 📈 **Visualization**: Bar chart of predictions
                """)
                




                
                
    

if __name__ == "__main__":
    main()