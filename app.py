import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.neural_network import NeuralNetwork

st.set_page_config(page_title="Digit Recognizer Dashboard", page_icon="🔢", layout="wide")

@st.cache_resource
def load_model():
    try:
        with open("models/digit_recognizer.pkl", 'rb') as f:
            model_data = pickle.load(f)
        layer_sizes = model_data.get('layer_sizes', [784, 128, 64, 10])
        lr = model_data.get('learning_rate', 0.01)
        nn = NeuralNetwork(layer_sizes, learning_rate=lr)
        nn.weights = model_data['weights']
        nn.biases = model_data['biases']
        return nn
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

nn = load_model()

def predict_image(img_array, is_light_theme):
    if nn is None:
        return "Model not found.", None
        
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    
    if is_light_theme:
        gray = cv2.bitwise_not(gray)
        
    if np.max(gray) == 0:
        return "empty", None

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -5)
    
    kernel_dilate = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    img_h, img_w = thresh.shape
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        if x <= 5 or y <= 5 or (x + w) >= (img_w - 5) or (y + h) >= (img_h - 5):
            continue
        if area < 50 or h < 15:
            continue
            
        aspect_ratio = w / float(h)
        if aspect_ratio > 3.0 or aspect_ratio < 0.05:
            continue
            
        valid_contours.append((x, y, w, h, c))
            
    valid_contours = sorted(valid_contours, key=lambda b: b[0])
    output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    predictions = []
    
    for (x, y, w, h, c) in valid_contours:
        pad = max(5, int(0.1 * h))
        y1, y2 = max(0, y - pad), min(thresh.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(thresh.shape[1], x + w + pad)
        
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue
            
        roi_h, roi_w = roi.shape
        max_dim = max(roi_h, roi_w)
        pad_h = (max_dim - roi_h) // 2
        pad_w = (max_dim - roi_w) // 2
        
        squared_roi = np.pad(roi, ((pad_h, max_dim - roi_h - pad_h), (pad_w, max_dim - roi_w - pad_w)), mode='constant', constant_values=0)
        resized_20 = cv2.resize(squared_roi, (20, 20), interpolation=cv2.INTER_AREA)
        roi_resized = np.pad(resized_20, ((4, 4), (4, 4)), mode='constant', constant_values=0)
        
        M = cv2.moments(roi_resized)
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            shift_x = 13.5 - cX
            shift_y = 13.5 - cY
            translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            roi_resized = cv2.warpAffine(roi_resized, translation_matrix, (28, 28))
        
        roi_norm = roi_resized.astype(np.float32) / 255.0
        roi_flat = roi_norm.reshape(1, 784)
        
        pred_proba = nn.forward_propagation(roi_flat)
        digit = np.argmax(pred_proba, axis=1)[0]
        predictions.append(str(digit))
        
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output_img, str(digit), (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if not predictions:
        if is_light_theme:
            output_img = cv2.bitwise_not(output_img)
        return "No digits detected. Please draw a number.", output_img
        
    if is_light_theme:
        output_img = cv2.bitwise_not(output_img)
        
    return "Detected Sequence: " + "".join(predictions), output_img

st.markdown("""
<style>
.main .block-container {
    max-width: 100% !important;
    padding: 1rem 2rem !important;
}

[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlock"] > [data-testid="element-container"] {
    max-width: 100% !important;
    overflow: hidden !important;
}

iframe[title="streamlit_drawable_canvas.st_canvas"] {
    width: 800px !important;
    max-width: 100% !important;
    height: 400px !important;
    border-radius: 5px !important;
    margin: 0 auto !important;
    display: block !important;
}

div[data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}

div[data-testid="stImage"] > img {
    border-radius: 5px !important;
    margin: 0 auto !important;
}

@media (max-width: 768px) {
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)

col_title, col_theme = st.columns([0.8, 0.2])
with col_title:
    st.title("🔢 Handwritten Digit Recognizer")
with col_theme:
    st.markdown("<br>", unsafe_allow_html=True)
    theme = st.radio("Theme Mode", ["Dark", "Light"], horizontal=True, label_visibility="collapsed")

st.markdown("Draw a single digit or multiple digits below. The model will automatically process them left-to-right.")

if theme == "Dark":
    bg_color = "#000000"
    stroke_color = "#FFFFFF"
    is_light = False
else:
    bg_color = "#FFFFFF"
    stroke_color = "#000000"
    is_light = True

col_slider, col_clear = st.columns([0.85, 0.15])
with col_slider:
    stroke_width = st.slider("Stroke width", 10, 30, 20)
with col_clear:
    st.markdown("<br>", unsafe_allow_html=True)
    if 'canvas_key' not in st.session_state:
        st.session_state['canvas_key'] = 0
    if st.button("🗑️ Clear Canvas", use_container_width=True):
        st.session_state['canvas_key'] += 1

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=400,
    width=800,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state['canvas_key']}",
)

if st.button("Predict Sequence", type="primary"):
    if canvas_result.image_data is not None:
        img_array = canvas_result.image_data
        sequence_str, annotated_img = predict_image(img_array, is_light)
        
        if sequence_str == "empty":
            st.warning("Please draw something on the canvas first!")
        else:
            st.success(sequence_str)
            st.markdown("### How the Model Saw It")
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, width=800)
