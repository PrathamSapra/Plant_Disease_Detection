
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import ViTForImageClassification
import os
import cv2
import numpy as np
from PIL import Image

# --------------------------
# Set page config FIRST thing after imports
# --------------------------
st.set_page_config(page_title="🌿 Agri-Vision Plant Disease Detector", page_icon="🌿", layout="wide")

# --------------------------
# CSS Styling for all pages
# --------------------------
st.markdown(
    """
    <style>
    /* Import beautiful natural fonts from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@600&family=Quicksand:wght@400;600&display=swap');

    /* App background */
    .stApp {
        background-color: #e6f4e3;
        font-family: 'Quicksand', sans-serif;
        color: #3e3e3e;
    }

    /* Header styling - natural handwritten look */
    h1, h2, h3 {
        color: #2e7d32;
        font-family: 'Dancing Script', cursive;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #c8e6c9;
        font-family: 'Quicksand', sans-serif;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #8d6e63;
        color: white;
        border-radius: 12px;
        padding: 0.5em 1em;
        font-weight: bold;
        font-family: 'Quicksand', sans-serif;
        transition: 0.3s ease-in-out;
    }

    div.stButton > button:hover {
        background-color: #6d4c41;
        color: #ffffff;
    }

    /* Markdown text styling */
    .markdown-text-container {
        font-size: 17px;
        line-height: 1.6;
        font-family: 'Quicksand', sans-serif;
        color: #4b3e2f;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------
# Label Map with Colors (for badges)
# --------------------------
label_map = {
    0: ('Apple scab', '#F44336'),              # Red
    1: ('Apple Cedar rust', '#FF9800'),        # Orange
    2: ('Corn Common rust ', '#FFC107'),       # Amber
    3: ('Potato Early Blight', '#4CAF50'),     # Green
    4: ('Potato Healthy', '#2196F3'),           # Blue
    5: ('Tomato Early Blight', '#9C27B0'),     # Purple
    6: ('Tomato Yellow Leaf Curl Virus', '#00BCD4'), # Cyan
    7: ('Tomato Healthy', '#8BC34A')            # Light Green
}

# --------------------------
# Custom CNN Model Definition
# --------------------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(256 * 12 * 12, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --------------------------
# Helper Functions
# --------------------------
@st.cache(allow_output_mutation=True)
def load_models(device):
    models_dict = {}

    cnn_model = CustomCNN(num_classes=8)
    cnn_model.load_state_dict(torch.load(os.path.join(os.getcwd(), "saved_models", "baseline_custom_cnn_model.pth"), map_location=device))
    cnn_model.to(device).eval()
    models_dict["Custom CNN"] = (cnn_model, False)

    resnet_model = models.resnet50(pretrained=False)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 8)
    resnet_model.load_state_dict(torch.load(os.path.join(os.getcwd(), "saved_models", "transfer_resnet50_model.pth"), map_location=device))
    resnet_model.to(device).eval()
    models_dict["ResNet50"] = (resnet_model, False)

    vit_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k', num_labels=8
    )
    vit_model.load_state_dict(torch.load(os.path.join(os.getcwd(), "saved_models", "fine_tuned_vit_model.pth"), map_location=device))
    vit_model.to(device).eval()
    models_dict["Vision Transformer (ViT)"] = (vit_model, True)

    return models_dict

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

def detect_disease_regions(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_bound = np.array([10, 100, 100])
    upper_bound = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = image_np.copy()
    cv2.drawContours(annotated, contours, -1, (255, 0, 0), 2)
    return annotated

def predict(image_tensor, model, is_vit, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor).logits if is_vit else model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), label_map[pred.item()][0], conf.item(), label_map[pred.item()][1]

# --------------------------
# Page Navigation Setup
# --------------------------
st.sidebar.title("🌿 Dashboard")
page = st.sidebar.selectbox("Navigate", ["Home", "About", "Disease Recognition"])

# --------------------------
# Home Page
# --------------------------
if page == "Home":
    st.title("🌱 AGRI-VISION: PLANT DISEASE RECOGNITION SYSTEM")
    st.markdown("""
    <div class="markdown-text-container">
    Welcome to the Plant Disease Recognition System!<br><br>
    Our mission is to help identify plant diseases efficiently. Upload an image of a plant leaf, and our system will analyze it to detect any signs of disease.

    ### 🌼 How It Works
    1. <b>Upload Image:</b> Go to the <b>Disease Recognition</b> page and upload an image.
    2. <b>Analysis:</b> Our models process the image using deep learning.
    3. <b>Results:</b> Get the disease name instantly!

    ### 🌿 Why Choose Us?
    - High Accuracy
    - Simple Interface
    - Fast Results

    👉 Head over to the <b>Disease Recognition</b> tab to begin!
    </div>
    """, unsafe_allow_html=True)

# --------------------------
# About Page
# --------------------------
elif page == "About":
    st.title("📊 About")
    st.markdown("""
    <div class="markdown-text-container">
    <h4>About Dataset</h4>
    This dataset was augmented offline from the original public dataset.<br>
    It contains ~87,000 RGB images of plant leaves, both healthy and diseased.<br><br>

    <b>Split:</b>
    <ul>
        <li>Train: 70,295 images</li>
        <li>Validation: 17,572 images</li>
        <li>Test: 33 images</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# --------------------------
# Disease Recognition Page
# --------------------------
elif page == "Disease Recognition":
    st.title("🌿 Agri-Vision: Plant Disease Detection App")
    st.write(
        "Upload a leaf image to detect the plant disease and highlight affected regions."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device)

    uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])
    model_names = st.multiselect(
        "Select one or more models to use for prediction",
        options=list(models.keys()),
        default=list(models.keys())
    )

    if uploaded_file:
        img_pil = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Uploaded Image")
            st.image(img_pil, use_column_width=True, caption="Original Image")

        if st.button("🔍 Predict"):
            image_tensor = preprocess_image(img_pil)
            image_np = np.array(img_pil)
            highlighted = detect_disease_regions(image_np)

            with col2:
                st.subheader("Detected Disease Regions")
                st.image(highlighted, use_column_width=True)

            st.markdown("---")
            st.header("Model Predictions")

            prediction_cols = st.columns(len(model_names))

            best_conf = -1
            best_pred = None

            for idx, model_name in enumerate(model_names):
                model, is_vit = models[model_name]
                pred_idx, pred_class, confidence, color = predict(image_tensor, model, is_vit, device)

                with prediction_cols[idx]:
                    st.markdown(f"### {model_name}")
                    st.markdown(
                        f'<p style="background-color:{color}; padding:8px; border-radius:8px; color:white; text-align:center;">'
                        f'<b>{pred_class}</b></p>', unsafe_allow_html=True
                    )
                    st.metric(label="Confidence", value=f"{confidence:.2%}")

                if confidence > best_conf:
                    best_conf = confidence
                    best_pred = (pred_class, confidence, color, model_name)

            if best_pred is not None:
                st.markdown("---")
                st.header("🏆 Best Prediction")

                pred_class, confidence, color, model_name = best_pred
                st.markdown(
                    f'<div style="background-color:{color}; padding:20px; border-radius:15px; color:white; text-align:center;">'
                    f'<h2>Model: {model_name}</h2>'
                    f'<h1 style="font-size:48px; margin:10px 0;">{pred_class}</h1>'
                    f'<h3>Confidence: {confidence:.2%}</h3>'
                    f'</div>', unsafe_allow_html=True
                )
    else:
        st.info("Please upload an image to start prediction.")
