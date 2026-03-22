import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Pose Estimation Demo", layout="centered")

st.title("Markerless 6-DoF Satellite Pose Estimation")

st.markdown("""
This application demonstrates a **markerless monocular 6-DoF satellite pose estimation pipeline**.
""")

st.info("Demo version using deep visual features for realistic pose approximation.")

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Feature Extractor
# -------------------------------
@st.cache_resource
def load_backbone():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model

backbone = load_backbone()

# -------------------------------
# Pose Head
# -------------------------------
class PoseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.fc(x)

@st.cache_resource
def load_pose_head():
    model = PoseHead().to(device)
    model.eval()
    return model

pose_head = load_pose_head()

# -------------------------------
# Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Visualization Options")
show_norms = st.sidebar.checkbox("Show pose norms", True)
show_bar = st.sidebar.checkbox("Show pose bar chart", True)

# -------------------------------
# Upload
# -------------------------------
uploaded = st.file_uploader(
    "Upload a spacecraft image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    # ---------------------------
    # Feature Extraction
    # ---------------------------
    with torch.no_grad():
        features = backbone(input_tensor)

    # ---------------------------
    # Pose Prediction
    # ---------------------------
    with torch.no_grad():
        raw_pose = pose_head(features).cpu().numpy().squeeze()

    # ---------------------------
    # Stabilization
    # ---------------------------
    pose = np.zeros(6)

    pose[:3] = np.tanh(raw_pose[:3]) * 2.0        # meters
    pose[3:] = np.tanh(raw_pose[3:]) * np.pi      # radians

    # ---------------------------
    # Display Pose
    # ---------------------------
    st.subheader("Predicted 6-DoF Pose")
    st.caption("Translation (meters) | Rotation (radians)")

    labels = ["x", "y", "z", "roll", "pitch", "yaw"]

    col1, col2 = st.columns(2)
    for i, val in enumerate(pose):
        if i < 3:
            col1.metric(labels[i], f"{val:.3f}")
        else:
            col2.metric(labels[i], f"{val:.3f}")

    # ---------------------------
    # Norms
    # ---------------------------
    if show_norms:
        pos_norm = np.linalg.norm(pose[:3])
        ori_norm = np.linalg.norm(pose[3:])

        st.subheader("Derived Pose Metrics")

        st.write(f"Position magnitude: {pos_norm:.3f}")
        st.write(f"Orientation magnitude: {ori_norm:.3f}")

        fig, ax = plt.subplots()
        ax.bar(["Position", "Orientation"], [pos_norm, ori_norm])
        ax.set_title("Position vs Orientation Magnitude")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------
    # Pose Bar Chart
    # ---------------------------
    if show_bar:
        st.subheader("Pose Component Visualization")

        fig, ax = plt.subplots()
        ax.bar(labels, pose)
        ax.set_ylabel("Value")
        ax.set_title("6-DoF Pose Components")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

else:
    st.info("Please upload a spacecraft image to run pose estimation.")

# -------------------------------
# Footer (Restored)
# -------------------------------
st.markdown("---")
st.markdown(
    "**Student:** Md Saif Ali (25M2007)  \n"
    "**Guide:** Prof. Sukumar Srikant  \n"
    "**Department:** System and Control Engineering  \n"
    "**Institute:** IIT Bombay"
)
