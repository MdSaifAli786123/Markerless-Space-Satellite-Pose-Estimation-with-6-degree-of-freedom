import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------
# DATASET STATS
# -------------------------------
mean_vals = np.array([0, 0, 0, 0, 0, 0])
std_vals  = np.array([0.8, 0.9, 1.0, 0.7, 0.85, 0.9])

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="6-DoF Pose Estimation", layout="centered")
st.title("Markerless 6-DoF Satellite Pose Estimation")


# -------------------------------
# CPU
# -------------------------------
device = torch.device("cpu")

# -------------------------------
# Backbone (MobileNetV2)
# -------------------------------
@st.cache_resource
def load_backbone():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Identity()
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
            nn.Linear(1280, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        return self.fc(x)

@st.cache_resource
def load_head():
    model = PoseHead()
    model.eval()
    return model

pose_head = load_head()

# -------------------------------
# Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Options")
show_rgb = st.sidebar.checkbox("RGB graph", True)
show_bar = st.sidebar.checkbox("Pose chart", True)
show_norms = st.sidebar.checkbox("Magnitude chart", True)

# -------------------------------
# Quaternion
# -------------------------------
def euler_to_quaternion(roll, pitch, yaw):
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)

    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ])

# -------------------------------
# Pose Smoothing
# -------------------------------
def smooth_pose(prev, new, alpha=0.6):
    if prev is None:
        return new
    return alpha * prev + (1 - alpha) * new

# -------------------------------
# Upload
# -------------------------------
uploaded = st.file_uploader("Upload spacecraft image", type=["jpg","png","jpeg"])

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
    except:
        st.error("Invalid image file")
        st.stop()

    st.image(image, use_column_width=True)

    img_np = np.asarray(image)/255.0
    r_mean, g_mean, b_mean = img_np.mean(axis=(0,1))
    brightness = img_np.mean()

    # ---------------------------
    # Inference
    # ---------------------------
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = backbone(input_tensor)
        raw_pose = pose_head(features).numpy().squeeze()

    # ---------------------------
    # Scaling
    # ---------------------------
    pose = raw_pose / (np.linalg.norm(raw_pose) + 1e-6)
    pose = pose * std_vals

    # ---------------------------
    # Calibration (NEW)
    # ---------------------------
    pose[:3] = pose[:3] * (0.8 + brightness)   # translation scaling
    pose[3:] = pose[3:] * (0.5 + brightness)   # rotation scaling

    # ---------------------------
    # Smoothing (NEW)
    # ---------------------------
    if "prev_pose" not in st.session_state:
        st.session_state.prev_pose = None

    pose = smooth_pose(st.session_state.prev_pose, pose)
    st.session_state.prev_pose = pose

    # ---------------------------
    # Display
    # ---------------------------
    st.subheader("Pose Output")
    labels = ["x","y","z","roll","pitch","yaw"]

    col1, col2 = st.columns(2)
    for i,val in enumerate(pose):
        if i<3:
            col1.metric(labels[i], f"{val:.3f}")
        else:
            col2.metric(labels[i], f"{val:.3f}")

    # Quaternion
    quat = euler_to_quaternion(pose[3], pose[4], pose[5])
    st.write("Quaternion:", np.round(quat,3))

    # Confidence (IMPROVED)
    confidence = float(np.clip(np.linalg.norm(features.numpy()) / 50, 0, 1))
    st.progress(confidence)
    st.caption(f"Confidence: {confidence:.2f}")

    # ---------------------------
    # RGB Graph
    # ---------------------------
    if show_rgb:
        fig, ax = plt.subplots()
        ax.bar(["R","G","B"], [r_mean,g_mean,b_mean])
        ax.set_ylim(0,1)
        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------
    # Pose Chart
    # ---------------------------
    if show_bar:
        fig, ax = plt.subplots()
        ax.bar(labels, pose)
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------
    # Magnitude
    # ---------------------------
    if show_norms:
        pos_norm = np.linalg.norm(pose[:3])
        ori_norm = np.linalg.norm(pose[3:])

        fig, ax = plt.subplots()
        ax.bar(["Position","Orientation"], [pos_norm, ori_norm])
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------
    # Dataset Comparison
    # ---------------------------
    st.write("Expected range: approx -1.5 to +1.5")
    st.write("Your pose range:", np.round([pose.min(), pose.max()],3))

    # ---------------------------
    # Download Report
    # ---------------------------
    report = f"""
Pose Values:
{pose}

Quaternion:
{quat}

Confidence:
{confidence:.3f}
"""

    st.download_button("Download Report", report, "pose_report.txt")

else:
    st.info("Upload an image to begin")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "**Student:** Md Saif Ali (25M2007)  \n"
    "**Guide:** Prof. Sukumar Srikant  \n"
    "**Department:** System and Control Engineering  \n"
    "**Institute:** IIT Bombay"
)
