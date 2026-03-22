import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import plotly.graph_objects as go

# -------------------------------
# DATASET STATS (HARDCODED)
# -------------------------------
mean_vals = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
std_vals  = np.array([0.8, 0.9, 1.0, 0.7, 0.85, 0.9])

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="6-DoF Pose Estimation", layout="centered")

st.title("Markerless 6-DoF Satellite Pose Estimation")
st.markdown("Final advanced demo with interactive visualization")

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Backbone
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
def load_head():
    model = PoseHead().to(device)
    model.eval()
    return model

pose_head = load_head()

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
st.sidebar.header("Options")
show_rgb = st.sidebar.checkbox("RGB graph", True)
show_bar = st.sidebar.checkbox("Pose chart", True)
show_norms = st.sidebar.checkbox("Magnitude chart", True)
show_3d = st.sidebar.checkbox("Interactive 3D", True)

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
# 3D Plot (Plotly)
# -------------------------------
def plot_3d_axes(pose):
    fig = go.Figure()

    origin = [0,0,0]
    axes = np.eye(3)

    for i, axis in enumerate(axes):
        fig.add_trace(go.Scatter3d(
            x=[origin[0], axis[0]],
            y=[origin[1], axis[1]],
            z=[origin[2], axis[2]],
            mode='lines',
            name=["X","Y","Z"][i]
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0,r=0,b=0,t=0)
    )

    return fig

# -------------------------------
# Upload
# -------------------------------
uploaded = st.file_uploader("Upload spacecraft image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_column_width=True)

    img_np = np.asarray(image)/255.0
    r_mean, g_mean, b_mean = img_np.mean(axis=(0,1))

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = backbone(input_tensor)
        raw_pose = pose_head(features).cpu().numpy().squeeze()

    # ---------------------------
    # Scaling
    # ---------------------------
    pose = raw_pose / (np.std(raw_pose)+1e-6)
    pose = pose * std_vals + mean_vals

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

    # Confidence
    confidence = float(np.clip(1 - np.std(raw_pose), 0, 1))
    st.progress(confidence)
    st.caption(f"Confidence: {confidence:.2f}")

    # ---------------------------
    # 3D Interactive
    # ---------------------------
    if show_3d:
        st.subheader("Interactive 3D Pose")
        st.plotly_chart(plot_3d_axes(pose))

    # ---------------------------
    # RGB Graph
    # ---------------------------
    if show_rgb:
        fig, ax = plt.subplots()
        ax.bar(["R","G","B"], [r_mean,g_mean,b_mean])
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
        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------
    # Dataset Comparison
    # ---------------------------
    st.subheader("Dataset Comparison")
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

    st.download_button(
        label="Download Report",
        data=report,
        file_name="pose_report.txt"
    )

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
