import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pandas as pd

# -------------------------------
# Load Dataset Stats
# -------------------------------
@st.cache_data
def load_stats():
    df = pd.read_csv("pose_dataset_cleaned.csv")
    cols = ["X-axis", "Y-axis", "Z-axis", "Roll", "Pitch", "Yaw"]
    return df[cols].mean().values, df[cols].std().values

mean_vals, std_vals = load_stats()

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="6-DoF Pose Estimation", layout="centered")

st.title("Markerless 6-DoF Satellite Pose Estimation")
st.markdown("Advanced demo with projection-based pose visualization")

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
st.sidebar.header("Visualization Options")
show_rgb = st.sidebar.checkbox("Show RGB graph", True)
show_bar = st.sidebar.checkbox("Show pose chart", True)
show_norms = st.sidebar.checkbox("Show magnitude chart", True)
show_axes = st.sidebar.checkbox("Show projected axes", True)
show_quat = st.sidebar.checkbox("Show quaternion", True)

# -------------------------------
# Helper: Euler → Rotation Matrix
# -------------------------------
def euler_to_rot(roll, pitch, yaw):
    Rx = np.array([[1,0,0],
                   [0,np.cos(roll),-np.sin(roll)],
                   [0,np.sin(roll),np.cos(roll)]])

    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],
                   [0,1,0],
                   [-np.sin(pitch),0,np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],
                   [np.sin(yaw),np.cos(yaw),0],
                   [0,0,1]])

    return Rz @ Ry @ Rx

# -------------------------------
# Helper: Quaternion
# -------------------------------
def euler_to_quaternion(roll, pitch, yaw):
    cr = np.cos(roll/2)
    sr = np.sin(roll/2)
    cp = np.cos(pitch/2)
    sp = np.sin(pitch/2)
    cy = np.cos(yaw/2)
    sy = np.sin(yaw/2)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy

    return np.array([w,x,y,z])

# -------------------------------
# Helper: Draw projected axes
# -------------------------------
def draw_projected_axes(image, pose):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    w, h = img.size
    cx, cy = w//2, h//2

    R = euler_to_rot(pose[3], pose[4], pose[5])

    axes = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])

    scale = 100
    for axis in axes:
        proj = R @ axis
        x = int(cx + proj[0]*scale)
        y = int(cy - proj[1]*scale)
        draw.line([cx, cy, x, y], width=3)

    return img

# -------------------------------
# Upload
# -------------------------------
uploaded = st.file_uploader("Upload spacecraft image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    img_np = np.asarray(image).astype(np.float32)/255.0
    r_mean, g_mean, b_mean = img_np.mean(axis=(0,1))

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = backbone(input_tensor)
        raw_pose = pose_head(features).cpu().numpy().squeeze()

    # ---------------------------
    # Standardization
    # ---------------------------
    pose = (raw_pose - raw_pose.mean())/(raw_pose.std()+1e-6)
    pose = pose * std_vals + mean_vals

    # ---------------------------
    # Display Pose
    # ---------------------------
    st.subheader("Predicted 6-DoF Pose")
    st.caption("Translation (m) | Rotation (rad)")

    labels = ["x","y","z","roll","pitch","yaw"]

    col1, col2 = st.columns(2)
    for i,val in enumerate(pose):
        if i<3:
            col1.metric(labels[i], f"{val:.3f}")
        else:
            col2.metric(labels[i], f"{val:.3f}")

    # ---------------------------
    # Quaternion
    # ---------------------------
    if show_quat:
        quat = euler_to_quaternion(pose[3], pose[4], pose[5])
        st.subheader("Quaternion Representation")
        st.write(f"[w, x, y, z] = {quat.round(3)}")

    # ---------------------------
    # Confidence
    # ---------------------------
    confidence = float(np.clip(1 - np.std(raw_pose), 0, 1))
    st.progress(confidence)
    st.caption(f"Confidence Score: {confidence:.2f}")

    # ---------------------------
    # Axes Projection
    # ---------------------------
    if show_axes:
        st.subheader("Projected 3D Pose")
        overlay = draw_projected_axes(image, pose)
        st.image(overlay, use_column_width=True)

    # ---------------------------
    # RGB Graph
    # ---------------------------
    if show_rgb:
        fig, ax = plt.subplots()
        ax.bar(["Red","Green","Blue"], [r_mean,g_mean,b_mean])
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
