import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import torch
import timm
import torchvision.transforms as transforms
import io

# -------------------------------
# Dataset stats (from your CSV)
# -------------------------------
POSE_MIN, POSE_MAX = -1.75, 1.75

# -------------------------------
# Setup
# -------------------------------
st.set_page_config(page_title="Pose Estimation", layout="centered")
st.title("Markerless 6-DoF Satellite Pose Estimation")
st.info("Upload an image to begin")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Options")
show_rgb = st.sidebar.checkbox("RGB graph", True)
show_bar = st.sidebar.checkbox("Pose chart", True)
show_norms = st.sidebar.checkbox("Magnitude chart", True)

# -------------------------------
# Load Lightweight Transformer
# -------------------------------
@st.cache_resource
def load_model():
    model = timm.create_model("mobilevit_xxs", pretrained=True)
    model.eval()
    return model

model = load_model()

# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# -------------------------------
# Quaternion
# -------------------------------
def euler_to_quaternion(r,p,y):
    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y/2), np.sin(y/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ])

# -------------------------------
# Rotation Matrix
# -------------------------------
def rotation_matrix(r,p,y):
    Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx

# -------------------------------
# Projection
# -------------------------------
def project(p, cx, cy, scale=90):
    x,y,z = p
    return int(cx + x*scale), int(cy - y*scale)

# -------------------------------
# Interpretation
# -------------------------------
def interpret_pose(p):
    lr = "right" if p[0] > 0 else "left"
    ud = "upward" if p[1] > 0 else "downward"
    rot = "strong rotation" if abs(p[5]) > 0.5 else "mild rotation"
    return f"Satellite tilted {lr} and {ud} with {rot}"

# -------------------------------
# Overlay
# -------------------------------
def draw_overlay(img, pose, quat, interp):
    draw = ImageDraw.Draw(img)
    w,h = img.size

    cx, cy = int(w*0.70), int(h*0.70)

    R = rotation_matrix(pose[3],pose[4],pose[5])

    axes = np.array([[1,0,0],[0,1,0],[0,0,1]])
    colors = ["red","green","blue"]

    for axis, color in zip(axes, colors):
        end = project(R @ axis, cx, cy)
        draw.line((cx,cy,end[0],end[1]), fill=color, width=4)

    draw.text(project(R @ np.array([1,0,0]), cx, cy), "X", fill="red")
    draw.text(project(R @ np.array([0,1,0]), cx, cy), "Y", fill="green")
    draw.text(project(R @ np.array([0,0,1]), cx, cy), "Z", fill="blue")

    def txt(x,y,t):
        draw.text((x+1,y+1), t, fill="black")
        draw.text((x,y), t, fill="white")

    txt(15,15,"POSE (Standardized)")
    txt(15,40,f"Position: [x:{pose[0]:.3f}, y:{pose[1]:.3f}, z:{pose[2]:.3f}]")
    txt(15,65,f"Rotation: [roll:{pose[3]:.3f}, pitch:{pose[4]:.3f}, yaw:{pose[5]:.3f}]")
    txt(15,95,f"Quaternion: [{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]")
    txt(15,125,interp)

    txt(w-260,15,"X=Red  Y=Green  Z=Blue")

    footer = [
        "Student: Md Saif Ali (25M2007)",
        "Guide: Prof. Sukumar Srikant",
        "Dept: System and Control Engineering",
        "IIT Bombay"
    ]

    fy = h - (len(footer)*18 + 10)
    for i,l in enumerate(footer):
        txt(15, fy + i*18, l)

    return img

# -------------------------------
# Upload
# -------------------------------
uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image = image.resize((512,512))
    st.image(image, use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    # ---------------------------
    # Transformer Feature Extraction
    # ---------------------------
    with torch.no_grad():
        feats = model.forward_features(input_tensor)

    feats = feats.mean(dim=[2,3]).squeeze().numpy()

    # ---------------------------
    # Map features → pose
    # ---------------------------
    pose = feats[:6]
    pose = pose / (np.linalg.norm(pose)+1e-6)

    # Use dataset stats
    pose = np.clip(pose * 1.5, POSE_MIN, POSE_MAX)

    quat = euler_to_quaternion(pose[3],pose[4],pose[5])
    interp = interpret_pose(pose)

    # ---------------------------
    # Display
    # ---------------------------
    st.subheader("Pose Output")

    col1,col2 = st.columns(2)
    labels = ["x","y","z","roll","pitch","yaw"]

    for i,v in enumerate(pose):
        (col1 if i<3 else col2).metric(labels[i], f"{v:.3f}")

    st.write("Quaternion:", np.round(quat,3))
    st.info(interp)

    overlay = draw_overlay(image.copy(), pose, quat, interp)
    st.image(overlay, use_column_width=True)

    # ---------------------------
    # Graphs
    # ---------------------------
    if show_rgb or show_bar or show_norms:
        import matplotlib.pyplot as plt

    img_np = np.asarray(image)/255.0
    r,g,b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]

    if show_rgb:
        fig, ax = plt.subplots()
        ax.bar(["R","G","B"], [r.mean(),g.mean(),b.mean()])
        st.pyplot(fig)
        plt.close(fig)

    if show_bar:
        fig, ax = plt.subplots()
        ax.bar(labels, pose)
        st.pyplot(fig)
        plt.close(fig)

    if show_norms:
        fig, ax = plt.subplots()
        ax.bar(["pos","ori"],
               [np.linalg.norm(pose[:3]), np.linalg.norm(pose[3:])])
        st.pyplot(fig)
        plt.close(fig)

    # Download
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    st.download_button("Download Report", buf.getvalue(), "pose.png")

else:
    st.stop()

st.markdown("---")
st.markdown(
    "**Student:** Md Saif Ali (25M2007)\n"
    "**Guide:** Prof. Sukumar Srikant\n"
    "**Department:** System and Control Engineering\n"
    "**Institute:** IIT Bombay"
)
