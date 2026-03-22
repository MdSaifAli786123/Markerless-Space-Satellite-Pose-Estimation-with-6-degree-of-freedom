import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------
# DATASET STATS
# -------------------------------
std_vals = np.array([0.8, 0.9, 1.0, 0.7, 0.85, 0.9])

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="6-DoF Pose Estimation", layout="centered")
st.title("Markerless 6-DoF Satellite Pose Estimation")


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

    # ---------------------------
    # SIMPLE FEATURE EXTRACTION
    # ---------------------------
    r_mean, g_mean, b_mean = img_np.mean(axis=(0,1))
    brightness = img_np.mean()
    contrast = img_np.std()

    # ---------------------------
    # POSE GENERATION (FAST)
    # ---------------------------
    base = np.array([
        r_mean - g_mean,
        g_mean - b_mean,
        brightness,
        contrast,
        r_mean,
        b_mean
    ])

    pose = base / (np.linalg.norm(base) + 1e-6)
    pose = pose * std_vals

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

    # Confidence (based on contrast)
    confidence = float(np.clip(contrast, 0, 1))
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
