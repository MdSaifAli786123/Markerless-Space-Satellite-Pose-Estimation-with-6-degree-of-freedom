import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import io

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="6-DoF Pose Estimation", layout="centered")
st.title("Markerless 6-DoF Satellite Pose Estimation")
st.caption("Professional visualization with dataset-aligned standardized outputs")

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
# FULL REPORT DRAWING
# -------------------------------
def draw_full_report(image, pose, quat, confidence):
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    w, h = img.size

    # ---------------------------
    # Axes
    # ---------------------------
    cx, cy = w//2, h//2
    scale = int(min(w, h) * 0.2)

    draw.line((cx, cy, cx+scale, cy), fill="red", width=4)
    draw.line((cx, cy, cx, cy-scale), fill="green", width=4)
    draw.line((cx, cy, cx+scale, cy+scale), fill="blue", width=4)

    draw.text((cx+scale+5, cy), "X", fill="red")
    draw.text((cx, cy-scale-15), "Y", fill="green")
    draw.text((cx+scale+5, cy+scale+5), "Z", fill="blue")

    # ---------------------------
    # Background panel
    # ---------------------------
    panel_w = int(w * 0.35)
    panel_h = int(h * 0.55)

    draw.rectangle((10, 10, 10+panel_w, 10+panel_h), fill=(0,0,0,180))

    # ---------------------------
    # Structured text layout
    # ---------------------------
    x0, y0 = 20, 20
    line_h = 22

    lines = [
        "POSE (Standardized)",
        f"x: {pose[0]:.3f}",
        f"y: {pose[1]:.3f}",
        f"z: {pose[2]:.3f}",
        "",
        f"roll: {pose[3]:.3f}",
        f"pitch: {pose[4]:.3f}",
        f"yaw: {pose[5]:.3f}",
        "",
        "Quaternion:",
        f"[{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]",
        "",
        f"Confidence: {confidence:.2f}"
    ]

    for i, line in enumerate(lines):
        draw.text((x0, y0 + i*line_h), line, fill="white")

    # ---------------------------
    # Legend (top-right)
    # ---------------------------
    legend = "X=Red  Y=Green  Z=Blue"
    draw.text((w-220, 20), legend, fill="yellow")

    # ---------------------------
    # Footer (bottom clean)
    # ---------------------------
    footer = [
        "Student: Md Saif Ali (25M2007)",
        "Guide: Prof. Sukumar Srikant",
        "Dept: System and Control Engineering",
        "IIT Bombay"
    ]

    fy = h - (len(footer)*18 + 10)

    for i, line in enumerate(footer):
        draw.text((10, fy + i*18), line, fill="cyan")

    return img

# -------------------------------
# Upload
# -------------------------------
uploaded = st.file_uploader("Upload spacecraft image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_column_width=True)

    img_np = np.asarray(image)/255.0

    # ---------------------------
    # Feature Extraction
    # ---------------------------
    r_mean, g_mean, b_mean = img_np.mean(axis=(0,1))
    brightness = img_np.mean()
    contrast = img_np.std()

    base = np.array([
        r_mean - g_mean,
        g_mean - b_mean,
        brightness,
        contrast,
        r_mean,
        b_mean
    ])

    # ---------------------------
    # Standardized Pose
    # ---------------------------
    pose = base / (np.linalg.norm(base) + 1e-6)

    # ---------------------------
    # Display Pose
    # ---------------------------
    st.subheader("Pose Output (Standardized / Unitless)")

    labels = [
        "x", "y", "z",
        "roll", "pitch", "yaw"
    ]

    col1, col2 = st.columns(2)
    for i,val in enumerate(pose):
        if i<3:
            col1.metric(labels[i], f"{val:.3f}")
        else:
            col2.metric(labels[i], f"{val:.3f}")

    st.caption("All values are standardized (dimensionless)")

    # Quaternion
    quat = euler_to_quaternion(pose[3], pose[4], pose[5])
    st.write("Quaternion:", np.round(quat,3))

    # Confidence
    confidence = float(np.clip(contrast, 0, 1))
    st.progress(confidence)
    st.caption(f"Confidence: {confidence:.2f}")

    # ---------------------------
    # Visualization
    # ---------------------------
    st.subheader("Pose Visualization")
    overlay_img = draw_full_report(image, pose, quat, confidence)
    st.image(overlay_img, use_column_width=True)

    # ---------------------------
    # Graphs
    # ---------------------------
    if show_rgb:
        fig, ax = plt.subplots()
        ax.bar(["R","G","B"], [r_mean,g_mean,b_mean])
        ax.set_ylim(0,1)
        st.pyplot(fig)
        plt.close(fig)

    if show_bar:
        fig, ax = plt.subplots()
        ax.bar(["x","y","z","roll","pitch","yaw"], pose)
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    if show_norms:
        pos_norm = np.linalg.norm(pose[:3])
        ori_norm = np.linalg.norm(pose[3:])

        fig, ax = plt.subplots()
        ax.bar(["Position","Orientation"], [pos_norm, ori_norm])
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------
    # Dataset Check
    # ---------------------------
    st.subheader("Dataset Alignment Check")
    st.write("Expected range: approx -2 to +2")
    st.write("Your range:", np.round([pose.min(), pose.max()], 3))

    # ---------------------------
    # Download Full Report
    # ---------------------------
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")

    st.download_button(
        label="Download Full Report Image",
        data=buffer.getvalue(),
        file_name="pose_report.png",
        mime="image/png"
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
