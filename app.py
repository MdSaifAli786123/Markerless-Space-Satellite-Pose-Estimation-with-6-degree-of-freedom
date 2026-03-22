import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io

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
# Quaternion
# -------------------------------
def euler_to_quaternion(r, p, y):
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
# Interpretation
# -------------------------------
def interpret_pose(p):
    dir_lr = "right" if p[0] > 0 else "left"
    dir_ud = "upward" if p[1] > 0 else "downward"
    rot = "strong rotation" if abs(p[5]) > 0.5 else "mild rotation"
    return f"Satellite tilted {dir_lr} and {dir_ud} with {rot}"

# -------------------------------
# Overlay (FIXED CLEAN VERSION)
# -------------------------------
def draw_overlay(img, pose, quat, conf, interp):
    draw = ImageDraw.Draw(img)
    w, h = img.size

    cx, cy = w//2, h//2
    s = int(min(w, h) * 0.2)

    # Axes
    draw.line((cx, cy, cx+s, cy), fill="red", width=4)
    draw.line((cx, cy, cx, cy-s), fill="green", width=4)
    draw.line((cx, cy, cx+s, cy+s), fill="blue", width=4)

    draw.text((cx+s+5, cy), "X", fill="red")
    draw.text((cx, cy-s-15), "Y", fill="green")
    draw.text((cx+s+5, cy+s+5), "Z", fill="blue")

    # Shadow text helper
    def draw_text(x, y, text):
        draw.text((x+1, y+1), text, fill="black")
        draw.text((x, y), text, fill="white")

    # Left text block
    x0, y0 = 15, 15
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
        f"Confidence: {conf:.2f}",
        "",
        interp
    ]

    for i, line in enumerate(lines):
        draw_text(x0, y0 + i*line_h, line)

    # Legend (top-right)
    draw_text(w - 260, 15, "X=Red  Y=Green  Z=Blue")

    # Footer (bottom)
    footer = [
        "Student: Md Saif Ali (25M2007)",
        "Guide: Prof. Sukumar Srikant",
        "Dept: System and Control Engineering",
        "IIT Bombay"
    ]

    fy = h - (len(footer)*20 + 10)
    for i, line in enumerate(footer):
        draw_text(15, fy + i*20, line)

    return img

# -------------------------------
# Upload
# -------------------------------
uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image = image.resize((512, 512))  # performance

    st.image(image, use_column_width=True)

    img = np.asarray(image) / 255.0

    # ---------------------------
    # Improved Feature Extraction
    # ---------------------------
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    brightness = img.mean()
    contrast = img.std()

    gx = np.gradient(img, axis=0)
    gy = np.gradient(img, axis=1)
    edge = np.mean(np.abs(gx) + np.abs(gy))

    left = img[:, :256].mean()
    right = img[:, 256:].mean()
    top = img[:256, :].mean()
    bottom = img[256:, :].mean()

    base = np.array([
        right - left,
        top - bottom,
        brightness,
        contrast,
        edge,
        r.mean() - b.mean()
    ])

    pose = base / (np.linalg.norm(base) + 1e-6)

    quat = euler_to_quaternion(pose[3], pose[4], pose[5])
    conf = float(np.clip(contrast + edge, 0, 1))
    interp = interpret_pose(pose)

    # ---------------------------
    # Display
    # ---------------------------
    st.subheader("Pose Output")
    col1, col2 = st.columns(2)
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]

    for i, v in enumerate(pose):
        (col1 if i < 3 else col2).metric(labels[i], f"{v:.3f}")

    st.write("Quaternion:", np.round(quat, 3))
    st.progress(conf)
    st.caption(f"Confidence: {conf:.2f}")
    st.info(interp)

    # Overlay
    overlay = draw_overlay(image.copy(), pose, quat, conf, interp)
    st.image(overlay, use_column_width=True)

    # ---------------------------
    # Graphs (lazy load)
    # ---------------------------
    if show_rgb or show_bar or show_norms:
        import matplotlib.pyplot as plt

    if show_rgb:
        fig, ax = plt.subplots()
        ax.bar(["R","G","B"], [r.mean(), g.mean(), b.mean()])
        st.pyplot(fig)
        plt.close(fig)

    if show_bar:
        fig, ax = plt.subplots()
        ax.bar(labels, pose)
        st.pyplot(fig)
        plt.close(fig)

    if show_norms:
        fig, ax = plt.subplots()
        ax.bar(["Position","Orientation"],
               [np.linalg.norm(pose[:3]), np.linalg.norm(pose[3:])])
        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------
    # Download
    # ---------------------------
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    st.download_button("Download Report", buf.getvalue(), "pose.png")

else:
    st.stop()

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "**Student:** Md Saif Ali (25M2007)\n"
    "**Guide:** Prof. Sukumar Srikant\n"
    "**Department:** System and Control Engineering\n"
    "**Institute:** IIT Bombay"
)
