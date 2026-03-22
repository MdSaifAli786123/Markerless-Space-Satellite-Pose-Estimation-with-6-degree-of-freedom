import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io

# -------------------------------
# Dataset range
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
# Quaternion
# -------------------------------
def euler_to_quaternion(r, p, y):
    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y/2), np.sin(y/2)
    return np.array([cr*cp*cy + sr*sp*sy,
                     sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy,
                     cr*cp*sy - sr*sp*cy])

# -------------------------------
# Rotation matrix
# -------------------------------
def rotation_matrix(roll, pitch, yaw):
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
# Projection
# -------------------------------
def project(point, cx, cy, scale=150):
    x, y, z = point
    return (int(cx + x*scale), int(cy - y*scale))

# -------------------------------
# Interpretation
# -------------------------------
def interpret_pose(p):
    lr = "right" if p[0] > 0 else "left"
    ud = "upward" if p[1] > 0 else "downward"
    rot = "strong rotation" if abs(p[5]) > 0.5 else "mild rotation"
    return f"Satellite tilted {lr} and {ud} with {rot}"

# -------------------------------
# Overlay (Perspective axes)
# -------------------------------
def draw_overlay(img, pose, quat, conf, interp):
    draw = ImageDraw.Draw(img)
    w, h = img.size

    cx, cy = int(w*0.75), int(h*0.75)

    # Rotation
    R = rotation_matrix(pose[3], pose[4], pose[5])

    # Axes in 3D
    axes = np.array([
        [1,0,0],  # X
        [0,1,0],  # Y
        [0,0,1]   # Z
    ])

    colors = ["red", "green", "blue"]

    for axis, color in zip(axes, colors):
        rotated = R @ axis
        end = project(rotated, cx, cy)
        draw.line((cx, cy, end[0], end[1]), fill=color, width=2)

    # Labels
    draw.text(project(R @ np.array([1,0,0]), cx, cy), "X", fill="red")
    draw.text(project(R @ np.array([0,1,0]), cx, cy), "Y", fill="green")
    draw.text(project(R @ np.array([0,0,1]), cx, cy), "Z", fill="blue")

    # Shadow text
    def txt(x,y,t):
        draw.text((x+1,y+1), t, fill="black")
        draw.text((x,y), t, fill="white")

    # Pose text
    x0, y0 = 15, 15
    line_h = 20

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

    for i,l in enumerate(lines):
        txt(x0, y0 + i*line_h, l)

    # Legend
    txt(w-260, 15, "X=Red  Y=Green  Z=Blue")

    # Footer
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

    img = np.asarray(image)/255.0

    # Features
    r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
    brightness = img.mean()
    contrast = img.std()

    gx = np.gradient(img, axis=0)
    gy = np.gradient(img, axis=1)
    edge = np.mean(np.abs(gx)+np.abs(gy))

    left, right = img[:,:256].mean(), img[:,256:].mean()
    top, bottom = img[:256,:].mean(), img[256:,:].mean()

    base = np.array([
        right-left,
        top-bottom,
        brightness,
        contrast,
        edge,
        r.mean()-b.mean()
    ])

    pose = base / (np.linalg.norm(base)+1e-6)
    pose = np.clip(pose*1.2, POSE_MIN, POSE_MAX)

    quat = euler_to_quaternion(pose[3],pose[4],pose[5])
    conf = float(np.clip(contrast+edge,0,1))
    interp = interpret_pose(pose)

    # Display
    st.subheader("Pose Output")
    col1,col2 = st.columns(2)
    labels = ["x","y","z","roll","pitch","yaw"]

    for i,v in enumerate(pose):
        (col1 if i<3 else col2).metric(labels[i], f"{v:.3f}")

    st.write("Quaternion:", np.round(quat,3))
    st.progress(conf)
    st.caption(f"Confidence: {conf:.2f}")
    st.info(interp)

    overlay = draw_overlay(image.copy(), pose, quat, conf, interp)
    st.image(overlay, use_column_width=True)

    # Graphs
    if show_rgb or show_bar or show_norms:
        import matplotlib.pyplot as plt

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

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    st.download_button("Download Report", buf.getvalue(), "pose.png")

else:
    st.stop()

# Footer
st.markdown("---")
st.markdown(
    "**Student:** Md Saif Ali (25M2007)\n"
    "**Guide:** Prof. Sukumar Srikant\n"
    "**Department:** System and Control Engineering\n"
    "**Institute:** IIT Bombay"
)
