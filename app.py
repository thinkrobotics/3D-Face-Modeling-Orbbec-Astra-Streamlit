import streamlit as st
import cv2
import numpy as np
import os
import plotly.graph_objects as go
from depth_estimation import estimate_depth
from generate_3d_model import generate_3d_model

st.title("📸 Image to 3D Face Reconstruction")

# ✅ Ensure the "assets/" directory exists before saving files
assets_dir = "assets"
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_path = os.path.join(assets_dir, uploaded_file.name)

    # ✅ Save the uploaded image
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ✅ Display uploaded image
    st.image(image_path, caption="📷 Uploaded Image", use_column_width=True)

    if st.button("🔍 Generate Depth Map"):
        try:
            depth_map = estimate_depth(image_path)

            # ✅ Ensure depth map is valid before processing
            if depth_map is None or len(depth_map.shape) != 2:
                st.error("❌ Error: Depth map has incorrect dimensions.")
            else:
                depth_map_path = os.path.join(assets_dir, "depth_map.png")

                # ✅ Convert depth map to 8-bit and save it
                depth_map_8bit = (depth_map * 255).astype(np.uint8)
                cv2.imwrite(depth_map_path, depth_map_8bit)

                # ✅ Display depth map
                st.image(depth_map_path, caption="🗺️ Depth Map", use_column_width=True)

                # ✅ Store the depth map path in session state
                st.session_state["depth_map_path"] = depth_map_path
                st.success("✅ Depth map generated successfully!")

        except Exception as e:
            st.error(f"❌ Error generating depth map: {e}")

if st.button("🎨 Generate 3D Model"):
    if "depth_map_path" in st.session_state:
        depth_map_path = st.session_state["depth_map_path"]

        try:
            # ✅ Generate 3D model
            X, Y, Z = generate_3d_model(depth_map_path, image_path)

            # ✅ Create an interactive 3D plot using Plotly
            fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="gray")])
            fig.update_layout(
                title="🌀 3D Model",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Depth",
                ),
                margin=dict(l=0, r=0, t=30, b=0),
            )

            # ✅ Display the 3D model
            st.plotly_chart(fig, use_container_width=True)
            st.success("✅ 3D model displayed successfully!")

        except Exception as e:
            st.error(f"❌ Error generating 3D model: {e}")

    else:
        st.warning("⚠️ Please generate a depth map first.")
