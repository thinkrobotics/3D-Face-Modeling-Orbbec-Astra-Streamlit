# 3D Face Reconstruction from 2D Images using Depth Mapping

## Introduction

With advancements in computer vision and artificial intelligence, converting 2D images into 3D models has become an exciting area of research. This project utilizes image processing, depth estimation, and 3D point cloud visualization to reconstruct a 3D model of a human face from a 2D image.

## Objective

The goal of this project is to develop an application that allows users to:

1. Upload a 2D image of a person’s face.
2. Generate a depth map to estimate the distance of various facial features from the camera.
3. Convert the depth map into a 3D model that can be rotated and viewed interactively.

## System Workflow

The project is built using Streamlit, an interactive web-based framework, and Python, along with libraries such as OpenCV, PyTorch, NumPy, and Plotly for image processing and visualization.

### Step-by-Step Breakdown

1. **Image Upload:**
   - Users upload a JPEG or PNG image of a face.
   - The image is read and displayed in the application.

2. **Depth Estimation:**
   - A pre-trained depth estimation model (MiDaS) predicts the relative depth of each pixel.
   - The depth map is generated as a grayscale image, where lighter areas are closer to the camera and darker areas are farther.

3. **3D Model Generation:**
   - The depth map is converted into a 3D point cloud, assigning each pixel a 3D coordinate (X, Y, Z).
   - The face structure is refined by filtering noisy data and smoothing depth transitions.

4. **3D Visualization:**
   - The point cloud is rendered in 3D space using Plotly.
   - The model can be rotated, zoomed, and explored interactively within the Streamlit web app.

## Technical Components

### Tools & Technologies Used

- Python
- Streamlit (for UI)
- OpenCV (image processing)
- PyTorch (depth estimation with MiDaS)
- NumPy (numerical operations)
- Plotly (3D visualization)

### Project Structure

```
3D_Modeling_Project/
│── app.py                # Main Streamlit app
│── depth_estimation.py    # Depth estimation using MiDaS
│── generate_3d_model.py   # Converts depth map to 3D point cloud
│── assets/               # Stores images and depth maps
│── requirements.txt      # Dependencies and libraries
```

## Implementation Details

### 1. Depth Estimation

- The depth estimation model is based on MiDaS, a pre-trained neural network that predicts depth from a single image.
- The model processes the image and produces a depth map, assigning a depth value to each pixel.
- The depth values are normalized to maintain consistency across different images.

### 2. 3D Point Cloud Generation

- Using NumPy, the depth map is converted into X, Y, and Z coordinates.
- A mesh grid is created to map the pixels to a 3D coordinate system.
- The face structure is refined using Gaussian smoothing to reduce noise.

### 3. 3D Visualization with Plotly

- The 3D model is displayed as a scatter plot of points.
- The points are colored using a grayscale gradient, maintaining the black-and-white theme.
- The interactive viewer allows rotation, zooming, and panning.

## User Guide

### Step 1: Upload an Image

- Click the **Upload** button and select a JPEG or PNG image.
- The uploaded image will be displayed in the application.

### Step 2: Generate Depth Map

- Click the **Generate Depth Map** button.
- The application will process the image and display a grayscale depth map.

### Step 3: Convert to 3D Model

- Click the **Generate 3D Model** button.
- The 3D visualization of the face will be generated.
- Users can hover, rotate, and zoom into the model for a closer look.

## Conclusion

This project successfully demonstrates how a 2D image can be converted into a 3D model using depth estimation and point cloud visualization. The interactive viewer makes it easy for users to explore the 3D model, while the depth estimation algorithm ensures realistic structure.

With further improvements, this technology can be used in 3D animation, virtual reality, and medical imaging.


