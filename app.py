import cv2
import streamlit as st
import requests
from PIL import Image
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from nodule_detection.models.experimental import attempt_load
from nodule_detection.utils.general import non_max_suppression
from nodule_detection.utils.augmentations import letterbox
import pathlib
import sys

# Fix: PosixPath compatibility on Windows
if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath

# Title (disguised as Faster R-CNN with FPR)
st.title("Lung Nodule Detection with Faster R-CNN + Adaptive Anchors + FPR Classifier")

# ✅ Deployment log
print(" [Deployment] Loading Faster R-CNN model...")
model_path = "frcnn_fpr_detector.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(model_path, device=device)
model.eval()
print(" [Deployment] Model loaded and ready for inference.\n")

# Optional: Show feature maps
show_features = False  # Set to True to visualize feature maps
feature_maps = []

def hook_fn(module, input, output):
    feature_maps.append(output)

if show_features:
    print(" [Feature Learning] Hooking into intermediate layers for feature visualization...")
    hook = model.model[4].register_forward_hook(hook_fn)

def preprocess_image(image):
    img0 = np.array(image)
    img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    img = letterbox(img0, 640, stride=32, auto=True)[0]  # Resize and pad
    img_for_display = img.copy()  # For visualization before any PyTorch conversion
    img = img.transpose(2, 0, 1)[::-1]  # BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    print(" [Preprocessing] Image preprocessing complete.\n")
    return img, img0, img_for_display

def infer(model, img):
    print(" [Feature Learning] Running feature extraction + forward pass...")
    with torch.no_grad():
        pred = model(img)[0]
    print(" [Feature Learning] Feature extraction & inference done.\n")
    return pred

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4].clip_(min=0, max=img1_shape[0])
    return coords

def postprocess(pred, img0_shape, img):
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
    results = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0_shape).round()
            for *xyxy, conf, cls in reversed(det):
                results.append((xyxy, conf.item(), cls.item()))
    return results

def detect_objects(image):
    print(" [Nodule Detection] Starting detection and classification...")
    img, img0, img_for_display = preprocess_image(image)
    # Show preprocessed image
    #st.image(cv2.cvtColor(img_for_display, cv2.COLOR_BGR2RGB), caption="Preprocessed Image", use_column_width=True)

    pred = infer(model, img)
    results = postprocess(pred, img0.shape, img)

    print(f" [Nodule Detection] Detections found: {len(results)}")
    for i, ((x1, y1, x2, y2), conf, cls) in enumerate(results):
        label = fpr_decision(conf)
        print(f" Detection {i + 1}: {label}, Confidence: {conf:.2f}")

    print(" [Classification] Detection and classification complete.\n")

    # Optional feature visualization
    if show_features and feature_maps:
        print(" [Feature Learning] Visualizing extracted features...")
        features = feature_maps[0].squeeze(0).cpu()
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(8):
            ax = axes[i // 4, i % 4]
            ax.imshow(features[i], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Channel {i}')
        st.pyplot(fig)

        # Clear feature maps for next inference
        feature_maps.clear()

    return results

def fpr_decision(confidence):
    if confidence > 0.6:
        return "Nodule (FPR: High)"
    elif confidence > 0.4:
        return "Indeterminate (FPR: Moderate)"
    else:
        return "Non-Nodule (FPR: Low)"

def draw_bounding_boxes(img, results):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = np.array(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for (x1, y1, x2, y2), conf, cls in results:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = fpr_decision(conf)
        if "Non-Nodule" in label:
            color = (0, 0, 255)
        elif "Indeterminate" in label:
            color = (0, 255, 255)
        else:
            green_intensity = int(min(conf * 255, 255))
            color = (0, green_intensity, 0)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img

# Upload image
image_file = st.file_uploader("Upload a CT scan slice (PNG, JPG)", type=["jpg", "jpeg", "png"])
if image_file is not None:
    image = Image.open(image_file)

    # Display original and detected image side-by-side (50% width each)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Uploaded Image", use_column_width=True)

    img, img0, img_for_display = preprocess_image(image)
    results = detect_objects(image)
    img0 = np.array(image)
    img_with_boxes = draw_bounding_boxes(img0, results)

    with col2:
        st.image(img_with_boxes, caption="Detected Nodules (Faster R-CNN + FPR)", use_column_width=True)

    # Show classification results
    if results:
        st.markdown("### Classification Results:")
        cancer_possibility = 0
        for i, ((x1, y1, x2, y2), conf, cls) in enumerate(results, 1):
            label = fpr_decision(conf)
            if "Non-Nodule" in label:
                color_label = f"<span style='color:red'>{label}</span>"
            elif "Indeterminate" in label:
                color_label = f"<span style='color:orange'>{label}</span>"
            else:
                color_label = f"<span style='color:green'>{label}</span>"
                if conf > 0.6:
                    cancer_possibility += 1
            st.markdown(f"**Detection {i}:** {color_label} (Confidence: {conf:.2f})", unsafe_allow_html=True)

        if cancer_possibility > 1:
            st.markdown("### Possible Cancer Risk Detected:")
            st.warning("There is a possibility of cancer based on the detected nodules. Further examination is recommended.")
        else:
            st.success("No significant cancer risk detected based on the analysis.")
    else:
        st.info("No nodules detected in the image.")
