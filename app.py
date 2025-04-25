import cv2
import streamlit as st
import requests
from PIL import Image
import os
import torch
import numpy as np
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

# Load simulated Faster R-CNN model
model_path = "frcnn_fpr_detector.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(model_path, device=device)
model.eval()

def preprocess_image(image):
    img0 = np.array(image)
    img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    img = letterbox(img0, 640, stride=32, auto=True)[0]
    img = img.transpose(2, 0, 1)[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, img0

def infer(model, img):
    with torch.no_grad():
        pred = model(img)[0]
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
    img, img0 = preprocess_image(image)
    pred = infer(model, img)
    results = postprocess(pred, img0.shape, img)
    return results

def fpr_decision(confidence):
    if confidence > 0.6:
        return "Nodule (FPR: High)"
    elif confidence > 0.4:
        return "Indeterminate (FPR: Moderate)"
    else:
        return "Non-Nodule (FPR: Low)"

def draw_bounding_boxes(img, results):
    # Convert image to BGR format if it is in RGB (PIL Image to OpenCV)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = np.array(img)  # Convert PIL image to numpy array if needed
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for (x1, y1, x2, y2), conf, cls in results:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = fpr_decision(conf)

        # Adjust color based on classification and confidence (BGR format)
        if "Non-Nodule (FPR: Low)" in label:
            color = (0, 0, 255)  # Red in BGR for Non-Nodule
        elif "Indeterminate (FPR: Moderate)" in label:
            color = (0, 255, 255)  # Yellow in BGR for Indeterminate
        else:
            # Green for Nodule, with more intensity for higher confidence
            green_intensity = int(min(conf * 255, 255))  # Scale the confidence to a color intensity
            color = (0, green_intensity, 0)  # Green in BGR (note the reversed order)

        # Draw the bounding box with the chosen color
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    return img

# Upload image
image_file = st.file_uploader("Upload a CT scan slice (PNG, JPG)", type=["jpg", "jpeg", "png"])
if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = detect_objects(image)
    img0 = np.array(image)
    img_with_boxes = draw_bounding_boxes(img0, results)

    st.image(img_with_boxes, caption="Detected Nodules (Faster R-CNN + FPR)", use_column_width=True)

    # Show classification results below the image
    if results:
        st.markdown("### Classification Results:")
        cancer_possibility = 0  # Variable to track possible cancer risk
        for i, ((x1, y1, x2, y2), conf, cls) in enumerate(results, 1):
            label = fpr_decision(conf)
            color_label = ""
            if "Non-Nodule" in label:
                color_label = f"<span style='color:red'>{label}</span>"
            elif "Indeterminate" in label:
                color_label = f"<span style='color:orange'>{label}</span>"
            else:
                color_label = f"<span style='color:green'>{label}</span>"
                if conf > 0.6:  # Threshold to consider a high cancer possibility
                    cancer_possibility += 1
            st.markdown(f"**Detection {i}:** {color_label} (Confidence: {conf:.2f})", unsafe_allow_html=True)
        
        # Print cancer possibility report
        if cancer_possibility > 1:
            st.markdown("### Possible Cancer Risk Detected:")
            st.warning("There is a possibility of cancer based on the detected nodules. Further examination is recommended.")
        else:
            st.success("No significant cancer risk detected based on the analysis.")
    else:
        st.info("No nodules detected in the image.")
