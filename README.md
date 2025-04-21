# Lung Nodule Detection using Faster R-CNN with Adaptive Anchors and FPR Classifier

This project simulates a lung nodule detection pipeline that uses a pretrained object detection model pretending to be a Faster R-CNN enhanced with adaptive anchor boxes and a False Positive Reduction (FPR) classifier. The application uses Streamlit to provide an interactive interface for uploading and analyzing CT scan slices.

---

##  Features

- **Faster R-CNN architecture (simulated)**
- **Adaptive anchor boxes** automatically generated from dataset annotations
- **False Positive Reduction (FPR) classifier** simulated based on confidence levels
- **Interactive web interface** built with Streamlit

---

##  File Structure

```
Lung_Nodule_Detection/
├── frcnn_app.py                      # Main Streamlit app (run this)
├── frcnn_fpr_detector.pt            # Simulated model checkpoint
├── frcnn_annotations.csv           # Annotation file
├── adaptive_anchor_boxes.py        # Script to generate anchor boxes
├── frcnn_adaptive_anchors.npy      # Numpy file with adaptive anchor boxes
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

##  Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Lung_Nodule_Detection.git
cd Lung_Nodule_Detection
```

### 2. Set Up a Python Environment (Recommended)
```bash
conda create -n lung-nodule python=3.8 -y
conda activate lung-nodule
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```
> **Note:** Make sure you have PyTorch installed with CUDA if you're using GPU. You can install it from https://pytorch.org/get-started/locally/

---

##  Running the App

### Step 1: Generate Anchor Boxes (Optional)
If you haven’t already generated the adaptive anchor boxes:
```bash
python adaptive_anchor_boxes.py
```
This creates the `frcnn_adaptive_anchors.npy` file.

### Step 2: Launch the App
```bash
streamlit run app.py
```

### Step 3: Upload an Image
Use the interface to upload a PNG/JPG image of a CT scan slice. The app will:
- Run the image through the simulated Faster R-CNN model
- Apply adaptive anchor boxes (internally loaded)
- Simulate FPR classification based on detection confidence
- Display the image with bounding boxes and labels

---

##  Simulated Components

- **Faster R-CNN:** The detection model is a YOLOv5 model internally but is referred to as Faster R-CNN for demonstration purposes.
- **Adaptive Anchors:** Derived from real annotation statistics using k-means clustering.
- **FPR Classifier:** Simulated by mapping detection confidence levels to malignancy classes.

---

##  Requirements

- Python >= 3.7
- PyTorch >= 1.8
- OpenCV
- Streamlit
- PIL
- Numpy
- Requests

Install all of these via:
```bash
pip install -r requirements.txt
```
