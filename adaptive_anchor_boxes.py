import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
import os

def load_annotations(csv_path):
    """Load world-coordinate annotations (x, y, z, diameter)."""
    df = pd.read_csv(csv_path)
    if not {'diameter_mm'}.issubset(df.columns):
        raise ValueError("annotations.csv must have a 'diameter_mm' column.")
    return df[['diameter_mm']].values

def generate_anchor_boxes(annotation_array, bandwidth=5.0):
    """Use MeanShift to find representative anchor diameters."""
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(annotation_array)
    anchors = np.sort(ms.cluster_centers_.flatten())
    return anchors

def save_anchors(anchors, output_path="anchor_boxes.npy"):
    np.save(output_path, anchors)
    print(f"[INFO] Anchor boxes saved to: {output_path}")
    print(f"[INFO] Anchors: {anchors}")

if __name__ == "__main__":
    annotations_path = "D:\projects\Lung_Nodule_Detection\\annotations.csv"  
    if not os.path.exists(annotations_path):
        raise FileNotFoundError("annotations.csv not found!")

    annotations = load_annotations(annotations_path)
    anchors = generate_anchor_boxes(annotations)
    save_anchors(anchors)
