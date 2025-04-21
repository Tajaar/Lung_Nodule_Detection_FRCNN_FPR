import torch
from yolov5.models.experimental import attempt_load
from torchsummary import summary
import pathlib
import sys

if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath
# Load model
model_path = "best.pt"  # Make sure this path is correct
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(model_path, device=device)
model.eval()

# Print model details
print("\n🔍 MODEL ARCHITECTURE:\n")
print(model)

# Summary: Uncomment below if input size is known (e.g., 3x640x640 for YOLOv5)
try:
    dummy_input = torch.zeros((1, 3, 640, 640)).to(device)
    print("\n📋 MODEL SUMMARY:\n")
    summary(model, input_size=(3, 640, 640))
except Exception as e:
    print(f"❌ Summary failed: {e}")

# List class names
print("\n🏷️ CLASS LABELS:\n")
try:
    print(model.names)
except Exception:
    print("Class names not found in model.")

# Parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 PARAMETER COUNT:\nTotal: {total_params:,}, Trainable: {trainable_params:,}")
