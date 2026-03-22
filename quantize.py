import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic

class SimpleAudioNet(nn.Module):
    def __init__(self):
        super(SimpleAudioNet, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load model
model = SimpleAudioNet()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# ⚡ Quantize: converts float32 weights → int8 (4x smaller!)
quantized_model = quantize_dynamic(
    model,
    {nn.Linear},      # which layers to quantize
    dtype=torch.qint8 # target data type
)

# Save quantized model
torch.save(quantized_model.state_dict(), "model_quantized.pth")

# Compare sizes
import os
original = os.path.getsize("model.pth") / 1024
quantized = os.path.getsize("model_quantized.pth") / 1024
print(f"Original model:   {original:.1f} KB")
print(f"Quantized model:  {quantized:.1f} KB")
print(f"Size reduction:   {((original-quantized)/original*100):.1f}%")