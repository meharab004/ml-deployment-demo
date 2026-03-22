import torch
import torch.nn as nn

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

model = SimpleAudioNet()
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

dummy_input = torch.randn(1, 128)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=9,
    input_names=["input"],
    output_names=["output"]
)

print("✅ model.onnx created successfully!")