import onnxruntime as ort
import numpy as np

# Load ONNX model (this is what runs on mobile/NPU)
session = ort.InferenceSession("model.onnx")

# Get input name
input_name = session.get_inputs()[0].name
print(f"Input name: {input_name}")

# Create fake audio input (1 sample, 128 features)
fake_audio = np.random.randn(1, 128).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: fake_audio})

print(f"✅ Output shape: {outputs[0].shape}")
print(f"✅ Predicted class: {np.argmax(outputs[0])}")
print("🎉 Model running via ONNX Runtime (like on a mobile device!)")