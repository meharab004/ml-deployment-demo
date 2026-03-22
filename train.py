import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define a simple neural network
class SimpleAudioNet(nn.Module):
    def __init__(self):
        super(SimpleAudioNet, self).__init__()
        self.fc1 = nn.Linear(128, 64)   # input: 128 audio features
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)    # output: 10 classes

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 2. Create model
model = SimpleAudioNet()

# 3. Fake training (just to simulate — real training needs a dataset)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print("✅ Model created and ready!")
print(model)

# 4. Save trained model
torch.save(model.state_dict(), "model.pth")
print("✅ Model saved as model.pth")