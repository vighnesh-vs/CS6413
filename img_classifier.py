import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def show_image_prediction(model, dataset, index):
    model.eval()
    
    # Get the image and label
    image, label = dataset[index]
    
    # Add batch dimension and move to device
    input_img = image.unsqueeze(0).to(device)
    
    # Get model output
    with torch.no_grad():
        output = model(input_img)
        predicted = torch.argmax(output, dim=1).item()
    
    # Plot the image
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"True Label: {label}, Predicted: {predicted}")
    plt.axis('off')
    plt.show()


# ----- Square Activation -----
class SquareActivation(nn.Module):
    def forward(self, x):
        return x * x


# ----- ZK-Friendly MLP -----
class ZKMLP(nn.Module):
    def __init__(self):
        super(ZKMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.act = SquareActivation()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x   # logits only (NO softmax)


# ----- Load Dataset -----
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# ----- Initialize Model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ZKMLP().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----- Training Loop -----
epochs = 7

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ----- Evaluate -----
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")


dummy_input = torch.randn(1, 1, 28, 28).to(device)

# torch.onnx.export(
#     model,
#     dummy_input,
#     "model.onnx",
#     opset_version=11
# )

# torch.onnx.export(
#     model,
#     dummy_input,
#     "zk_mlp.onnx",
#     input_names=["input"],
#     output_names=["output"],
#     opset_version=14
# )

# print("Model exported to zk_mlp.onnx")

# Show prediction for the first 5 test images
for i in range(5):
    show_image_prediction(model, test_dataset, i)

# model.eval()

# Use a real MNIST image
# real_image, _ = test_dataset[0]
# real_input = real_image.unsqueeze(0).to(device)

# torch.onnx.export(
#     model,
#     real_input,
#     "zk_mlp.onnx",
#     input_names=["input"],
#     output_names=["output"],
#     opset_version=14,
#     do_constant_folding=True
# )