
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import MB_VIT
import matplotlib.pyplot as plt
import  numpy as np
import random
# Define data paths and transformations
seed = 19
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
train_data_path = "data/Splicedimage/train"  # Replace with your training data set path
val_data_path = "data/Splicedimage/test"  # Replace with your validation dataset path

torch.cuda.is_available()
transform = transforms.Compose([
    transforms.Resize((236, 236)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the training data set and validation data set
train_dataset = ImageFolder(train_data_path, transform=transform)
val_dataset = ImageFolder(val_data_path, transform=transform)

# Create a data loader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


#  Initialize the model
num_classes = len(train_dataset.classes)
model = MB_VIT.MB_VIT(num_classes)

# Define loss functions and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training and verification
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):

    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_accuracy = 100.0 * correct / total
    train_loss /= len(train_loader)

    # 验证
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_accuracy = 100.0 * correct / total
    val_loss /= len(val_loader)

    # Print training and validation results for each epoch
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}% - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%")


from sklearn.manifold import TSNE

# 从训练好的模型中提取特征
model.eval()
features = []
labels = []
with torch.no_grad():
    for images, lbls in val_loader:
        images = images.to(device)
        lbls = lbls.to(device)
        outputs = model(images)
        features.append(outputs.cpu().numpy())  # 假设outputs是一个张量
        labels.append(lbls.cpu().numpy())

features = np.concatenate(features)
labels = np.concatenate(labels)


