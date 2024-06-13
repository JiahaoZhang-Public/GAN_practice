# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config.config import Config
from model.classifier.classifier import build_classifier

config = Config()

transform = transforms.Compose([
    transforms.Resize(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.ImageFolder(root='datasets/flowers', transform=transform)
train_generated_dataset = datasets.ImageFolder(root='datasets/flowers/generated', transform=transform)
combined_dataset = torch.utils.data.ConcatDataset([train_dataset, train_generated_dataset])
train_loader = DataLoader(combined_dataset, batch_size=config.batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_classifier(config.num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

for epoch in range(config.num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {running_loss/len(train_loader):.4f}")