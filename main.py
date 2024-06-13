# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from config.config import Config
from model.classifier.classifier import build_classifier
import numpy as np

config = Config()

transform = transforms.Compose([
    transforms.Resize(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def sample_dataset(dataset, max_images_per_class):
    class_indices = {cls: [] for cls in range(config.num_classes)}
    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < max_images_per_class:
            class_indices[label].append(idx)

    sampled_indices = [idx for indices in class_indices.values() for idx in indices]
    return Subset(dataset, sampled_indices)


train_dataset = datasets.ImageFolder(root=config.train_dataset, transform=transform)
train_generated_dataset = datasets.ImageFolder(root=config.train_dataset+'/generated', transform=transform)

train_dataset = sample_dataset(train_dataset, config.max_images_per_class)
train_generated_dataset = sample_dataset(train_generated_dataset, config.max_images_per_class)

combined_dataset = torch.utils.data.ConcatDataset([train_dataset, train_generated_dataset])
train_loader = DataLoader(combined_dataset, batch_size=config.batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_classifier(config.num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

os.makedirs('saved_models', exist_ok=True)

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
    print(f"Epoch [{epoch + 1}/{config.num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    if epoch % 5 == 0:
        torch.save(model.state_dict(), f"saved_models/classifier_epoch_{epoch}.pth")

# Save the final model
torch.save(model.state_dict(), "saved_models/classifier_final.pth")