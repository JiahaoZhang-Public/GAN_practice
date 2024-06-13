# gan_main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from config.config import Config
from model.gan.generator import build_generator
from model.gan.discriminator import build_discriminator
import os

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = build_generator(config.latent_dim).to(device)
discriminator = build_discriminator().to(device)

adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))

transform = transforms.Compose([
    transforms.Resize(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.ImageFolder(root=config.train_dataset, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

os.makedirs('saved_models', exist_ok=True)

for epoch in range(config.num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        # Train Discriminator
        optimizer_D.zero_grad()
        real_imgs = imgs.to(device)
        b_size = real_imgs.size(0)
        real_labels = torch.full((b_size,), 1, dtype=torch.float, device=device)
        fake_labels = torch.full((b_size,), 0, dtype=torch.float, device=device)

        real_validity = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_validity, real_labels)

        z = torch.randn(b_size, config.latent_dim, device=device)
        fake_imgs = generator(z)
        fake_validity = discriminator(fake_imgs.detach())
        d_fake_loss = adversarial_loss(fake_validity, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_validity = discriminator(fake_imgs)
        g_loss = adversarial_loss(fake_validity, real_labels)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{config.num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    if epoch % 5 == 0:
        save_image(fake_imgs.data[:25], f"results/epoch_{epoch}.png", nrow=5, normalize=True)
        torch.save(generator.state_dict(), f"saved_models/generator_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"saved_models/discriminator_epoch_{epoch}.pth")