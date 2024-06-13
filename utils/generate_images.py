# utils/generate_images.py
"""
    @description    : use the saved gan model to generate more data to advance the classifier
"""
import torch
from torchvision.utils import save_image
from model.gan.generator import build_generator
from config.config import Config
import os

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_images(generator_path, num_images, save_dir):
    generator = build_generator(config.latent_dim).to(device)
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(num_images):
            z = torch.randn(1, config.latent_dim, device=device)
            fake_img = generator(z)
            save_image(fake_img, os.path.join(save_dir, f"generated_{i}.png"), normalize=True)


# Example usage
generate_images('saved_models/generator_epoch_20.pth', 1000, 'datasets/flowers/generated')