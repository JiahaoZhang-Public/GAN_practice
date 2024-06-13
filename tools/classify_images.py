# tools/classify_images
import torch
import torchvision.transforms as transforms
from PIL import Image
from model.classifier.classifier import build_classifier
from config.config import Config

config = Config()

def classify_image(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    model = build_classifier(config.num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()

    return class_idx

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify an image using a trained classifier.")
    parser.add_argument("image_path", type=str, help="Path to the image to classify")
    parser.add_argument("model_path", type=str, help="Path to the saved classifier model")

    args = parser.parse_args()
    class_idx = classify_image(args.image_path, args.model_path)
    print(f"Predicted class index: {class_idx}")