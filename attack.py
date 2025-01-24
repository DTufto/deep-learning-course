import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import cv2
import kagglehub
import json
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.structures import ImageList
from detectron2.checkpoint import DetectionCheckpointer


class ImageDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Load labels
        with open(labels_file) as f:
            self.labels = {item['name']: item['attributes'] for item in json.load(f)}
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.labels.get(img_name, {})
        # Convert label dict to tensor or vector based on your needs
        # Example: weather label
        weather_label = 1 if label.get('weather') == 'rainy' else 0

        if self.transform:
            image = self.transform(image)
        return image, weather_label


def setup_model(model_path):
    # Setup Detectron2 configuration
    cfg = get_cfg()
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build model
    model = build_model(cfg)

    # Load weights manually
    checkpoint = torch.load(model_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Handle backbone prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k] = v
        else:
            new_state_dict[f'backbone.{k}'] = v

    model.backbone.load_state_dict(new_state_dict, strict=False)

    return model, cfg


def process_batch(model, images, device):
    # Convert to format expected by Detectron2
    images = [x.to(device) for x in images]
    images = ImageList.from_tensors(images)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)

    return outputs


# Attack functions
def blur_img(image, value=10):
    return cv2.blur(image, (value, value))


def brightness_img(image, value=100):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def no_demosaic_img(image):
    w, h, _ = image.shape
    result = np.zeros((w, h, 3))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result[::2, ::2, 2] = gray_image[::2, ::2]  # blue
    result[1::2, ::2, 1] = gray_image[1::2, ::2]  # green
    result[::2, 1::2, 1] = gray_image[::2, 1::2]  # green
    result[1::2, 1::2, 0] = gray_image[1::2, 1::2]  # red
    return result


def main():
    # Define paths
    BASE_PATH = kagglehub.dataset_download("solesensei/solesensei_bdd100k")
    VAL_PATH = os.path.join(BASE_PATH, "bdd100k/bdd100k/images/100k/val")
    LABELS_PATH = os.path.join(BASE_PATH, "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json")
    MODEL_PATH = "models/resnet_50.pth"

    # Load validation dataset
    val_dataset = ImageDataset(
        VAL_PATH,
        LABELS_PATH,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = setup_model(MODEL_PATH)
    model = model.to(device)
    model.eval()

    # Perform inference
    results = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in val_loader:
            outputs = process_batch(model, images, device)
            # Process outputs according to your needs
            # This will depend on your specific task (classification, detection, etc.)
            predictions = outputs  # Modify this based on your output format
            results.extend(predictions)
            labels.extend(batch_labels.numpy())

    # Calculate metrics
    # Modify this according to your task and output format
    accuracy = np.mean(np.array(results) == np.array(labels))
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Processed {len(results)} images")


if __name__ == "__main__":
    main()