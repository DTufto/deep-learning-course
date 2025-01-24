import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import kagglehub
import cv2
import numpy as np
import json
from numpy import dtype

#Load all the images from kaggle
path = kagglehub.dataset_download("solesensei/solesensei_bdd100k")
path_extended = f'{path}/bdd100k/bdd100k/images/100k/test'
# Load BDDK100 model
model = models.resnet50()
state_dict = torch.load("models/resnet_50.pth")['state_dict']
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Data loader for test set
test_set = f'{path_extended}/testA'
test_dataset = ImageFolder(root=test_set, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Do inference on test set to validate results
results = []
with torch.no_grad():
    for images, _ in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        results.extend(predicted.cpu().numpy())

# Statistical analysis of results
test_labels = [label for _, label in test_dataset.samples]
accuracy = np.mean(np.array(results) == np.array(test_labels))
print(f"Test Accuracy: {accuracy:.4f}")


# Apply attack on images

def blur_img(image, value=10):
    return cv2.blur(image,(value,value)) # most effective in the paper was 12,12, but that's a heavy blur
    # return cv2.GaussianBlur(image, (value,value)) # gaussian blur could also be used

def brightness_img(image, value=100):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.add(hsv[:,:,2], value)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image

def nodemosaic_img(image):
    w, h, _ = image.shape
    result = np.zeros((w, h, 3))        # make a target array, which will be colored with the blue, green, and red values as if it has not been demosaiced
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result[::2, ::2, 2] = gray_image[::2, ::2]       # blue
    result[1::2, ::2, 1] = gray_image[1::2, ::2]     # green
    result[::2, 1::2, 1] = gray_image[::2, 1::2]     # green
    result[1::2, 1::2, 0] = gray_image[1::2, 1::2]   # red
    return result


# Do inference on test set to see results after attacks

