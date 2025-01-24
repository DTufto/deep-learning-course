import cv2
import kagglehub
import numpy as np
from torch import tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


# from attack import blur_img, brightness_img, nodemosaic_img


def change_brightness(image: np.ndarray, value: int = 1.9):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] *= value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def avg_brightness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray.mean()


def normalise_brightness(image: np.ndarray, goal: float = 0.5) -> np.ndarray:
    brightness = avg_brightness(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] *= (goal / brightness)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def blur_img(image: np.ndarray, value: int = 10) -> np.ndarray:
    # return cv2.blur(image,(value,value)) # most effective in the paper was 12,12, but that's a heavy blur
    return cv2.GaussianBlur(image, (5, 5), 0)  # gaussian blur could also be used (needs uneven size for kernel)


def check_blur(image: np.ndarray) -> float:
    # Check if image is blurry by using global 2nd derivative
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def sharpen_gaussian(iamge: np.ndarray):
    # TODO: compare to laplacian or frequency domain sharpening
    # TODO: Try different ratio between blur and original (or get from literature)
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.addWeighted(image, 1.5, blur, -0.5, 0)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406],
    #                      [0.229, 0.224, 0.225])
])

path = kagglehub.dataset_download("solesensei/solesensei_bdd100k")
path_extended = f'{path}\\bdd100k\\bdd100k\\images\\100k\\test'
test_dataset = ImageFolder(root=path_extended, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for images, labels in test_loader:
    for image in images:
        image = image.numpy()
        image = np.transpose(image, (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # cv2.imshow("original", image)

        original = avg_brightness(image)
        image = change_brightness(image)
        # cv2.imshow("changed", image)
        changed = avg_brightness(image)

        image = normalise_brightness(image)
        restored = avg_brightness(image)
        # cv2.imshow("restored", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print(f"Original: {original:.4f}, Changed: {changed:.4f}, restored: {restored:.4f}")
    break
