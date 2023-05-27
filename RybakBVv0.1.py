import random
import time
import pydirectinput
import pygetwindow
import pyautogui
import os
import keyboard
from pathlib import Path
from PIL import Image
from datetime import datetime
import torchvision
from torchvision import datasets, transforms
import torch
from torch import nn

class MNISTModelV0(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 7 * 7,
                      out_features=num_classes
                      )
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

# # Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = MNISTModelV0()
#
# # Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f="D:\projekty\Py\RybakBV\Model\liczby.pth"))

def screenshot():
    path = Path('D:/bot/result.png')
    titles = pygetwindow.getAllTitles()

    window = pygetwindow.getWindowsWithTitle('Valium.pl')[0]
    left, top = window.topleft
    right, bottom = window.bottomright
    pyautogui.screenshot(path)

    # curr_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # splitted_path = os.path.splitext(path)
    # modified_picture_path = splitted_path[0] + curr_datetime + splitted_path[1]

    im = Image.open(path)
    im = im.crop((left + 653, top + 390, right - 583, bottom - 310))
    im.save(path)
    #im.show(modified_picture_path)

    return path


path = "D:/bot/result.png"


def jakaliczba():
    custom_image = torchvision.io.read_image(str(path)).type(torch.float32)
    custom_image = custom_image / 255.

    # Create transform pipleine to resize image
    custom_image_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(1)
    ])
    custom_image_transformed = custom_image_transform(custom_image)

    loaded_model_0.eval()
    with torch.inference_mode():
      target_image = custom_image_transformed.unsqueeze(dim=0)
      target_image_pred = loaded_model_0(target_image)
      target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
      target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    if target_image_pred_probs.max().item() > 0.7:
        x = target_image_pred_label.item()
        return x
    else:
        return 11

# print(jakaliczba())

def zarzut():
    time.sleep(0.1)
    pydirectinput.press("1")
    time.sleep(random.uniform(0.1, 0.3))
    pydirectinput.press("2")

while(1):
    screenshot()
    time.sleep(0.05)
    liczba = jakaliczba()
    print(liczba)
    if liczba!=11:
        for i in range(int(liczba)):
            time.sleep(0.1)
            pydirectinput.press("space")
        time.sleep(8)
        zarzut()
    time.sleep(0.05)
    if keyboard.is_pressed("j"):
        break


