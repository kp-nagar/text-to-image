import os, random
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
dataset = CIFAR10(root="data", download=True, transform=transform)
classes = dataset.classes

os.makedirs("data/images", exist_ok=True)
captions = []
for i in range(20):
    img, label = random.choice(dataset)
    filename = f"img_{i}.png"
    transforms.ToPILImage()(img).save(f"data/images/{filename}")
    captions.append(f"{filename}\ta photo of a {classes[label]}")

with open("data/captions.txt", "w") as f:
    f.write("\n".join(captions))
