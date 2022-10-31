import torch
from torchvision import models
from models import MODELS
from config import configuration as C
import cv2
from utils.image import preprocess_image
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

args = {
    "model": models.vgg16(pretrained=True),
    "image": "images/bmw.png",
}

# model = MODELS[args["model"]].to(C.DEVICE)
model = args["model"]
model.eval()


print("[INFO] loading image...")
image = cv2.imread(args["image"])
orig = image.copy()


image = preprocess_image(image)
image = torch.from_numpy(image)
image = image.to(C.DEVICE)

print("[INFO] loading ImageNet labels...")
imagenet_labels = dict(enumerate(open(os.path.join(BASE_DIR, "config", C.IN_LABELS))))


y_hat = model(image)
probabilities = torch.nn.Softmax(dim=-1)(y_hat)
sortedProba = torch.argsort(probabilities, dim=-1, descending=True)
for (i, idx) in enumerate(sortedProba[0, :5]):
    labels = imagenet_labels[idx.item()].strip()
    proba = probabilities[0, idx.item()] * 100
    print("{}.\t{}:\t{:.2f}".format(i, labels, proba))

label = (imagenet_labels[probabilities.argmax().item()],)
prob = probabilities.max().item()
text = f"Label: {label}, {prob*100}"
cv2.putText(
    img=orig,
    text=text,
    org=(10, 30),
    fontFace=cv2.FONT_HERSHEY_COMPLEX,
    fontScale=0.8,
    color=(0, 0, 255),
    thickness=2,
)
cv2.imshow("Classification", orig)
cv2.waitKey(0)
