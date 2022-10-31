import os

import torch
from torch import optim, nn
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import KMNIST
from torchvision import transforms
try:
    from config import configuration as C
    from ml_lib.lenet import LeNet
except:
    from CNN.config import configuration as C
    from CNN.ml_lib.lenet import LeNet

print("[INFO] loading the KMNIST dataset...")

transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = KMNIST(root="data", train=True, download=True, transform=transform)
test_data = KMNIST(root="data", train=False, download=True, transform=transform)

print("[INFO] generating the train/validation split...")
num_train_samples = int(len(train_data) * C.TRAIN_SPLIT)
num_val_samples = int(len(train_data) * C.VAL_SPLIT)

(train_data, val_data) = random_split(train_data, [num_train_samples, num_val_samples], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_data, shuffle=True, batch_size=C.BATCH_SIZE)
val_loader = DataLoader(val_data, shuffle=False, batch_size=C.BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=False, batch_size=C.BATCH_SIZE)

##############################################
# x, y = next(iter(train_data))
# num_classes = len(train_data.dataset.classes)
# model = LeNet(num_channels=1, classes=num_classes)
# y_hat = model.forward(x)
# y_hat.flatten().size() # 800
#############################################

train_steps = len(train_loader.dataset) // C.BATCH_SIZE
val_steps = len(val_loader.dataset) // C.BATCH_SIZE
test_steps = len(test_loader.dataset) // C.BATCH_SIZE



model = LeNet(num_channels=1, classes=len(train_data.dataset.classes))
loss_fn = nn.NLLLoss()
opt = optim.Adam(model.parameters(), lr=C.INIT_LR)

for epoch in range(C.EPOCHS):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    size = 0.0
    for i, (x, y) in enumerate(train_loader):
        (x, y) = (x.to(C.DEVICE), y.to(C.DEVICE))
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

        if i % C.LOG_INTERVAL == C.LOG_INTERVAL-1:
            size += C.LOG_INTERVAL
            print(f"[{epoch}/{C.EPOCHS}] train_loss: {train_loss / C.LOG_INTERVAL: .5f} train_acc: {train_acc * 100 / size / C.BATCH_SIZE:.2f}%", end="\r", flush=True)
            train_loss = 0.0

    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        model.eval()
        for (x, y) in val_loader:
            (x, y) = (x.to(C.DEVICE), y.to(C.DEVICE))
            pred = model(x)
            val_loss += loss_fn(pred, y)
            val_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

    print(f"\n[{epoch}/{C.EPOCHS}] val_loss: {val_loss / val_steps} val_acc: {val_acc / len(val_loader.dataset)*100:.2f}%")

with torch.no_grad():
    model.eval()
    preds = []
    for (x, y) in test_loader:
        x = x.to(C.DEVICE)
        pred = model(x)
        preds.extend(pred.argmax(1).cpu().numpy())

os.makedirs(C.OUTPUT_DIR, exist_ok=True)
torch.save(model, os.path.join(C.OUTPUT_DIR, "model.pth"))



