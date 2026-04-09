import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict

# ================= CONFIG =================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = "FairFace_50k"
NUM_CLIENTS = 10
ROUNDS = 10
BATCH_SIZE = 32
IMG_SIZE = 64
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= DATASET =================
class FairFaceDataset(Dataset):
    def __init__(self, root, files, transform=None):
        self.root = root
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.root, name)).convert("RGB")

        parts = name.split("_")
        gender = int(parts[1])
        race = int(parts[2])

        if self.transform:
            img = self.transform(img)

        return img, gender, race

# ================= MODEL (SAME AS UTK) =================
class GenderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*16*16, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

# ================= HELPERS =================
def train(model, loader):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

def evaluate(model, loader):
    model.eval()
    preds, labels, races = [], [], []

    with torch.no_grad():
        for x, y, r in loader:
            x = x.to(DEVICE)
            p = model(x).argmax(1).cpu().numpy()
            preds.extend(p)
            labels.extend(y.numpy())
            races.extend(r.numpy())

    return np.array(preds), np.array(labels), np.array(races)

def fedavg(weights):
    return [np.mean(w, axis=0) for w in zip(*weights)]

def get_weights(model):
    return [p.detach().cpu().numpy() for p in model.parameters()]

def set_weights(model, weights):
    for p, w in zip(model.parameters(), weights):
        p.data = torch.tensor(w).to(DEVICE)

def race_accuracy(preds, labels, races):
    results = {}

    for r in sorted(np.unique(races)):
        idx = races == r
        acc = (preds[idx] == labels[idx]).mean()
        results[r] = acc

    gap = max(results.values()) - min(results.values())

    return results, gap

# ================= LOAD DATA =================
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]

print("Total images:", len(files))
print("Device:",DEVICE)
