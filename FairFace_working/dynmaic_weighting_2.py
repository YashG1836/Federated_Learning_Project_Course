# =========================================================
# FairFace Federated Learning + Dynamic Fair Clients
# =========================================================

import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

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
FAIR_RATIO = 0.4   # 40% fair clients

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

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

# ================= MODEL =================
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
def get_weights(model):
    return [p.detach().cpu().numpy() for p in model.parameters()]

def set_weights(model, weights):
    for p, w in zip(model.parameters(), weights):
        p.data = torch.tensor(w).to(DEVICE)

def fedavg(weights):
    return [np.mean(w, axis=0) for w in zip(*weights)]

# ================= FAIR TRAIN =================
def train(model, loader, fair=False):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for x, y, r in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        out = model(x)

        # ---------------- NORMAL LOSS ----------------
        loss = nn.CrossEntropyLoss()(out, y)

        # ---------------- FAIRNESS WEIGHTING ----------------
        if fair:
            preds = out.argmax(1)

            race_acc = {}
            for race in torch.unique(r):
                idx = (r == race)
                if idx.sum() > 0:
                    acc = (preds[idx].cpu() == y[idx].cpu()).float().mean().item()
                    race_acc[int(race)] = acc

            if len(race_acc) > 0:
                max_acc = max(race_acc.values())

                weights = torch.ones_like(y).float()

                for i in range(len(y)):
                    race_i = int(r[i])
                    weights[i] = max_acc - race_acc.get(race_i, max_acc) + 1.0

                loss = (nn.CrossEntropyLoss(reduction='none')(out, y) * weights.to(DEVICE)).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

# ================= EVAL =================
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

def race_accuracy(preds, labels, races):
    results = {}
    for r in sorted(np.unique(races)):
        idx = races == r
        results[r] = (preds[idx] == labels[idx]).mean()

    gap = max(results.values()) - min(results.values())
    return results, gap

# ================= LOAD DATA =================
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]

train_f, test_f = train_test_split(files, test_size=0.2, random_state=SEED)
client_data = np.array_split(train_f, NUM_CLIENTS)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_loader = DataLoader(
    FairFaceDataset(DATA_DIR, test_f, transform),
    batch_size=128, shuffle=False
)

# ================= FL TRAIN =================
global_model = GenderCNN().to(DEVICE)

fair_clients = set(random.sample(range(NUM_CLIENTS), int(NUM_CLIENTS * FAIR_RATIO)))
print("Fair clients:", fair_clients)

for round_num in range(ROUNDS):

    local_weights = []

    for cid in range(NUM_CLIENTS):

        loader = DataLoader(
            FairFaceDataset(DATA_DIR, client_data[cid], transform),
            batch_size=BATCH_SIZE, shuffle=True
        )

        local_model = GenderCNN().to(DEVICE)
        set_weights(local_model, get_weights(global_model))

        if cid in fair_clients:
            train(local_model, loader, fair=True)
        else:
            train(local_model, loader, fair=False)

        local_weights.append(get_weights(local_model))

    set_weights(global_model, fedavg(local_weights))

    print(f"Round {round_num+1}/{ROUNDS} completed")

# ================= FINAL =================
preds, labels, races = evaluate(global_model, test_loader)

race_acc, gap = race_accuracy(preds, labels, races)
overall_acc = (preds == labels).mean()

print("\n🔥 FINAL RESULTS")
print(f"Overall Accuracy: {overall_acc:.4f}")

print("\nRace-wise Accuracy:")
for r in race_acc:
    print(f"Race {r}: {race_acc[r]:.4f}")

print(f"\n⚖️ Fairness Gap: {gap:.4f}")