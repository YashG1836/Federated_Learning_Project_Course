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
FAIR_RATIO = 0.6   # 40% fairness clients
ROUNDS = 10

BATCH_SIZE = 32
IMG_SIZE = 64
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# FairFace race map (7 classes)
RACE_MAP = {
    0: "White",
    1: "Black",
    2: "East Asian",
    3: "Indian",
    4: "Latino_Hispanic",
    5: "Middle Eastern",
    6: "Southeast Asian"
}

# ================= DATASET =================
class FairFaceDataset(Dataset):
    def __init__(self, root, files, transform):
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

        img = self.transform(img)
        return img, gender, race

# ================= MODEL =================
class GenderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*16*16,128), nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):
        return self.net(x)

# ================= FEDAVG =================
def get_weights(model):
    return [p.detach().cpu().numpy() for p in model.parameters()]

def set_weights(model, weights):
    for p, w in zip(model.parameters(), weights):
        p.data = torch.tensor(w).to(DEVICE)

def fedavg(weights):
    return [np.mean(w, axis=0) for w in zip(*weights)]

# ================= NORMAL TRAIN =================
def train(model, loader):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    for x,y,_ in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = ce(model(x), y)
        loss.backward()
        opt.step()

# ================= FAIRNESS TRAIN =================
def train_fair(model, loader):

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss(reduction='none')

    # ---- Step 1: local race accuracy ----
    race_correct = {i:0 for i in range(7)}
    race_total = {i:0 for i in range(7)}

    with torch.no_grad():
        for x,y,r in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            p = out.argmax(1)

            for i in range(len(r)):
                rr = int(r[i])
                race_total[rr] += 1
                if p[i] == y[i]:
                    race_correct[rr] += 1

    race_acc = {}
    max_acc = 0

    for k in race_total:
        if race_total[k] > 0:
            race_acc[k] = race_correct[k] / race_total[k]
            max_acc = max(max_acc, race_acc[k])
        else:
            race_acc[k] = 0

    # ---- Step 2: dynamic weights ----
    weights = {}
    for k in race_acc:
        weights[k] = (max_acc - race_acc[k])**2 + 1e-3

    total = sum(weights.values())
    for k in weights:
        weights[k] /= total

    # ---- Step 3: weighted training ----
    for x,y,r in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)

        opt.zero_grad()
        out = model(x)
        losses = ce(out,y)

        w = torch.tensor([weights[int(rr)] for rr in r], dtype=torch.float32).to(DEVICE)

        loss = (losses * w).mean()
        loss.backward()
        opt.step()

# ================= EVALUATION =================
def evaluate(model, loader):
    model.eval()
    preds, labels, races = [], [], []

    with torch.no_grad():
        for x,y,r in loader:
            x = x.to(DEVICE)
            p = model(x).argmax(1).cpu().numpy()

            preds.extend(p)
            labels.extend(y.numpy())
            races.extend(r.numpy())

    return np.array(preds), np.array(labels), np.array(races)

def race_accuracy(preds, labels, races):
    res = {}
    for r in np.unique(races):
        idx = races == r
        res[r] = (preds[idx] == labels[idx]).mean()
    return res

def gap(acc):
    return max(acc.values()) - min(acc.values())

# ================= DATA PREP =================
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]
print("Total images:", len(files))

train_f, test_f = train_test_split(files, test_size=0.2, random_state=42)
client_data = np.array_split(train_f, NUM_CLIENTS)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

test_loader = DataLoader(
    FairFaceDataset(DATA_DIR, test_f, transform),
    batch_size=128
)

# ================= FL TRAIN =================
def run_fl():

    global_model = GenderCNN().to(DEVICE)

    FAIR_CLIENTS = set(random.sample(range(NUM_CLIENTS), int(NUM_CLIENTS * FAIR_RATIO)))
    print("Fairness Clients:", FAIR_CLIENTS)

    for rnd in range(ROUNDS):

        client_weights = []

        for cid in range(NUM_CLIENTS):

            local_model = GenderCNN().to(DEVICE)
            set_weights(local_model, get_weights(global_model))

            loader = DataLoader(
                FairFaceDataset(DATA_DIR, client_data[cid], transform),
                batch_size=BATCH_SIZE,
                shuffle=True
            )

            if cid in FAIR_CLIENTS:
                train_fair(local_model, loader)
            else:
                train(local_model, loader)

            client_weights.append(get_weights(local_model))

        set_weights(global_model, fedavg(client_weights))

        print(f"Round {rnd+1} completed")

    preds, labels, races = evaluate(global_model, test_loader)
    return race_accuracy(preds, labels, races)

# ================= RUN =================
acc = run_fl()

print("\nRace-wise Accuracy:")
for r,a in acc.items():
    print(RACE_MAP[r], ":", round(a,3))

print("\nFairness Gap:", round(gap(acc),4))