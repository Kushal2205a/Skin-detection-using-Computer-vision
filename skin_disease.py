from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch, torch.nn as nn, torch.optim as optim
from torchvision import models
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
import os, json

# Prepare paths
root_file = Path("/home/kushal/Documents/SKindetection/skin-disease-datasaet/")
root_file_count = root_file/"test_set"
for cls in sorted(root_file_count.iterdir()):
    print(cls.name, sum(1 for _ in cls.glob('*')))
    plt.figure(figsize=(12,4))
    plt.suptitle(cls.name, fontsize=16)

img_size = 224
imgnet_mean = (0.485, 0.456, 0.406)
imgnet_std = (0.229, 0.224, 0.225)

train_tf = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(imgnet_mean,imgnet_std)
])
val_tf = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize(imgnet_mean,imgnet_std)
])

train_ds = datasets.ImageFolder(root_file/"train_set", transform=train_tf)
val_size = int(0.2 * len(train_ds))
train_size = len(train_ds) - val_size
train_ds, val_ds = random_split(train_ds, [train_size, val_size])
val_ds.dataset.transform = val_tf
test_ds = datasets.ImageFolder(root_file/"test_set", transform=val_tf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = torch.cuda.is_available()
num_workers = 4 if torch.cuda.is_available() else 0

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
class_to_idx = train_ds.dataset.class_to_idx
num_classes = len(class_to_idx)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
checkpoint_path = "/home/kushal/Documents/SKindetection/best_model.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
    torch.save(model.state_dict(), checkpoint_path)

pre_brightness = transforms.Compose([transforms.Resize((img_size, img_size))])
post_brightness = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(imgnet_mean,imgnet_std)
])

class TestWithBrightness(torch.utils.data.Dataset):
    def __init__(self, folder_root, brightness_factor):
        self.files = list((folder_root).glob("*/*"))
        self.brightness = brightness_factor
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        img = pre_brightness(img)
        img = TF.adjust_brightness(img, self.brightness)
        img = post_brightness(img)
        # Defensive mapping - handle missing mapping
        try:
            label = class_to_idx[p.parent.name]
        except KeyError:
            label = -1
        return img, label

def eval_loader_per_class(dl, model=model, device=device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in dl:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    per_class_acc = {}
    for cls in range(num_classes):
        idxs = (y_true == cls)
        if idxs.sum() > 0:
            per_class_acc[str(cls)] = float((y_pred[idxs] == y_true[idxs]).mean())
        else:
            per_class_acc[str(cls)] = None
    return y_true, y_pred, per_class_acc

recompute = False

results_file = "brightness_results.json"
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)
    # Defensive check for non-empty results
    if not results or "per_class_acc" not in results:
        recompute = True
else:
    recompute = True

factors = [0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]

if recompute:
    results = []
    for f in factors:
        ds_f = TestWithBrightness(root_file/"test_set", f)
        dl_f = DataLoader(ds_f, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)
        y_true, y_pred, per_class_acc = eval_loader_per_class(dl_f)
        acc = (y_true == y_pred).mean()
        results.append({"brightness": f, "accuracy": float(acc), "per_class_acc": per_class_acc})
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

# Plot accuracy vs brightness
xs = [r["brightness"] for r in results]
ys = [r["accuracy"] for r in results]
plt.figure(figsize=(6,4))
plt.plot(xs, ys, marker="o")
plt.xlabel("Brightness factor (1.0 = original)")
plt.ylabel("Accuracy")
plt.title("Robustness to brightness variations")
plt.grid(True)
plt.savefig("brightness_robustness.png", dpi=300)

idx_to_class = {v:k for k,v in class_to_idx.items()}
base_acc = next(r for r in results if r["brightness"]==1.0)["per_class_acc"]
sensitivity = {}
for r in results:
    if r["brightness"] == 1.0:
        continue
    drop_per_class = {
        idx_to_class[int(c)]: (base_acc[c] - r["per_class_acc"][c]) if base_acc[c] is not None and r["per_class_acc"][c] is not None else 0.0
        for c in base_acc
    }
    most_sensitive_class = max(drop_per_class, key=drop_per_class.get)
    sensitivity[r["brightness"]] = (most_sensitive_class, drop_per_class[most_sensitive_class])

for f, (cls, drop) in sensitivity.items():
    print(f"Brightness {f}: most sensitive class = {cls}, accuracy drop = {drop:.3f}")

# For colored plotting
import matplotlib
colors = [matplotlib.colors.to_rgb(c) for c in plt.cm.tab10.colors]

classes = list(class_to_idx.keys())
x = list(sensitivity.keys())
y_numeric = [classes.index(cls) for cls, drop in sensitivity.values()]

plt.figure(figsize=(10,4))
plt.plot(x, y_numeric, marker="o")
plt.xticks(x)
plt.yticks(range(len(classes)), classes)
plt.xlabel("Brightness factor (1.0 = original)")
plt.ylabel("Most sensitive class")
plt.title("Sensitive class")
plt.grid(True)
plt.savefig("sensitive_class.png", dpi=300)

# Acc drop per class
all_drop_per_class = {}
base_acc = next(r for r in results if r["brightness"]==1.0)["per_class_acc"]

for r in results:
    if r["brightness"] == 1.0:
        continue
    drop_per_class = {
        idx_to_class[int(c)]: (base_acc[c] - r["per_class_acc"][c]) if base_acc[c] is not None and r["per_class_acc"][c] is not None else 0.0
        for c in base_acc
    }
    all_drop_per_class[r["brightness"]] = drop_per_class

factors_for_plot = sorted(all_drop_per_class.keys())
def rgb2hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*(int(x*255) for x in rgb))


plt.figure(figsize=(12,6))
for i, cls in enumerate(classes):
    drops = [all_drop_per_class[f][cls] for f in factors_for_plot]
    c = rgb2hex(colors[i % len(colors)])  # cycle colors if >10 classes
    plt.stem(factors_for_plot, drops, linefmt=c, markerfmt=c, basefmt='k-', label=cls)
plt.xlabel("Brightness factor (1.0 = original)")
plt.ylabel("Accuracy drop")
plt.title("Accuracy drop (sensitivity) per class")
plt.xticks(factors_for_plot)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("sensitivity_all_classes.png", dpi=300)
plt.show()

