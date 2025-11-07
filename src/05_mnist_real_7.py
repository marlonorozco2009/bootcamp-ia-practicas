from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(r"C:\bootcamp-ia")
DATA = ROOT / "data" / "mnist"
OUT  = ROOT / "out"; OUT.mkdir(parents=True, exist_ok=True)

# --- Transforms ---
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# --- Cargar TEST y buscar el primer '7' ---
test_ds = datasets.MNIST(DATA, train=False, download=True, transform=tfm)
idx7 = (test_ds.targets == 7).nonzero(as_tuple=True)[0][0].item()
x, y = test_ds[idx7]        # x: [1,28,28], y: 7
x = x.unsqueeze(0)          # -> [1,1,28,28]

print(f"Imagen ejemplo (7) -> shape: {tuple(x.shape)} | etiqueta: {int(y)}")

# --- Guardar imagen normalizada y des-normalizada ---
try:
    import matplotlib.pyplot as plt
    x_denorm = (x * 0.3081 + 0.1307).clamp(0,1)
    plt.figure(figsize=(2,2)); plt.imshow(x[0,0].detach().numpy(), cmap="gray")
    plt.axis("off"); plt.title("Entrada (normalizada)")
    plt.tight_layout(); plt.savefig(OUT/"img7_norm.png"); plt.close()

    plt.figure(figsize=(2,2)); plt.imshow(x_denorm[0,0].detach().numpy(), cmap="gray")
    plt.axis("off"); plt.title("Entrada (0..1)")
    plt.tight_layout(); plt.savefig(OUT/"img7_denorm.png"); plt.close()

    print("Guardé img7_norm.png e img7_denorm.png en", OUT)
except Exception as e:
    print("No pude guardar imágenes (¿falta matplotlib?). Error:", e)

# --- CNN mínima ---
class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc   = nn.Linear(8*14*14, 10)
    def forward(self, x, verbose=False):
        h1 = self.conv(x)          # [N,8,28,28]
        h2 = F.relu(h1)
        h3 = self.pool(h2)         # [N,8,14,14]
        h4 = torch.flatten(h3, 1)  # [N,1568]
        logits = self.fc(h4)       # [N,10]
        if verbose:
            print("\n=== Paso a paso ===")
            print("Conv :", tuple(h1.shape), "  (8 mapas de 28x28)")
            print("ReLU :", tuple(h2.shape))
            print("Pool :", tuple(h3.shape), "  (reduce a 14x14)")
            print("Flat :", tuple(h4.shape), "  (8*14*14 = 1568)")
            print("Linear-> logits:", tuple(logits.shape), " (10 clases)")
        return h1, h3, logits

model = MiniCNN()

# --- Forward explicativo (sin entrenar) ---
h1, h3, logits = model(x, verbose=True)

# --- Guardar mapas de características (conv y pool) ---
try:
    import matplotlib.pyplot as plt
    fmap  = h1.detach().numpy()[0]  # [8,28,28]
    fmap2 = h3.detach().numpy()[0]  # [8,14,14]

    # conv maps
    plt.figure(figsize=(6,3))
    for i in range(8):
        plt.subplot(2,4,i+1); plt.imshow(fmap[i], cmap="gray")
        plt.axis("off"); plt.title(f"conv {i}")
    plt.tight_layout(); plt.savefig(OUT/"img7_maps_conv.png"); plt.close()

    # pool maps
    plt.figure(figsize=(6,3))
    for i in range(8):
        plt.subplot(2,4,i+1); plt.imshow(fmap2[i], cmap="gray")
        plt.axis("off"); plt.title(f"pool {i}")
    plt.tight_layout(); plt.savefig(OUT/"img7_maps_pool.png"); plt.close()

    print("Guardé img7_maps_conv.png e img7_maps_pool.png en", OUT)
except Exception as e:
    print("No pude guardar mapas (¿falta matplotlib?). Error:", e)

# --- Predicción con softmax (ANTES de entrenar) ---
probs_before = F.softmax(logits, dim=1)[0]
pred_before  = int(torch.argmax(probs_before, dim=0))
conf_before  = float(probs_before[pred_before])
print(f"\nPredicción ANTES de entrenar: pred={pred_before}  conf={conf_before:.3f}  (gold={int(y)})")

# --- Mini-entrenamiento (200 pasos) ---
from torch.utils.data import DataLoader
train_ds = datasets.MNIST(DATA, train=True, download=True, transform=tfm)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

crit = nn.CrossEntropyLoss()
opt  = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\n=== Mini-entrenamiento (200 pasos) ===")
steps, loss_hist = 0, []
model.train()
for xb, yb in train_loader:
    _, _, lg = model(xb)
    loss = crit(lg, yb)
    opt.zero_grad(); loss.backward(); opt.step()
    steps += 1; loss_hist.append(loss.item())
    if steps % 50 == 0:
        print(f"Paso {steps}/200 - loss: {sum(loss_hist[-50:])/50:.4f}")
    if steps >= 200:
        break

# --- Predicción DESPUÉS ---
model.eval()
with torch.no_grad():
    _, _, logits2 = model(x)
probs_after = F.softmax(logits2, dim=1)[0]
pred_after  = int(torch.argmax(probs_after, dim=0))
conf_after  = float(probs_after[pred_after])
print(f"\nPredicción DESPUÉS de entrenar: pred={pred_after}  conf={conf_after:.3f}  (gold={int(y)})")
