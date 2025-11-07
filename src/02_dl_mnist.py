# C:\bootcamp-ia\src\02a_cnn_didactico.py
# CNN didáctica: entender capas, formas y mapas de características con MNIST (1 imagen).
# Requisitos: torch, torchvision (y opcional matplotlib para guardar imágenes).

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np

# ---------- Config ----------
ROOT = Path(r"C:\bootcamp-ia")
OUT  = ROOT / "out"
DATA = ROOT / "data" / "mnist"
OUT.mkdir(parents=True, exist_ok=True)

MOSTRAR_PASO_A_PASO = True     # imprime formas después de cada capa
GUARDAR_IMAGENES    = True     # intenta guardar png (requiere matplotlib)
ENTRENAR_UN_POQUITO = True     # mini-entrenamiento de 200 pasos para ver loss bajar

# ---------- Dataset (1 muestra) ----------
tfm = transforms.Compose([
    transforms.ToTensor(),                      # [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # normaliza: más estable
])

test_ds = datasets.MNIST(DATA, train=False, download=True, transform=tfm)
# Tomamos UNA imagen del test para explicar paso a paso
x0, y0 = test_ds[0]            # x0: tensor [1,28,28] | y0: etiqueta (0..9)
x0 = x0.unsqueeze(0)           # -> [batch=1, channel=1, H=28, W=28]

print(f"Imagen ejemplo: batch={x0.shape[0]}, canales={x0.shape[1]}, alto={x0.shape[2]}, ancho={x0.shape[3]}")
print(f"Etiqueta (número escrito): {y0}")

# ---------- Modelo: CNN mínima ----------
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 1 -> 8 mapas (más pequeño que el script “grande”)
    nn.ReLU(),
    nn.MaxPool2d(2),                             # reduce a la mitad: 28x28 -> 14x14
    nn.Flatten(),                                # aplana todo
    nn.Linear(8 * 14 * 14, 10)                   # 10 clases (dígitos 0..9)
)

# ---------- Forward paso a paso (visualizar formas) ----------
def forward_explicado(x):
    def shape(t): return tuple(t.shape)
    print("\n=== Paso a paso ===")
    h1 = model[0](x);   print("Conv2d ->", shape(h1), "  (8 mapas de 28x28)")
    h2 = model[1](h1);  print("ReLU   ->", shape(h2), "  (mismas dims; cambia valores)")
    h3 = model[2](h2);  print("MaxPool->", shape(h3), "  (reduce resolución: 14x14)")
    h4 = model[3](h3);  print("Flatten->", shape(h4), "  (vector 8*14*14)")
    logits = model[4](h4); print("Linear ->", shape(logits), "(10 logits)")
    return h1, h3, logits  # devolvemos mapas útiles

h1, h3, logits = forward_explicado(x0)

# ---------- Guardar imágenes (opcional) ----------
if GUARDAR_IMAGENES:
    try:
        import matplotlib.pyplot as plt
        # Imagen original (des-normalizada para verla bien)
        img = x0[0,0].detach().cpu().numpy()*0.3081 + 0.1307
        plt.figure(figsize=(2,2)); plt.imshow(img, cmap="gray"); plt.axis("off")
        plt.title(f"Entrada (label={int(y0)})")
        plt.tight_layout(); plt.savefig(OUT/"cnn_demo_entrada.png"); plt.close()

        # Primeros 8 mapas de la Conv (h1) para esta imagen
        fmap = h1.detach().cpu().numpy()[0]  # [8, 28, 28]
        cols = 4; rows = 2
        import math
        plt.figure(figsize=(6,3))
        for i in range(8):
            plt.subplot(rows, cols, i+1)
            plt.imshow(fmap[i], cmap="gray")
            plt.axis("off"); plt.title(f"mapa {i}")
        plt.tight_layout(); plt.savefig(OUT/"cnn_demo_maps_conv.png"); plt.close()

        # Mapas después del MaxPool (h3)
        fmap2 = h3.detach().cpu().numpy()[0]  # [8, 14, 14]
        plt.figure(figsize=(6,3))
        for i in range(8):
            plt.subplot(rows, cols, i+1)
            plt.imshow(fmap2[i], cmap="gray")
            plt.axis("off"); plt.title(f"pool {i}")
        plt.tight_layout(); plt.savefig(OUT/"cnn_demo_maps_pool.png"); plt.close()

        print("Imágenes guardadas en:", OUT)
    except Exception as e:
        print("No pude guardar imágenes (¿falta matplotlib?). Error:", e)

# ---------- Micro-entrenamiento (opcional; 200 pasos) ----------
if ENTRENAR_UN_POQUITO:
    from torch.utils.data import DataLoader
    train_ds = datasets.MNIST(DATA, train=True, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3)

    pasos, loss_hist = 0, []
    print("\n=== Mini-entrenamiento: 200 pasos ===")
    model.train()
    for x, y in train_loader:
        logits = model(x)
        loss = crit(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        pasos += 1; loss_hist.append(loss.item())
        if pasos % 50 == 0:
            print(f"Paso {pasos}/200 - loss: {sum(loss_hist[-50:])/50:.4f}")
        if pasos >= 200:
            break

    # Ver predicción en la imagen ejemplo ANTES y DESPUÉS
    model.eval()
    with torch.no_grad():
        _, _, logits2 = forward_explicado(x0)
        pred_antes = int(torch.argmax(logits))
        pred_desp  = int(torch.argmax(logits2))
    print(f"\nPredicción ejemplo (aprox.): antes={pred_antes}  después={pred_desp}")
    # Nota: “antes” se toma del último batch; sirve solo como orientación.
