# C:\bootcamp-ia\src\01a_mnist_intro.py
# Objetivo: ver cómo es MNIST (imágenes, etiquetas, formas) sin CNN.
from pathlib import Path
from collections import Counter
import numpy as np

import torch
from torchvision import datasets, transforms

ROOT = Path(r"C:\bootcamp-ia")
DATA = ROOT / "data" / "mnist"
OUT  = ROOT / "out"; OUT.mkdir(parents=True, exist_ok=True)

# 1) Cargar MNIST con el preprocesado mínimo
tfm = transforms.Compose([
    transforms.ToTensor(),                      # pasa a tensor [C,H,W] con valores 0..1
    transforms.Normalize((0.1307,), (0.3081,))  # normaliza (media, desvío) típicos de MNIST
])

train_ds = datasets.MNIST(DATA, train=True,  download=True, transform=tfm)
test_ds  = datasets.MNIST(DATA, train=False, download=True, transform=tfm)

print(f"Train: {len(train_ds)} imágenes  | Test: {len(test_ds)} imágenes")

# 2) Mirar UNA muestra
x0, y0 = train_ds[0]          # x0: tensor [1,28,28], y0: entero 0..9
print("Una muestra -> shape:", tuple(x0.shape), "| etiqueta:", y0)

# 3) Ver distribución de etiquetas (¿hay clases balanceadas?)
y_train = [int(train_ds[i][1]) for i in range(10000)]  # bastan 10k para ver tendencia
cont = Counter(y_train)
print("Conteo parcial de etiquetas (0..9):", cont)

# 4) Guardar una rejilla 10x10 (10 ejemplos por dígito) para observar
try:
    import matplotlib.pyplot as plt
    # recolectar 10 ejemplos por dígito
    buckets = {d: [] for d in range(10)}
    for img, y in train_ds:
        y = int(y)
        if len(buckets[y]) < 10:
            buckets[y].append(img[0].numpy()*0.3081 + 0.1307)  # des-normaliza
        if all(len(buckets[d]) == 10 for d in range(10)):
            break

    fig, axs = plt.subplots(10, 10, figsize=(8,8))
    for r in range(10):
        for c in range(10):
            axs[r,c].imshow(buckets[r][c], cmap="gray")
            axs[r,c].axis("off")
            if c == 0: axs[r,c].set_title(str(r))
    plt.tight_layout()
    path = OUT / "mnist_grid_10x10.png"
    plt.savefig(path); plt.close()
    print("Guardé una rejilla 10x10 en:", path)
except Exception as e:
    print("No pude guardar imagen (¿falta matplotlib?). Error:", e)

# 5) Comprobar que y es un entero y cómo se usa con CrossEntropy
import torch.nn as nn
dummy_logits = torch.randn(1, 10)  # 10 clases
dummy_y = torch.tensor([y0])       # etiqueta de la muestra (entero)
loss = nn.CrossEntropyLoss()(dummy_logits, dummy_y)
print("Ejemplo de CrossEntropy con y entero:", float(loss))
