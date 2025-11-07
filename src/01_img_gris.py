import torch

# N=1 (un ejemplo), C=1 (grises), H=W=28
x = torch.randn(1, 1, 28, 28)   # tensor simulado
print("shape:", x.shape)        # torch.Size([1, 1, 28, 28])

# Accesos útiles:
img = x[0]          # -> [1, 28, 28]  (la única imagen del batch)
canal = img[0]      # -> [28, 28]     (único canal gris)
px = canal[14, 14]  # un píxel
print("un píxel:", float(px))
