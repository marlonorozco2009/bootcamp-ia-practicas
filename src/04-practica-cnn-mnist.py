import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 1. Configuración y Parámetros ---
# Definimos "hiperparámetros"
# batch_size: Cuántas imágenes "estudia" la IA en cada paso
batch_size = 64
# learning_rate: Qué tan grandes son los ajustes del optimizador
learning_rate = 0.001
# epochs: Cuántas veces "verá" el dataset completo
num_epochs = 3 # 3 es suficiente para una buena demo

# --- 2. Dataset y DataLoader ---
# MNIST es un dataset de números escritos a mano (imágenes de 28x28 píxeles)

# Transformaciones: 
# 1. Convertir la imagen (que es un formato PIL) a un Tensor de PyTorch.
# 2. (ToTensor también escala los píxeles de [0, 255] a [0.0, 1.0])
transform = transforms.ToTensor()

# Descargar el "libro de texto" (Dataset) de MNIST
train_dataset = datasets.MNIST(
    root='./data',  # Dónde guardar los datos
    train=True,     # Queremos la parte de ENTRENAMIENTO
    transform=transform, 
    download=True   # Descárgalo si no lo tenemos
)
test_dataset = datasets.MNIST(
    root='./data', 
    train=False,    # Queremos la parte de PRUEBA
    transform=transform,
    download=True
)

# DataLoader: El "ayudante" que carga los datos en "batches" (lotes)
# y los "baraja" (shuffle) para que la IA no memorice el orden.
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print("✅ Datos de MNIST cargados.")

# --- 3. Modelo (La Red Neuronal Convolucional - CNN) ---
# Esta es la arquitectura de nuestro "cerebro" para ver.

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # --- CAPAS DE "VISIÓN" (Detectives) ---
        # Conv2d: Es la "linterna" que escanea la imagen 2D buscando patrones
        # (bordes, curvas, etc.).
        # in_channels=1: 1 canal de entrada (la imagen es blanco y negro)
        # out_channels=16: 16 "linternas" (filtros) diferentes buscaremos
        # kernel_size=3: La linterna tiene un tamaño de 3x3 píxeles
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU() # Función de activación
        
        # MaxPool2d: Achica la imagen (de 28x28 a 14x14) para quedarse con lo más importante
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- CAPAS DE "DECISIÓN" (Jueces) ---
        # Flatten: El "aplanador" que convierte la imagen 2D (16x14x14) en un vector 1D
        self.flatten = nn.Flatten()
        
        # Linear: El "juez" que toma el vector 1D (de 16*14*14 = 3136 neuronas)
        # y decide cuál de las 10 clases (0-9) es la correcta.
        self.fc = nn.Linear(16 * 14 * 14, 10) 

    def forward(self, x):
        # Este es el "flujo de pensamiento"
        x = self.conv1(x)     # 1. Pasa por la linterna (Conv)
        x = self.relu(x)      # 2. Activa
        x = self.pool(x)      # 3. Achica
        x = self.flatten(x)   # 4. Aplana (de 2D a 1D)
        x = self.fc(x)        # 5. Toma la decisión final
        return x

model = SimpleCNN()
print(f"✅ Modelo CNN creado. Listo para aprender.")

# --- 4. Pérdida y Optimizador ---
# (¡Exactamente igual que el script anterior, pero con una pérdida diferente!)

# Función de Pérdida: CrossEntropyLoss es la ideal para clasificación
# de MÚLTIPLES categorías (10 números), en lugar de solo 2 (BCE).
criterion = nn.CrossEntropyLoss()

# Optimizador: Seguimos usando Adam.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 5. Loop de Entrenamiento ---
# (¡Este flujo es IDÉNTICO al script anterior!)
print(f"🧠 Empezando el entrenamiento por {num_epochs} epochs...")

for epoch in range(num_epochs):
    # Iteramos sobre cada "batch" (lote) de imágenes de entrenamiento
    for i, (images, labels) in enumerate(train_loader):
        # images es un tensor de [64, 1, 28, 28] (64 imágenes de 1 color de 28x28)
        # labels es un tensor de [64] (las 64 respuestas correctas)
        
        # 1. Forward Pass (Predecir)
        outputs = model(images)
        
        # 2. Calcular la Pérdida (Error)
        loss = criterion(outputs, labels)
        
        # 3. Backward Pass (Calcular corrección)
        optimizer.zero_grad()
        loss.backward()
        
        # 4. Actualizar Pesos (Aplicar corrección)
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("✅ ¡Entrenamiento completado!")

# --- 6. Probar el Modelo (El "Examen Final") ---
# Ponemos el modelo en modo "evaluación" (desactiva funciones de entrenamiento)
model.eval()

# No necesitamos calcular gradientes (correcciones) durante la prueba
with torch.no_grad():
    correct = 0
    total = 0
    # Iteramos sobre las imágenes de PRUEBA
    for images, labels in test_loader:
        outputs = model(images)
        # torch.max devuelve el (valor, índice) de la predicción más alta.
        # Solo nos importa el índice (la categoría, el número predicho)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0) # Sumamos el tamaño del batch (ej. 64)
        correct += (predicted == labels).sum().item() # Contamos cuántas acertó

    accuracy = 100 * correct / total
    print("\n--- Resultados de la Prueba ---")
    print(f"La IA acertó {correct} de {total} imágenes de prueba.")
    print(f'Precisión (Accuracy) del modelo en las imágenes de prueba: {accuracy:.2f} %')