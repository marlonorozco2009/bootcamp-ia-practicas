import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models # Importamos 'models'
# Necesitamos DataLoader y random_split
from torch.utils.data import DataLoader, random_split 
import os
import time # Para medir tiempos (opcional)

# --- 1. Configuración ---
# Asegúrate de que esta sea la ruta a la carpeta que contiene 'benign/' y 'malignant/'
# Ejemplo: si descomprimiste y tienes 'Dataset_BUSI_with_GT/benign', entonces data_dir es 'Dataset_BUSI_with_GT/'
data_dir = 'Dataset_BUSI_with_GT/' # <<< ¡¡¡AJUSTA ESTA RUTA!!! 
batch_size = 16 # Puedes ajustar esto según la memoria de tu GPU/CPU
learning_rate = 0.001
num_epochs = 5 # El Transfer Learning es rápido

# Verificar si hay GPU disponible, si no, usar CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# --- 2. Dataset y DataLoader (Modificado para estructura simple) ---
# Transformaciones: Redimensionar, cortar, normalizar y aumentar datos
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(), # Aumento de datos
        transforms.RandomRotation(10),     # Aumento de datos
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([ # Validación (sin aumento aleatorio)
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Cargar el dataset completo desde la carpeta principal
print(f"Cargando imágenes desde '{data_dir}'...")
try:
    # Aplicamos las transformaciones de 'val' inicialmente para cargar
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['val']) 
except FileNotFoundError:
    print(f"Error: No se encontró la carpeta '{data_dir}'. Asegúrate de que la ruta sea correcta.")
    exit()

# Obtener nombres de clases ('benign', 'malignant', ¿'normal'?)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"✅ Clases encontradas: {num_classes} ({', '.join(class_names)})")

# Dividir el dataset en entrenamiento (80%) y validación (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset_split, val_dataset_split = random_split(full_dataset, [train_size, val_size])

# ¡IMPORTANTE! Aplicar las transformaciones de 'train' (con aumento) SOLO al subset de entrenamiento
# Necesitamos crear "nuevos" datasets a partir de los subsets para asignarles las transformaciones correctas
# Esto es un poco avanzado, pero necesario para el aumento de datos correcto
class DatasetWithTransforms:
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        # Convertir x a tensor si no lo es ya (ImageFolder ya lo hace, pero por seguridad)
        if not isinstance(x, torch.Tensor):
             x = transforms.ToTensor()(x) # Asegurar que es Tensor    
        return x, y

    def __len__(self):
        return len(self.subset)

train_dataset = DatasetWithTransforms(train_dataset_split, transform=data_transforms['train'])
val_dataset = DatasetWithTransforms(val_dataset_split, transform=data_transforms['val'])


# Crear los DataLoaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

print(f"   Datos divididos: Entrenamiento={dataset_sizes['train']}, Validación={dataset_sizes['val']}")

# --- 3. Modelo (Transfer Learning con MobileNetV2) ---
print("🧠 Cargando modelo pre-entrenado (MobileNetV2)...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Congelar los pesos del modelo base
for param in model.parameters():
    param.requires_grad = False

# Reemplazar la última capa (el clasificador)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)

# Mover el modelo a la GPU si está disponible
model = model.to(device)

print("✅ Modelo listo para Transfer Learning.")

# --- 4. Pérdida y Optimizador ---
criterion = nn.CrossEntropyLoss()
# Optimizar SOLO los parámetros del nuevo clasificador
optimizer = optim.Adam(model.classifier[1].parameters(), lr=learning_rate)

# --- 5. Loop de Entrenamiento ---
print(f"\n🧠 Empezando el entrenamiento por {num_epochs} epochs...")
start_time = time.time()

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    # Cada época tiene una fase de entrenamiento y una de validación
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Poner el modelo en modo entrenamiento
        else:
            model.eval()   # Poner el modelo en modo evaluación

        running_loss = 0.0
        running_corrects = 0

        # Iterar sobre los datos.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Limpiar gradientes
            optimizer.zero_grad()

            # Forward pass
            # Rastrear historial solo en entrenamiento
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1) # Obtener la clase predicha
                loss = criterion(outputs, labels)

                # Backward pass + optimizar solo si es entrenamiento
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Estadísticas
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print()

time_elapsed = time.time() - start_time
print(f'Entrenamiento completado en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

# --- 6. Guardar el Modelo Entrenado ---
# Guardamos el modelo afinado para usarlo en el hackathon
model_save_path = 'modelo_cancer_mama_mobilenet.pth'
# Guardamos solo los 'pesos' aprendidos (state_dict)
torch.save(model.state_dict(), model_save_path) 
print(f"\n✅ Modelo entrenado guardado en: {model_save_path}")
print("   Puedes usar este archivo para cargar el modelo y hacer predicciones.")