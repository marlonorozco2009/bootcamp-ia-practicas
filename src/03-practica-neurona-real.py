import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Dataset (El "libro de texto" de la IA) ---
# Datos de entrada (X): Horas de estudio por día
X_train = torch.tensor([
    [1.0], [1.5], [2.0], [3.0], [3.5],  # Reprobados
    [4.5], [5.0], [5.5], [6.0], [7.0]   # Aprobados
], dtype=torch.float32)

# Datos de salida (y): 0 = Reprobado, 1 = Aprobado
y_train = torch.tensor([
    [0.0], [0.0], [0.0], [0.0], [0.0],
    [1.0], [1.0], [1.0], [1.0], [1.0]
], dtype=torch.float32)

print("Datos de entrenamiento listos. Queremos encontrar el 'umbral' (alrededor de 4.0 horas).")

# --- 2. Modelo (La Neurona que Aprende) ---
# Definimos nuestra neurona. Usamos las "Capas" (Layers) de PyTorch.
class NeuronaSimple(nn.Module):
    def __init__(self):
        super(NeuronaSimple, self).__init__()
        # Definimos una "Capa Lineal". Es nuestro (w*x + b).
        # 1 entrada (horas de estudio) -> 1 salida (la probabilidad de aprobar)
        self.linear = nn.Linear(1, 1) 
    
    def forward(self, x):
        # Aplicamos la "Función de Activación" Sigmoid.
        # Esta función aplasta el resultado entre 0 y 1 (perfecto para probabilidades).
        # (ReLU es otra activación, pero Sigmoid es la correcta para este problema).
        return torch.sigmoid(self.linear(x))

# Creamos una instancia de nuestra neurona
model = NeuronaSimple()
print(f"Neurona creada. Parámetros iniciales (aleatorios): {list(model.parameters())}")

# --- 3. Pérdida y Optimizador (El "Crítico" y el "Motor de Ajuste") ---

# Función de Pérdida (Loss): Mide qué tan equivocada está la neurona.
# Usamos "Binary Cross-Entropy" (BCE), ideal para clasificación binaria (0 o 1).
criterion = nn.BCELoss()

# Optimizador (Adam): El algoritmo que ajusta los pesos (w) y el sesgo (b)
# para minimizar la pérdida. Es el "motor" del aprendizaje.
optimizer = optim.Adam(model.parameters(), lr=0.01) # lr = Tasa de aprendizaje

# --- 4. Loop de Entrenamiento (El proceso de "Estudiar") ---
# Epoch: Una vuelta completa a todo el dataset.
num_epochs = 2000 # Haremos 2000 "pasadas" de estudio

print(f" Empezando el entrenamiento por {num_epochs} epochs...")

for epoch in range(num_epochs):
    # 1. Forward Pass: La neurona hace su predicción
    y_pred = model(X_train)
    
    # 2. Calcular la Pérdida: El "crítico" mide el error
    loss = criterion(y_pred, y_train)
    
    # 3. Backward Pass: Calcular cómo (gradientes)
    optimizer.zero_grad() # Limpiar cálculos anteriores
    loss.backward()       # Calcular la dirección del ajuste
    
    # 4. Actualizar Pesos: El "optimizador" aplica el ajuste
    optimizer.step()
    
    # Imprimir el progreso
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Pérdida (Loss): {loss.item():.4f}')

print(" ¡Entrenamiento completado!")

# --- 5. Resultados (El Conocimiento Aprendido) ---
# Extraemos los valores que la neurona "aprendió" por sí sola.
# ¡Estos son el equivalente al `peso_estudio` y al `umbral` que pusimos a mano en el Día 1!
W = model.linear.weight.item()
b = model.linear.bias.item()

print("\n--- Resultados del Aprendizaje ---")
print(f"Peso (w) aprendido: {W:.4f}")
print(f"Sesgo (b) aprendido: {b:.4f}")

# El umbral de decisión (donde la probabilidad es 0.5) se calcula así:
umbral_aprendido = -b / W
print(f"==> El modelo aprendió que el umbral de aprobación es ~{umbral_aprendido:.2f} horas.")

# --- 6. Probar la Neurona ---
print("\n--- Probando la neurona entrenada ---")
horas_test_1 = torch.tensor([[3.0]], dtype=torch.float32) # Debería reprobar
horas_test_2 = torch.tensor([[5.0]], dtype=torch.float32) # Debería aprobar

pred_1 = model(horas_test_1).item()
pred_2 = model(horas_test_2).item()

print(f"Predicción para 3.0 horas: {pred_1:.2f} (Valor cercano a 0 = Reprobado)")
print(f"Predicción para 5.0 horas: {pred_2:.2f} (Valor cercano a 1 = Aprobado)")