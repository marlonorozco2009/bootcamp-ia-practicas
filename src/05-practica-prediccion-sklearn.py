import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- 1. Dataset (El "CSV" de Datos) ---
# En un proyecto real, cargaríamos un CSV con pandas.read_csv()
# Aquí, creamos un pequeño dataset "a mano" para la práctica.
# Estos son nuestros "datos históricos"
data = {
    'altitud_msnm': [1200, 1300, 1400, 1500, 1600, 1700, 1800],
    'temp_promedio_c': [24.5, 23.0, 22.5, 21.0, 20.5, 19.0, 18.5],
    'puntaje_calidad_1_10': [6.5, 7.0, 7.5, 8.5, 8.8, 9.2, 9.5]
}

# Usamos Pandas para convertir nuestros datos en una "tabla" profesional
df = pd.DataFrame(data)

print("--- 1. Datos Históricos (Dataset) ---")
print(df)

# --- 2. Preparar los Datos para el Modelo ---
# La IA necesita saber qué son las "preguntas" (X) y qué es la "respuesta" (y)

# X (Features / Características): Las "preguntas" que le damos al modelo
# Serán las columnas de altitud y temperatura
X = df[['altitud_msnm', 'temp_promedio_c']]

# y (Target / Objetivo): La "respuesta" que queremos que aprenda a predecir
# Será la columna de puntaje de calidad
y = df['puntaje_calidad_1_10']

# --- 3. Dividir el Dataset ---
# ¡Buena práctica! No podemos entrenar y probar con los mismos datos.
# Separamos nuestros datos en dos grupos:
# - Un grupo para "estudiar" (train)
# - Un grupo para "hacer el examen" (test)
# test_size=0.3 significa que usamos el 30% de los datos para el examen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\n--- 2. Datos separados para Entrenar y Probar ---")
print(f"Usaremos {len(X_train)} ejemplos para entrenar.")
print(f"Usaremos {len(X_test)} ejemplos para probar.")

# --- 4. Crear y Entrenar el Modelo ---
# A diferencia de PyTorch, ¡Scikit-learn hace todo en 2 líneas!

# 1. Creamos el modelo (una Regresión Lineal)
model = LinearRegression()

# 2. Entrenamos el modelo con los datos de "estudio"
print("\n--- 3. Entrenando el modelo... ---")
model.fit(X_train, y_train)
print("✅ ¡Modelo entrenado!")

# --- 5. Ver qué Aprendió el Modelo ---
# Podemos espiar los "pesos" que aprendió el modelo
# (similar a los pesos de la neurona)
print("\n--- 4. ¿Qué aprendió el modelo? ---")
print(f"Importancia de 'altitud': {model.coef_[0]:.2f}")
print(f"Importancia de 'temp_promedio': {model.coef_[1]:.2f}")
print(f"Puntaje base (Intercepto): {model.intercept_:.2f}")

# --- 6. Probar el Modelo (El "Examen Final") ---
print("\n--- 5. Probando el modelo con datos de prueba... ---")
y_pred = model.predict(X_test)

# Comparamos las predicciones con las respuestas reales
print("   Respuesta Real (y_test) vs. Predicción del Modelo (y_pred)")
for real, predicho in zip(y_test, y_pred):
    print(f"   - Real: {real}  | Predicho: {predicho:.2f}")

# Calcular el error
error = mean_squared_error(y_test, y_pred)
print(f"Error (MSE) del modelo: {error:.2f} (un error más bajo es mejor)")

# --- 7. Usar el Modelo para Predecir (¡El Hackathon!) ---
print("\n--- 6. Haciendo una predicción nueva ---")
# ¿Qué puntaje tendrá un nuevo lote de café de 1650m y 20°C?
nuevo_cafe = pd.DataFrame({
    'altitud_msnm': [1650],
    'temp_promedio_c': [20.0]
})

puntaje_predicho = model.predict(nuevo_cafe)
print(f"El modelo predice un puntaje de calidad de: {puntaje_predicho[0]:.2f}")