from transformers import pipeline
from pathlib import Path
import csv

# --- 1. CONFIGURACIÓN ---
# Usamos un modelo entrenado en español para entender mejor los matices
MODEL_NAME = "pysentimiento/robertuito-sentiment-analysis"
print(f"Cargando modelo '{MODEL_NAME}'... Esto puede tardar un momento la primera vez.")

# Lista de frases que vamos a analizar
frases = [
    "Me encanta este taller de inteligencia artificial",
    "La clase está muy aburrida y no entiendo",
    "La explicación del profesor fue clara y concisa",
    "No entendí nada de lo que dijo",
    "El proyecto final parece muy interesante",
    "El ejercicio de la neurona fue algo confuso"
]

# Mapa simple para "traducir" las etiquetas del modelo a algo legible
MAPA_LABELS = {
    "POS": "POSITIVO",
    "NEG": "NEGATIVO",
    "NEU": "NEUTRAL"
}

# --- 2. CARGAR LA IA ---
# Esta es la línea "mágica". Creamos un analizador de sentimientos.
# Le decimos que use el backend "pt" (PyTorch)
analizador = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    framework="pt"
)
print("¡Modelo cargado con éxito!")

# --- 3. ANALIZAR LAS FRASES (Una por una) ---
print("\n--- Analizando Frases ---")
resultados_para_csv = []

# Usamos un bucle 'for' simple en lugar de una lista de compresión
# para que los estudiantes puedan ver cada paso.
for frase_actual in frases:
    
    # 1. Le pasamos la frase a la IA
    prediccion_raw = analizador(frase_actual)
    
    # 2. El resultado es una lista, tomamos el primer elemento
    prediccion = prediccion_raw[0] # Ej: {'label': 'POS', 'score': 0.99}
    
    # 3. Obtenemos los datos que nos interesan
    label_original = prediccion['label']
    confianza = prediccion['score']
    
    # 4. "Traducimos" la etiqueta a español usando nuestro mapa
    label_amigable = MAPA_LABELS.get(label_original, "OTRO")
    
    # 5. Imprimimos el resultado en la consola (¡feedback inmediato!)
    print(f"Frase: '{frase_actual}'")
    print(f"  -> Resultado: {label_amigable} (Confianza: {confianza:.2f})")
    
    # 6. Guardamos los datos limpios para el CSV
    resultados_para_csv.append([frase_actual, label_amigable, confianza])

# --- 4. GUARDAR EN CSV ---
# Creamos la carpeta de salida si no existe
ruta_salida = Path(r"C:\bootcamp-ia\out") # Puedes cambiar esta ruta
ruta_salida.mkdir(parents=True, exist_ok=True)
archivo_csv = ruta_salida / "results_sentiment.csv"

# Escribimos el archivo
with open(archivo_csv, "w", newline="", encoding="utf-8") as f:
    escritor_csv = csv.writer(f)
    # Escribir el encabezado
    escritor_csv.writerow(["texto", "sentimiento_detectado", "confianza"])
    # Escribir todos los resultados
    escritor_csv.writerows(resultados_para_csv)

print(f"\n¡Listo! 🚀 Resultados guardados en: {archivo_csv}")