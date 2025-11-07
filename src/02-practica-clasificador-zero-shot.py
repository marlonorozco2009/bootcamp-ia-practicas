from transformers import pipeline

# --- 1. CONFIGURACIÓN ---
# Usamos un modelo estándar de Hugging Face para esta tarea.
# MNLI = Multi-Genre Natural Language Inference (Inferencia de Lenguaje Natural Multi-Género)
# Esto significa que el modelo es experto en determinar si una frase "contradice", "neutral" o "implica" a otra.
# Usamos esta habilidad para ver qué etiqueta "implica" más a nuestra frase.
MODEL_NAME = "facebook/bart-large-mnli"
print(f"Cargando modelo '{MODEL_NAME}'... Esto puede tardar un momento la primera vez.")

# Lista de frases de ejemplo (ej. tickets de soporte técnico)
# Estas son las frases que queremos clasificar
frases_a_clasificar = [
    "Hola, olvidé la contraseña de mi correo @clases.edu.sv",
    "La laptop que me dieron no enciende, la pantalla se queda en negro.",
    "¿Cuál es la dirección de la sede de soporte técnico en San Miguel?",
    "No me funciona el mouse que me entregaron.",
    "Soy docente y necesito cambiar mi clave de acceso al sistema."
]

# Estas son las categorías que NOSOTROS INVENTAMOS.
# La IA nunca fue entrenada para conocerlas, pero las "entenderá".
categorias_candidatas = [
    "Gestión de Cuentas", 
    "Soporte Técnico de Equipo", 
    "Información de Sedes",
    "Problema de Software"
]

# --- 2. CARGAR LA IA ---
# Creamos el pipeline de "clasificación zero-shot"
print("Cargando el pipeline de clasificación zero-shot...")
clasificador = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME
)
print("¡Clasificador listo!")

# --- 3. CLASIFICAR LAS FRASES ---
print("\n--- Analizando Frases (Clasificación Zero-Shot) ---")

# Iteramos sobre cada frase para verla paso a paso
for frase in frases_a_clasificar:
    
    # Esta es la línea "mágica":
    # Le pasamos la frase Y las categorías que inventamos
    resultado = clasificador(
        frase,
        candidate_labels=categorias_candidatas
    )
    
    # El resultado es un diccionario que contiene las etiquetas (labels)
    # y las puntuaciones (scores), ordenadas de mayor a menor.
    
    etiqueta_ganadora = resultado['labels'][0]
    confianza = resultado['scores'][0]
    
    print(f"\nFrase: '{frase}'")
    print(f"  -> Categoría más probable: {etiqueta_ganadora} (Confianza: {confianza:.2f})")
    
    # Opcional: Mostrar todas las puntuaciones para que vean cómo "piensa" la IA
    # print("   (Puntuaciones detalladas):")
    # for label, score in zip(resultado['labels'], resultado['scores']):
    #     print(f"     - {label}: {score:.2f}")

print(f"\n¡Listo! 🚀 Observa cómo la IA clasificó cada frase en la categoría correcta sin entrenamiento previo.")