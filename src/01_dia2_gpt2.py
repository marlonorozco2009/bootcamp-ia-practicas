# Importamos las librerías necesarias
from transformers import pipeline
import torch

# NO NECESITAMOS notebook_login() AQUÍ PORQUE YA INICIAMOS SESIÓN EN LA TERMINAL

# --- 1. CARGA DEL MODELO ---
model_name = "google/gemma-2b"
print(f"Cargando el modelo '{model_name}'...")

try:
    generador = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print("¡Modelo cargado con éxito!")

except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# --- 2. CREACIÓN DE LA PLANTILLA DE PROMPT ---
rol = "Eres un guionista de ciencia ficción experto."
contexto = "Escribe el inicio de una escena para una película donde un robot descubre que puede sentir emociones por primera vez."
restricciones = "La escena debe ser corta, no más de 80 palabras, y terminar con una pregunta del robot a sí mismo."

prompt_completo = f"""
<start_of_turn>user
{rol}

{contexto}

{restricciones}
<end_of_turn>
<start_of_turn>model
ESCENA:
"""

# --- 3. GENERACIÓN DE LA RESPUESTA ---
print("\n--- Prompt que se enviará a la IA ---")
print(prompt_completo)
print("--------------------------------------")
print("\n🤖 Generando respuesta...")

resultado = generador(
    prompt_completo,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
)

# --- 4. VISUALIZACIÓN DEL RESULTADO ---
print("\n--- Respuesta de la IA ---")
print(resultado[0]['generated_text'])