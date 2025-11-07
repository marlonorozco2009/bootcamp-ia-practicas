# Práctica 1: Simulando una neurona para decidir si un estudiante aprueba
# la función neurona_aprobacion simula a un profesor que tiene que decidir. y se basa en las horas que estudio y las horas que durmió

def neurona_aprobacion(horas_estudio, horas_sueno):
    # Damos más importancia (peso) a las horas de estudio. Los pesos representa la iportancia de la neurona de cada entrada
    peso_estudio = 0.8
    peso_sueno = 0.4
    # Umbral de aprobación (bias)
    umbral = 6.0

    # La neurona calcula la suma ponderada
    calculo = (horas_estudio * peso_estudio) + (horas_sueno * peso_sueno)

    # La neurona "dispara" su decisión
    if calculo >= umbral:
        return "¡Aprobado! "
    else:
        return "A mejorar esos hábitos. "

# ¡Juguemos con los valores!
print(f"Estudiante 1 (9h estudio, 5h sueño): {neurona_aprobacion(9, 5)}")
print(f"Estudiante 2 (4h estudio, 8h sueño): {neurona_aprobacion(4, 8)}")