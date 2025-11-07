from transformers import pipeline
from pathlib import Path
import csv

MODEL_NAME = "pysentimiento/robertuito-sentiment-analysis"  # o el que uses
UMBRAL = 0.80

frases = [
    "Me encanta este taller",
    "La clase está aburrida",
    "La explicación fue clara",
    "No entendí nada",
    "El proyecto es interesante",
    "El ejercicio fue confuso"
]

# Forzar backend PyTorch (sin TF)
clf = pipeline("sentiment-analysis", model=MODEL_NAME, framework="pt")

def normalizar_label(pred):
    lab = str(pred["label"]).strip().upper()
    sc  = float(pred["score"])
    if sc < UMBRAL: return ("NEUTRAL", sc)
    if lab and lab[0].isdigit():
        n = int(lab.split()[0])
        return ("NEGATIVE", sc) if n<=2 else ("NEUTRAL", sc) if n==3 else ("POSITIVE", sc)
    mapa = {"POS":"POSITIVE","NEG":"NEGATIVE","NEU":"NEUTRAL",
            "POSITIVE":"POSITIVE","NEGATIVE":"NEGATIVE","NEUTRAL":"NEUTRAL"}
    return (mapa.get(lab, "NEUTRAL"), sc)

preds = [normalizar_label(clf(t)[0]) for t in frases]

out = Path(r"C:\bootcamp-ia\out"); out.mkdir(parents=True, exist_ok=True)
with open(out/"results_sentiment.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["texto","label","score"])
    for t,(lab,sc) in zip(frases,preds):
        w.writerow([t, lab, round(sc,4)])
print("Listo.")
