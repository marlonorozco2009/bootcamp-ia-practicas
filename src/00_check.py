import sys
print("Python:", sys.version)

import transformers, torch
print("Transformers:", transformers.__version__)
print("PyTorch:", torch.__version__)

# Mini test Transformers
from transformers import pipeline
clf = pipeline("sentiment-analysis")
print(clf("I love this bootcamp!")[0])  # Debe imprimir POSITIVE con score
