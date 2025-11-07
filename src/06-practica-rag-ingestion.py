# -*- coding: utf-8 -*-
"""
Crea/rehace el índice FAISS desde documentos locales (txt, pdf, docx).
Requiere:
  pip install langchain==0.1.16 langchain-community==0.0.32 langchain-huggingface==0.0.3
  pip install faiss-cpu transformers torch sentencepiece pypdf docx2txt python-docx
"""

import os
import sys
from pathlib import Path

# Evitar backend TF (no lo necesitamos para HF + PyTorch)
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Rutas absolutas (ajústalas si cambias carpetas) ---
DOCS_DIR = r"C:\bootcamp-ia\src\documentos_enlaces"
INDEX_DIR = r"C:\bootcamp-ia\src\faiss_index_enlaces"

# --- Modelos ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def info(msg: str):
    print(f"[INFO] {msg}")

def load_texts():
    # TXT
    txt_loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs_txt = txt_loader.load()
    info(f"TXT cargados: {len(docs_txt)}")
    return docs_txt

def load_pdfs():
    # PDF (carga por archivo con PyPDFLoader, mejor control)
    docs_pdf = []
    for p in Path(DOCS_DIR).rglob("*.pdf"):
        try:
            parts = PyPDFLoader(str(p)).load()
            docs_pdf.extend(parts)
        except Exception as e:
            print(f"[WARN] No pude cargar PDF {p}: {e}")
    info(f"PDF cargados (por página): {len(docs_pdf)}")
    return docs_pdf

def load_docx():
    # DOCX
    docs_docx = []
    for p in Path(DOCS_DIR).rglob("*.docx"):
        try:
            parts = Docx2txtLoader(str(p)).load()
            docs_docx.extend(parts)
        except Exception as e:
            print(f"[WARN] No pude cargar DOCX {p}: {e}")
    info(f"DOCX cargados: {len(docs_docx)}")
    return docs_docx

def main():
    # Chequeo de carpetas
    if not Path(DOCS_DIR).exists():
        print(f"[ERROR] No existe la carpeta de documentos: {DOCS_DIR}")
        sys.exit(1)

    # Borrar índice previo para evitar conflictos de versiones
    idx_path = Path(INDEX_DIR)
    if idx_path.exists():
        info(f"Eliminando índice previo en {INDEX_DIR}…")
        for f in idx_path.glob("*"):
            try: f.unlink()
            except: pass
    else:
        idx_path.mkdir(parents=True, exist_ok=True)

    # Cargar documentos multi-formato
    info(f"Cargando documentos desde {DOCS_DIR}…")
    docs = []
    docs += load_texts()
    docs += load_pdfs()
    docs += load_docx()

    total_docs = len(docs)
    if total_docs == 0:
        print("[ERROR] No se cargó NINGÚN documento. Revisa rutas y extensiones.")
        sys.exit(1)

    info(f"Total de documentos/unidades cargadas: {total_docs}")

    # Splitter (ajuste pensado para T5)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    info(f"Chunks generados: {len(chunks)}")

    # Embeddings
    info("Cargando modelo de embeddings…")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}  # en Windows CPU suele ir bien
    )

    # FAISS
    info("Creando índice FAISS… esto puede tardar.")
    vs = FAISS.from_documents(chunks, embeddings)

    # Guardar
    vs.save_local(INDEX_DIR)
    info(f"Índice guardado en: {INDEX_DIR}")

    # Tamano de archivos
    fa = Path(INDEX_DIR) / "index.faiss"
    pk = Path(INDEX_DIR) / "index.pkl"
    info(f"index.faiss: {fa.stat().st_size if fa.exists() else 0} bytes")
    info(f"index.pkl  : {pk.stat().st_size if pk.exists() else 0} bytes")

    print("\n✅ ¡Éxito! Índice creado y guardado.\n"
          "Ahora ejecuta el script de consulta 07…")

if __name__ == "__main__":
    main()
