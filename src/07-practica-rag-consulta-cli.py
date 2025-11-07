# -*- coding: utf-8 -*-
"""
Consulta RAG en consola (LangChain + HF + FAISS)
Lee el índice en: C:\bootcamp-ia\src\faiss_index_enlaces
"""

import os
import torch

# Evitar backend TF y silenciar avisos innecesarios
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------- CONFIG ----------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"
FAISS_DIR = r"C:\bootcamp-ia\src\faiss_index_enlaces"  # 👈 ruta absoluta al índice
FAISS_INDEX_NAME = "index"
TOP_K = 2
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.95
# ---------------------------

def main():
    # Dispositivo
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    # Embeddings (alineados al dispositivo)
    print("Cargando embeddings…")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": str(device)}
    )

    # Cargar índice
    print("Cargando índice FAISS…")
    vector_store = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        index_name=FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    print("¡Índice cargado!")

    # LLM
    print("Cargando LLM…")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if use_cuda else torch.float32
    ).to(device)

    gen_pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if use_cuda else -1,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P
    )
    llm = HuggingFacePipeline(pipeline=gen_pipe)
    print("¡LLM listo!")

    # Prompt específico para RAG
    template = """Eres un asistente experto de Soporte ENLACES.
Responde SÓLO con el contexto proporcionado. Si no hay respuesta en el contexto, dilo explícitamente.

CONTEXT:
{context}

PREGUNTA: {question}

Responde en español, claro y conciso. Si aplica, enumera pasos o viñetas."""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("✅ ¡Agente RAG listo para responder!")

    try:
        while True:
            q = input("\n👤 Pregunta (o 'salir'): ").strip()
            if q.lower() in {"salir", "exit", "quit"}:
                print("🤖 ¡Hasta luego!")
                break
            if not q:
                print("⚠️ Escribe algo o 'salir'.")
                continue

            print("🤖 Pensando…")
            out = qa_chain.invoke({"query": q})

            print("\n🤖 Respuesta:")
            print((out.get("result") or "").strip())

            print("\n--- Fuentes ---")
            src_docs = out.get("source_documents") or []
            if not src_docs:
                print("- (No se devolvieron fuentes)")
            else:
                for i, d in enumerate(src_docs, 1):
                    meta = getattr(d, "metadata", {}) or {}
                    src = meta.get("source") or meta.get("file_path") or meta.get("path") or "origen_desconocido"
                    page = meta.get("page", meta.get("page_number", "s/n"))
                    print(f"- [{i}] {src} (página {page})")

    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario. ¡Hasta luego!")


if __name__ == "__main__":
    main()
