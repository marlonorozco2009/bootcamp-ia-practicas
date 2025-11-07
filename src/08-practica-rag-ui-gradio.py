import gradio as gr
# Corrected LangChain imports based on the previous solution
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline # Correct import for LLM wrapper
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- 1. CARGAR TODO EL CEREBRO (Igual que el script 07) ---
print("Iniciando el Agente de IA, por favor espera...")

# Modelos
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

# Cargar Archivista (Índice FAISS)
print("Cargando índice...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'}
)
vector_store = FAISS.load_local(
    "faiss_index_enlaces",
    embeddings,
    allow_dangerous_deserialization=True # Necessary for loading the .pkl file
)

# Cargar Experto (LLM)
print("Cargando LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

# Crear Cadena RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

print("✅ ¡Agente listo! Iniciando interfaz web...")

# --- 2. FUNCIÓN DE RESPUESTA PARA GRADIO ---
# Gradio needs a function that takes the input and returns the output

def responder_pregunta(pregunta, historial_chat):
    """
    Takes the user's question and chat history,
    returns the response from the RAG agent.
    """
    print(f"Pregunta recibida: {pregunta}")

    # Invoke the chain (we don't use history in this simple RAG)
    resultado = qa_chain.invoke({"query": pregunta})

    respuesta = resultado["result"]

    # Extract sources to display them
    fuentes = set() # Use a 'set' to avoid duplicate sources
    for doc in resultado["source_documents"]:
        fuentes.add(doc.metadata['source'])

    # Append sources to the answer
    respuesta_con_fuentes = f"{respuesta}\n\n--- Fuentes ---\n" + "\n".join(list(fuentes))

    # Append the new interaction to the history (Gradio handles this format)
    historial_chat.append((pregunta, respuesta_con_fuentes))
    
    # Return None for the input textbox (to clear it) and the updated history
    return "", historial_chat


# --- 3. CREAR LA INTERFAZ WEB CON GRADIO ---
# This is the magic! We create a chat interface.
# 

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Agente de Soporte ENLACES")
    gr.Markdown("Haz una pregunta sobre reparación de equipos o cuentas @clases.edu.sv.")

    # The Chatbot component that displays the conversation
    chatbot = gr.Chatbot(label="Chat", height=500)

    # The Textbox component for typing the message
    msg = gr.Textbox(label="Tu Pregunta", placeholder="Escribe tu consulta aquí...")

    # The Clear button
    clear = gr.Button("🧹 Limpiar Chat")

    # Define the action when the message is submitted (Enter key)
    msg.submit(responder_pregunta, [msg, chatbot], [msg, chatbot])

    # Define the action of the Clear button
    clear.click(lambda: None, None, chatbot, queue=False)

# --- 4. LANZAR LA APLICACIÓN ---
print("Lanzando la aplicación web. Busca la URL local (http://127.0.0.1:XXXX) en la consola.")
# share=True creates a temporary public link (useful if running in Colab or sharing)
demo.launch(share=False)