import os
import google.generativeai as genai
import markdown
from flask import Flask, render_template, request, session, redirect, url_for
from dotenv import load_dotenv
# import chromadb # No es necesario importar chromadb directamente aquí si solo usas langchain_community.vectorstores
from langchain_community.document_loaders import TextLoader # Cambiado de PyPDFLoader a TextLoader para los TXT
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import shutil # Para la función de recrear la base de datos si fuera necesario

# --- Cargar variables de entorno ---
load_dotenv()

# --- Configuración de Flask ---
app = Flask(__name__)
secret_key_env = os.getenv('FLASK_SECRET_KEY')
if not secret_key_env:
    print("ADVERTENCIA: FLASK_SECRET_KEY no encontrada en .env. Usando una clave temporal NO SEGURA.")
    app.secret_key = "clave_secreta_temporal_de_desarrollo_cambiar_en_produccion"
else:
    app.secret_key = secret_key_env

# --- Configuración de la API de Google Gemini ---
google_api_key = os.getenv('GOOGLE_API_KEY')
if not google_api_key:
    print("ERROR CRÍTICO: GOOGLE_API_KEY no encontrada en .env. La aplicación no podrá funcionar.")
    # Podrías querer salir aquí: exit()
else:
    try:
        genai.configure(api_key=google_api_key)
        print("API Key de Gemini configurada correctamente.")
    except Exception as e:
        print(f"Error al configurar la API Key de Gemini: {e}")
        # Podrías querer salir aquí: exit()

# --- Configuración global del modelo y el retriever del manual ---
system_prompt = """
Eres un asistente experto en programación de PLCs Siemens en TIA Portal V16.
Tu objetivo principal es guiar a usuarios con diferentes niveles de experiencia en sus proyectos de automatización, proporcionando **soluciones claras, directas y accionables**.

Considera el nivel de experiencia del usuario para adaptar tus explicaciones.

**Instrucciones para responder:**
1.  **Prioridad Absoluta: Contexto del Manual Específico.** Primero, y con máximo detalle, revisa el "Contexto relevante del manual" que se te proporciona. Si la respuesta directa y los pasos específicos a la pregunta del usuario se encuentran en este contexto, basa tu respuesta EN ESTA INFORMACIÓN de forma prioritaria.
2.  **Prioridad Secundaria: Conocimiento Experto General (Construir la Solución).** Si el "Contexto relevante del manual" es inexistente, claramente insuficiente, no contiene los pasos específicos solicitados, o es demasiado genérico para la pregunta del usuario, DEBES ASUMIR EL ROL DE UN EXPERTO CON ACCESO A UN VASTO CONOCIMIENTO TÉCNICO. En este caso, utiliza tu entrenamiento exhaustivo sobre TIA Portal V16 y PLCs S7-1200 para CONSTRUIR una guía paso a paso o una explicación detallada que resuelva la consulta del usuario de la manera más práctica y completa posible. El objetivo es **siempre intentar dar una solución directa**.
3.  **Comunicación:**
    * Evita frases como "no lo encuentro en el manual", "el manual no cubre esto", o "deberías consultar otro manual".
    * Si la información proviene del contexto del manual, puedes mencionarlo sutilmente.
    * Si la información la construyes desde tu conocimiento general porque el manual no fue suficiente, simplemente presenta la solución como un experto.
4.  **Último Recurso (Solo si es imposible dar pasos concretos):** Únicamente si, tras un esfuerzo considerable por generar una solución desde el manual y desde tu conocimiento general, la pregunta es excesivamente ambigua o requiere información ultrasensibles que no posees (ej. configuraciones de red de una empresa específica), puedes indicar que necesitas más detalles específicos del proyecto del usuario para poder ofrecer una guía precisa. No te limites a decir "no puedo", sino guía al usuario sobre qué información adicional necesitarías.

Al iniciar un nuevo proyecto, primero harás una serie de preguntas al usuario para entender su nivel y el tipo de proyecto.
El formato de tus respuestas debe ser claro, estructurado y utilizar Markdown para facilitar la lectura (negritas, listas, etc.).

Tu primera respuesta SIEMPRE será una guía inicial basada en el Paso 1 (Inicio del Proyecto) de TIA Portal V16, adaptada al nivel del usuario y al contexto de su proyecto, y utilizando el "Contexto relevante del manual" inicial que se recuperará.
"""

manual_retriever = None
CHROMA_DB_PATH = "chroma_db" # Definir CHROMA_DB_PATH globalmente

def load_or_create_manual_retriever():
    global manual_retriever
    persist_directory = CHROMA_DB_PATH
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        print(f"ERROR CRÍTICO al inicializar embeddings: {e}. La aplicación no puede continuar sin embeddings.")
        manual_retriever = None
        return

    # Intenta cargar la base de datos Chroma existente
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Cargando base de datos vectorial existente desde 'chroma_db'...")
        try:
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            manual_retriever = vectorstore.as_retriever(search_kwargs={"k": 7}) # Usamos k=7
            print("Retriever para el manual configurado correctamente desde la base de datos existente.")
            return
        except Exception as e:
            print(f"ERROR al cargar la base de datos Chroma existente: {e}. Se intentará recrear.")
            # Si falla la carga, eliminamos para intentar recrear
            try:
                shutil.rmtree(persist_directory)
                print(f"Carpeta '{persist_directory}' eliminada para intentar recreación.")
            except Exception as e_rm:
                print(f"ERROR al intentar eliminar la carpeta '{persist_directory}' para recreación: {e_rm}")
                manual_retriever = None
                return
    
    # Si no existe o falló la carga y se eliminó, intenta crearla desde los TXT
    print("Intentando crear una nueva base de datos vectorial desde los archivos TXT en la carpeta 'manuals'...")
    manuals_folder = 'manuals'
    documents = []

    if not os.path.exists(manuals_folder):
        print(f"ERROR: La carpeta '{manuals_folder}' no se encontró. No se puede crear la base de datos.")
        manual_retriever = None
        return

    for filename in os.listdir(manuals_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(manuals_folder, filename)
            try:
                loader = TextLoader(filepath, encoding='utf-8')
                documents.extend(loader.load())
                print(f"Cargado: {filename}")
            except Exception as e:
                print(f"Error al cargar {filename}: {e}")
    
    if not documents:
        print("No se encontraron documentos TXT para procesar en la carpeta 'manuals'.")
        manual_retriever = None
        return

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        print(f"Documentos divididos en {len(docs)} fragmentos.")

        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
        manual_retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
        print(f"Base de datos vectorial creada y guardada en '{persist_directory}'.")
        print("Retriever para el manual configurado correctamente desde los archivos TXT.")
    except Exception as e:
        print(f"ERROR CRÍTICO al crear la base de datos vectorial o el retriever: {e}")
        manual_retriever = None

# Cargar o crear el retriever al inicio de la aplicación
with app.app_context():
    load_or_create_manual_retriever()

# --- RUTAS DE LA APLICACIÓN ---
@app.route('/', methods=['GET'])
def index():
    session.pop('chat_display_history', None)
    session.pop('raw_responses_cuestionario', None)
    return render_template('cuestionario.html')

@app.route('/ask_gemini', methods=['POST'])
def ask_gemini():
    if manual_retriever is None:
        return "Error interno: El retriever del manual no está disponible. Revise los logs del servidor.", 500

    experiencia_tia = request.form.get('experiencia_tia', '')
    experiencia_plc = request.form.get('experiencia_plc', '')
    tipo_aplicacion = request.form.get('tipo_aplicacion', '')
    entradas_digitales = request.form.get('entradas_digitales', '')
    salidas_digitales = request.form.get('salidas_digitales', '')
    analogicas_info = request.form.get('analogicas_info', '')
    hmi_necesario = request.form.get('hmi_necesario', '')
    cpu_modelo = request.form.get('cpu_modelo', '')

    if 'experiencia_tia_otro' in request.form and request.form['experiencia_tia_otro']:
        experiencia_tia = request.form['experiencia_tia_otro']
    if 'experiencia_plc_otro' in request.form and request.form['experiencia_plc_otro']:
        experiencia_plc = request.form['experiencia_plc_otro']

    respuestas_cuestionario = f"""
    RESPUESTAS AL CUESTIONARIO INICIAL:
    - Nivel de experiencia en TIA Portal: {experiencia_tia}
    - Experiencia general con PLCs/Automatización: {experiencia_plc}
    - Detalles del Proyecto Actual:
        - Tipo de aplicación/proceso: {tipo_aplicacion}
        - Entradas digitales esperadas: {entradas_digitales}
        - Salidas digitales esperadas: {salidas_digitales}
        - Información analógica: {analogicas_info}
        - HMI necesaria: {hmi_necesario}
        - CPU/Modelo de PLC: {cpu_modelo}
    """

    try:
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        
        # Para la primera pregunta, el contexto se basa en el system_prompt y las respuestas del cuestionario
        combined_initial_query = system_prompt + "\n" + respuestas_cuestionario
        retrieved_docs_initial = manual_retriever.invoke(combined_initial_query)
        context_for_gemini_initial = "\n\n".join([doc.page_content for doc in retrieved_docs_initial])

        chat_history_for_model = [
            {"role": "user", "parts": [
                {'text': system_prompt}, # El system_prompt siempre va primero
                {'text': f"Contexto relevante del manual:\n{context_for_gemini_initial}"},
                {'text': respuestas_cuestionario}
            ]},
            {"role": "model", "parts": "Entendido. Procesando tu información y preparándome para guiarte."}
        ]
        
        chat = model_gemini.start_chat(history=chat_history_for_model)
        
        first_user_message_to_llm = "Basándome en tus respuestas, dame la primera guía para iniciar tu proyecto en TIA Portal V16, enfocándome en el Paso 1 (Inicio del Proyecto) y adaptando la explicación a tu nivel y contexto. Recuerda consultar el manual si es necesario."
        response = chat.send_message(first_user_message_to_llm)
        formatted_response = markdown.markdown(response.text)

        session['chat_display_history'] = [
            {"role": "user", "content": "Cuestionario completado. " + respuestas_cuestionario.replace('\n', ' ').strip()}, # Un resumen para el display
            {"role": "assistant", "content": formatted_response}
        ]
        session['raw_responses_cuestionario'] = respuestas_cuestionario
        
        print(f"DEBUG en /ask_gemini: chat_history que se pasa a la plantilla: {session.get('chat_display_history', [])}")
        return render_template('respuesta.html', chat_history=session.get('chat_display_history', []))

    except Exception as e:
        print(f"ERROR en /ask_gemini: {e}")
        return f"Error al interactuar con Gemini: {e}", 500

@app.route('/continue_chat', methods=['POST'])
def continue_chat():
    user_query = request.form['user_query']
    print(f"DEBUG en /continue_chat: user_query recibida: {user_query}")

    if manual_retriever is None:
        return "Error interno: El retriever del manual no está disponible para continuar el chat.", 500

    chat_display_history = session.get('chat_display_history', [])
    raw_responses_cuestionario = session.get('raw_responses_cuestionario', '')
    print(f"DEBUG en /continue_chat: chat_display_history recuperado de sesión: {chat_display_history}")

    retrieved_docs_current_query = manual_retriever.invoke(user_query)
    context_for_gemini_current_query = "\n\n".join([doc.page_content for doc in retrieved_docs_current_query])

    # Reconstruir historial para Gemini
    gemini_chat_history_rebuilt = []
    # 1. El System Prompt y el contexto del cuestionario + el contexto de la pregunta actual van en el PRIMER turno del usuario
    first_user_turn_parts = [
        {'text': system_prompt},
        {'text': f"Respuestas al cuestionario inicial:\n{raw_responses_cuestionario}"},
        {'text': f"Contexto adicional relevante del manual para la pregunta actual ('{user_query}'):\n{context_for_gemini_current_query}"}
    ]
    gemini_chat_history_rebuilt.append({"role": "user", "parts": first_user_turn_parts})
    gemini_chat_history_rebuilt.append({"role": "model", "parts": "Entendido. Usaré esta información y nuestro historial para responder."}) # Acuse de recibo genérico

    # 2. Añadir el historial de conversación anterior de la sesión
    for entry in chat_display_history:
        if entry['role'] == 'user':
            # Evitar duplicar el cuestionario si ya está implícito en el primer turno
            if not entry['content'].startswith("Cuestionario completado."):
                 gemini_chat_history_rebuilt.append({"role": "user", "parts": [{'text': entry['content']}]})
        elif entry['role'] == 'assistant':
            gemini_chat_history_rebuilt.append({"role": "model", "parts": [{'text': entry['content']}]})
    
    # La pregunta actual del usuario se envía con send_message, no se añade al historial ANTES.

    try:
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        chat = model_gemini.start_chat(history=gemini_chat_history_rebuilt)
        
        response = chat.send_message(user_query) # Aquí se envía la pregunta actual
        formatted_response = markdown.markdown(response.text)

        chat_display_history.append({"role": "user", "content": user_query})
        chat_display_history.append({"role": "assistant", "content": formatted_response})
        session['chat_display_history'] = chat_display_history
        
        print(f"DEBUG en /continue_chat: chat_history actualizado para plantilla: {chat_display_history}")
        return render_template('respuesta.html', chat_history=chat_display_history)

    except Exception as e:
        print(f"ERROR en /continue_chat: {e}")
        return f"Error al continuar la conversación con Gemini: {e}", 500

# --- EJECUTAR LA APLICACIÓN FLASK ---
if __name__ == '__main__':
    if manual_retriever is not None:
        print("Iniciando la aplicación Flask...")
        app.run(debug=False) # debug=False para producción, True para desarrollo
    else:
        print("La aplicación no se puede iniciar porque el retriever del manual no se configuró correctamente. Revise los logs.")