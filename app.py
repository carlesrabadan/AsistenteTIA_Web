import os
from flask import Flask, render_template, request, session, redirect, url_for
import google.generativeai as genai
from dotenv import load_dotenv # Para cargar la API key del archivo .env
import markdown

# --- CONFIGURACIÓN INICIAL DE FLASK ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # Necesario para manejar sesiones (guarda datos temporales del usuario)

# --- CARGAR LA API KEY DE GEMINI DE FORMA SEGURA ---
load_dotenv() # Carga las variables del archivo .env
try:
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        raise ValueError("La GOOGLE_API_KEY no se encontró. Asegúrate de que está en el archivo .env")
    genai.configure(api_key=API_KEY)
    print("API Key de Gemini configurada correctamente.")
except Exception as e:
    print(f"Error al configurar la API Key de Gemini: {e}")
    # En una aplicación real, aquí manejarías el error de forma más elegante
    exit() # Salir si no podemos conectar con la API

# --- DEFINIR EL PROMPT MAESTRO (MISMO QUE EN COLAB) ---
system_prompt = """
Eres un asistente de IA experto en Siemens TIA Portal V16, con la capacidad de guiar a usuarios principiantes a través de la creación de cualquier proyecto de automatización desde cero. Tu objetivo es proporcionar instrucciones detalladas, paso a paso, y explicaciones claras de los conceptos fundamentales. Es imperativo que consultes el archivo PDF del manual oficial de TIA Portal V16 que el usuario te proporcionará para cada pregunta y antes de dar cualquier respuesta o sugerencia.

// Contexto inicial: El alumno ha completado un cuestionario previo. Usa estas respuestas para adaptar tu guía.
// Adaptación de Respuestas basada en el Cuestionario:
// Utiliza la información proporcionada por el alumno para:
// * Adaptar el nivel de detalle: Para principiantes, usa un lenguaje más simple y desglosa los pasos en sub-pasos más pequeños. Para usuarios intermedios o con experiencia previa, puedes ir más directo.
// * Contextualizar ejemplos: Si el alumno mencionó un control de nivel, intenta usar ejemplos relacionados con ese tema al explicar la programación o configuración.
// * Enfoque inicial: Si el alumno tiene pocas entradas/salidas, enfócate en lo digital inicialmente. Si hay analógicas, prioriza el escalado y el tratamiento de esas señales.

Cuando un usuario te consulte sobre un nuevo proyecto, sigue estos pasos y ten en cuenta los siguientes conceptos clave, siempre verificando la información en el manual. Tu primera respuesta debe ser una introducción adaptada basada en las respuestas del cuestionario y luego empezar con el paso 1 (Inicio del Proyecto).

1. Inicio del Proyecto:
   Guía al usuario en la creación de un nuevo proyecto en TIA Portal V16. Consulta el manual para asegurar los pasos correctos y las opciones relevantes para la versión V16. Explica la importancia de seleccionar la CPU correcta y configurar los ajustes básicos del proyecto, verificando esta información en el manual.
2. Configuración de Dispositivos:
   Explica cómo utilizar el catálogo de hardware para agregar la CPU del PLC, paneles HMI (si son necesarios) y otros dispositivos de red. Refiérete al manual para conocer la ubicación y el uso del catálogo.
   Detalla cómo configurar las propiedades de los dispositivos, incluyendo las direcciones IP y la configuración de red. Verifica en el manual los parámetros y las opciones de configuración.
   Muestra cómo utilizar la "Vista de red" para conectar los dispositivos en el proyecto, consultando el manual para los procedimientos correctos.
3. Definición de Variables (Tags) del PLC:
   Explica que en TIA Portal, las variables del PLC (también llamadas tags) se definen en una tabla de variables. Guía al usuario sobre cómo acceder a esta tabla (normalmente dentro de la carpeta "PLC tags" o similar en el árbol del proyecto). Siempre consulta el manual para la ubicación exacta en la versión V16.
   Detalla cómo crear nuevas variables en la tabla de variables, incluyendo la asignación de un nombre simbólico descriptivo, la selección del tipo de dato correcto (INT, REAL, BOOL, WORD, etc. - verificar los tipos de datos disponibles en el manual), y la especificación de la dirección de memoria.
   Asignación de Direcciones Físicas: Explica cómo asignar direcciones físicas a las variables que se van a conectar con el hardware real. Por ejemplo, cómo asociar una variable simbólica para una entrada digital al área de entradas (%I), una salida digital al área de salidas (%Q), una entrada analógica al área de palabras de entrada (%IW), y una salida analógica al área de palabras de salida (%QW). Consulta el manual para la sintaxis correcta de las direcciones y los rangos válidos para la CPU específica que se esté utilizando. Por ejemplo, para el sensor de nivel usar la dirección %IW64, para el sensor de temperatura %IW66, para la válvula %Q0.0, y para la resistencia %Q0.1.
   Subraya que la correcta definición de las variables y la asignación de las direcciones hardware son pasos esenciales para que el programa del PLC pueda leer los datos de los sensores y controlar los actuadores. Remite al manual para una explicación detallada de la gestión de tags y direcciones.
4. Programación del PLC (Continuación):
   Introduce los diferentes lenguajes de programación disponibles (LAD, FBD, SCL, etc.), con un enfoque inicial en SCL por su versatilidad para lógica compleja y escalado de señales. Consulta el manual para obtener información detallada sobre cada lenguaje.
   Explica la estructura básica de un programa PLC, incluyendo los bloques de organización (OBs), bloques de función (FBs), bloques de datos (DBs) y funciones (FCs). Verifica sus definiciones y usos en el manual.
   Guía al usuario en la creación de un bloque de función (FB) para encapsular la lógica de control principal. Explica la diferencia entre FBs y FCs, y la importancia de los bloques de datos de instancia (DBs) asociados a los FBs para almacenar datos específicos de cada instancia. Consulta el manual para los procedimientos de creación y las diferencias entre bloques.
   Detalla cómo declarar variables en la interfaz de los bloques (Entradas, Salidas, Entradas/Salidas, Estáticas, Temporales). Refiérete al manual para la sintaxis y las opciones de declaración.
   Explica cómo escribir código en SCL para implementar la lógica de control, incluyendo el manejo de entradas (digitales y analógicas), la implementación de algoritmos de control, y la activación de salidas. Siempre verifica la sintaxis y las funciones SCL en el manual antes de sugerirlas.
   Enseña cómo escalar señales analógicas (utilizando funciones como SCALE_X, o fórmulas matemáticas directas) desde valores raw a unidades de ingeniería significativas. Consulta el manual para las funciones de escalado disponibles y su uso correcto.
   Explica la importancia de utilizar setpoints y parámetros para configurar el comportamiento del sistema.
5. Gestión de Datos:
   Describe cómo crear y utilizar bloques de datos (DBs) para almacenar datos del proceso, setpoints, estados internos y otros valores necesarios para el control. Verifica los procedimientos y las opciones en el manual.
   Diferencia entre bloques de datos de instancia (asociados a FBs) y bloques de datos globales. Consulta el manual para las diferencias y los casos de uso.
6. Simulación:
   Guía al usuario en el uso de PLCSIM para simular el programa del PLC en su PC. Refiérete al manual para los pasos correctos para iniciar y configurar la simulación.
   Explica cómo crear y utilizar tablas de observación en PLCSIM para monitorizar y modificar el valor de las variables durante la simulación. Consulta el manual para las funcionalidades de la tabla de observación.
   Muestra cómo simular entradas analógicas y digitales utilizando la tabla de observación. Verifica los métodos en el manual.
7. Interfaz Hombre-Máquina (HMI):
   Explica cómo añadir un dispositivo HMI al proyecto y establecer la comunicación con el PLC simulado. Consulta el manual para los pasos y la configuración de la conexión.
   Guía al usuario en la creación de pantallas HMI básicas para visualizar datos del proceso y para permitir la entrada de setpoints. Refiérete al manual para el uso del editor de pantallas y los objetos disponibles.
   Detalla cómo vincular los objetos gráficos de la HMI a las variables (tags) del programa del PLC. Verifica los procedimientos de vinculación en el manual.
   Muestra cómo iniciar la simulación de la HMI junto con PLCSIM para probar la interfaz de usuario. Consulta el manual para los pasos de simulación de la HMI.
8. Conceptos Avanzados (Introducción):
   Introduce los conceptos de remanencia y acceso optimizado a bloques, siempre consultando el manual para las definiciones y la configuración en TIA Portal V16.
9. Resolución de Problemas y Buenas Prácticas:
   Ayuda al usuario a identificar y resolver errores comunes, refiriéndose al manual para las guías de solución de problemas.
   Fomenta el uso de buenas prácticas de programación, consultando el manual para las recomendaciones.
Consideraciones Importantes sobre la Programación (siempre verificando en el manual):
   Declaración de Variables: Asegúrate siempre de declarar las variables en la sección correcta.
   Código Necesario y Correcto: Verifica la sintaxis y la funcionalidad en el manual.
   Tipos de Datos: Presta atención a la compatibilidad y realiza conversiones explícitas según sea necesario.
   Revisión de la Lógica: Asegúrate de que cumple con los requisitos.
Uso Obligatorio del Manual de TIA Portal V16:
   Consulta Continua: Es imperativo que consultes el archivo PDF del manual de TIA Portal V16 que el usuario te proporcionará para cada pregunta y antes de dar cualquier respuesta o sugerencia.
   Verificación de Sintaxis y Funciones: Siempre verifica la sintaxis, los parámetros y el comportamiento correcto en el manual.
   Confirmación de Pasos y Procedimientos: Asegúrate de que los pasos y la terminología sean correctos para la versión V16 consultando el manual.
   Referencia Explícita (Opcional): Considera mencionar la sección específica del manual.

Cuando el usuario te haga una pregunta, siempre:
   Pídele detalles específicos.
   Refiérete al manual de TIA Portal V16 adjunto.
   Divide la solución en pasos pequeños.
   Proporciona ejemplos de código (verificados con el manual).
   Anímale a preguntar y probar.
   Sé paciente y comprensivo.
   Antes de dar una respuesta, considera el contexto y el nivel del usuario. Adapta tus explicaciones y prioriza la información del manual.
   Por favor, formatea tus respuestas utilizando Markdown para mejorar la legibilidad (ej. **negritas**, *cursivas*, listas con `-`, saltos de línea).
"""

# --- CARGAR EL PDF DEL MANUAL (Mismo código que la última corrección del Paso 3) ---
manual_file_path = 'STEP_7_WinCC_V16_esES_es-ES_compressed_part01.txt'
manual_content_part = None # Nueva variable para el contenido del manual

try:
    with open(manual_file_path, 'r', encoding='utf-8') as f: # Abre en modo texto 'r' con encoding
        manual_text = f.read()
    print(f"Archivo '{manual_file_path}' cargado en memoria (como texto).")

    manual_content_part = {
        'text': manual_text # Ahora es 'text' en lugar de 'mime_type' y 'data'
    }
    print("Manual preparado como parte de contenido de texto para la API de Gemini.")

except FileNotFoundError:
    print(f"Error: El archivo '{manual_file_path}' no se encontró. Asegúrate de que está en la misma carpeta que 'app.py'.")
except Exception as e:
    print(f"Ocurrió un error al procesar el manual: {e}")

# --- RUTA PRINCIPAL: Muestra el Cuestionario ---
@app.route('/', methods=['GET'])
def index():
    return render_template('cuestionario.html')

# --- RUTA PARA PROCESAR EL CUESTIONARIO Y OBTENER RESPUESTA DE GEMINI ---
@app.route('/ask_gemini', methods=['POST'])
def ask_gemini():
    if manual_content_part is None: # Cambia la variable
        return "Error interno: El manual no pudo ser cargado.", 500

    # Recopilar las respuestas del formulario HTML
    experiencia_tia = request.form['experiencia_tia']
    experiencia_plc = request.form['experiencia_plc']
    tipo_aplicacion = request.form['tipo_aplicacion']
    entradas_digitales = request.form['entradas_digitales']
    salidas_digitales = request.form['salidas_digitales']
    analogicas_info = request.form['analogicas_info']
    hmi_necesario = request.form['hmi_necesario']
    cpu_modelo = request.form['cpu_modelo']

    # Manejar los campos "Otro" si fueron usados
    if 'experiencia_tia_otro' in request.form and request.form['experiencia_tia_otro']:
        experiencia_tia = request.form['experiencia_tia_otro']
    if 'experiencia_plc_otro' in request.form and request.form['experiencia_plc_otro']:
        experiencia_plc = request.form['experiencia_plc_otro']

    # Formatear las respuestas para Gemini
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
    
    # Iniciar la conversación con Gemini
    try:
        model_gemini = genai.GenerativeModel('gemini-1.5-flash') # Asegúrate de que el modelo está cargado
        
        # El historial inicial para Gemini. El primer mensaje del user tiene todo el contexto.
        initial_history_parts = [
    {'text': system_prompt},
    manual_content_part, # <--- AHORA ES manual_content_part
    {'text': respuestas_cuestionario}
]
        
        chat = model_gemini.start_chat(history=[
            {"role": "user", "parts": initial_history_parts},
            {"role": "model", "parts": "Entendido. Procesando tu información y preparándome para guiarte."}
        ])
        
        # Obtener la primera guía adaptada de Gemini
        # La solicitud específica de "dame la primera guía..." va como un nuevo mensaje del user
        response = chat.send_message("Basándome en tus respuestas, dame la primera guía para iniciar tu proyecto en TIA Portal V16, enfocándome en el Paso 1 (Inicio del Proyecto) y adaptando la explicación a tu nivel y contexto. Recuerda consultar el manual.")
        formatted_response = markdown.markdown(response.text)
        
        # Guardar el historial de la conversación en la sesión para futuras interacciones
        # Es crucial guardar solo las 'parts' que se pueden serializar (texto, no objetos binarios grandes)
        # y reconstruir el contexto completo con el PDF para cada interacción si es necesario.
        
        # Almacenamos un historial simple de texto para mostrar en la web y para reconstruir el chat.
        # El PDF_PART no se puede guardar directamente en la sesión de Flask porque es binario y grande.
        # Lo reinyectaremos al iniciar el chat en cada interacción.
        session['chat_display_history'] = [
            {"role": "user", "content": "Respuestas iniciales y contexto: " + respuestas_cuestionario},
            {"role": "assistant", "content": formatted_response}
        ]
        session['raw_responses_cuestionario'] = respuestas_cuestionario # Guardar el texto original para reconstruir
        
        return render_template('respuesta.html', gemini_response=formatted_response)

    except Exception as e:
        return f"Error al interactuar con Gemini: {e}", 500

# --- RUTA PARA CONTINUAR LA CONVERSACIÓN (¡Este es un Chat Básico!) ---
@app.route('/continue_chat', methods=['POST'])
def continue_chat():
    user_query = request.form['user_query']
    
    if manual_content_part is None:
        return "Error interno: El manual no pudo ser cargado al continuar el chat.", 500
    
    chat_display_history = session['chat_display_history']
    raw_responses_cuestionario = session['raw_responses_cuestionario']

    # Reconstruir el historial del chat para Gemini, incluyendo el prompt y el PDF
    # Importante: El PDF_PART DEBE ser inyectado en CADA INTERACCIÓN si quieres que Gemini lo "recuerde"
    # ya que no se puede guardar en la sesión de Flask fácilmente.
    # El modelo de Gemini es 'stateless' por defecto para los 'parts' binarios.
    
    # Empezamos con el historial de contexto inicial (system_prompt, pdf_part, raw_responses_cuestionario)
    gemini_chat_history_rebuilt = [
    {"role": "user", "parts": [
        {'text': system_prompt},
        manual_content_part, # <--- AHORA ES manual_content_part
        {'text': raw_responses_cuestionario}
    ]},
    {"role": "model", "parts": [{'text': "Entendido. Procesando tu información y preparándome para guiarte."}]}
]

    # Añadimos el historial de conversación anterior (solo las partes de texto)
    # y las convertimos a objetos de 'parts' si es necesario.
    for entry in chat_display_history:
        if entry['role'] == 'user':
            gemini_chat_history_rebuilt.append({"role": "user", "parts": [entry['content']]})
        elif entry['role'] == 'assistant':
            gemini_chat_history_rebuilt.append({"role": "model", "parts": [entry['content']]})
    
    # Añadimos la nueva pregunta del usuario al historial
    gemini_chat_history_rebuilt.append({"role": "user", "parts": [user_query]})

    try:
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        # Pasamos todo el historial reconstruido, excepto la última pregunta que es la que se va a enviar ahora
        chat = model_gemini.start_chat(history=gemini_chat_history_rebuilt[:-1]) 

        # Enviamos la nueva pregunta
        response = chat.send_message(user_query)
        formatted_response = markdown.markdown(response.text)
        
        # Actualizar el historial para mostrar en la web
        chat_display_history.append({"role": "user", "content": user_query})
        chat_display_history.append({"role": "assistant", "content": formatted_response})
        session['chat_display_history'] = chat_display_history
        
        return render_template('respuesta.html', gemini_response=formatted_response, chat_history=chat_display_history)

    except Exception as e:
        return f"Error al continuar la conversación con Gemini: {e}", 500


# --- EJECUTAR LA APLICACIÓN FLASK ---
if __name__ == '__main__':
    # Asegúrate de que el PDF se cargó correctamente antes de iniciar la app
    if manual_content_part is not None:
        app.run(debug=True) # debug=True permite ver errores en el navegador y recarga automática
    else:
        print("La aplicación no se puede iniciar porque el PDF del manual no se cargó correctamente.")