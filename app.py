import os
import requests
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from typing import List, Dict, Any

# Cargar variables de entorno del archivo .env
load_dotenv()

app = Flask(__name__)

# Configuración y Constantes
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY no se encontró en las variables de entorno.")

CHAT_MODEL = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{CHAT_MODEL}:generateContent?key={GEMINI_API_KEY}"

@app.before_request
def check_api_key():
    """Asegura que la clave API esté presente antes de procesar las solicitudes."""
    if not GEMINI_API_KEY:
        # Esto nunca debería pasar si se lanza la excepción en la inicialización,
        # pero es una capa de seguridad adicional.
        return jsonify({"error": "La clave API de Gemini no está configurada en el servidor."}), 500

@app.route('/')
def serve_index():
    """Sirve el archivo HTML principal."""
    # Nota: Si el archivo index.html estuviera en un directorio 'templates', 
    # usaríamos `render_template('index.html')`.
    # Asumiendo que está en la misma carpeta:
    return send_file('index.html')

@app.route('/chat', methods=['POST'])
def chat_proxy():
    """
    Endpoint que actúa como proxy entre el frontend y la API de Gemini.
    Recibe el historial de chat y las instrucciones del sistema, y devuelve la respuesta del modelo.
    """
    try:
        data: Dict[str, Any] = request.get_json()
        
        # El frontend debe enviar 'contents' (historial de chat) y 'systemInstruction'
        contents: List[Dict] = data.get('contents', [])
        system_instruction: str = data.get('systemInstruction', "")

        # Construir el payload para la API de Gemini
        payload = {
            "contents": contents,
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            }
        }
        
        # Llamada a la API de Gemini
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() # Lanza un error para códigos de estado 4xx/5xx

        # Devolver la respuesta de la API directamente al frontend
        return jsonify(response.json())

    except requests.exceptions.RequestException as e:
        # Manejo de errores de red o errores de la API (por ejemplo, clave inválida, límite de rate)
        print(f"Error al llamar a la API de Gemini: {e}")
        error_message = f"Error en el servidor al contactar al modelo AI. Detalles: {e}"
        if response is not None and response.status_code == 400:
            try:
                # Intenta devolver un error más específico si es un 400 (Bad Request)
                error_message = f"Error de la API: {response.json().get('error', {}).get('message', 'Solicitud incorrecta.')}"
            except:
                pass # Si el JSON no es parseable, usa el error genérico.
        return jsonify({"error": error_message}), 500
    
    except Exception as e:
        # Manejo de otros errores (JSON inválido, etc.)
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": "Error interno del servidor al procesar la solicitud."}), 500

if __name__ == '__main__':
    # Ejecuta la aplicación. Usar `host='0.0.0.0'` es útil para entornos de desarrollo/contenedores.
    print(f"Servidor Flask iniciado. Hablando con el modelo: {CHAT_MODEL}")
    app.run(debug=True, host='127.0.0.1', port=5000)
