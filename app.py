import os
import json
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
from psycopg2 import pool
from psycopg2 import sql

# Cargar variables de entorno (para GEMINI_API_KEY y DATABASE_URL local)
load_dotenv()

# --- Configuración de Flask y Base de Datos ---
app = Flask(__name__)

# Obtener clave API y URL de la DB
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_URL = os.getenv("DATABASE_URL")

# Si no se encuentra la clave API, detener la aplicación inmediatamente (Crítico para seguridad)
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY no se encontró en las variables de entorno.")

# Inicializar cliente de Gemini
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error al inicializar el cliente de Gemini: {e}")
    # Esto puede ocurrir si la clave no está en el formato correcto
    raise

# Inicializar Pool de Conexiones de PostgreSQL (Recomendado para producción con Flask)
if DB_URL:
    try:
        # psycop2 se encarga de parsear la cadena DATABASE_URL
        db_pool = pool.SimpleConnectionPool(1, 20, DB_URL)
        print("Conexión a PostgreSQL establecida con éxito.")
    except Exception as e:
        print(f"Advertencia: No se pudo conectar a PostgreSQL usando DATABASE_URL. El guardado de leads fallará. Error: {e}")
        db_pool = None # Deshabilitar funciones de DB si falla
else:
    print("Advertencia: DATABASE_URL no está configurada. El guardado de leads estará deshabilitado.")
    db_pool = None

# --- Rutas del Servidor ---

@app.route('/')
def index():
    """Sirve el archivo HTML principal."""
    # En producción, usa send_from_directory, pero para este caso simple, render_template es suficiente
    return render_template('index.html')

# --- ENDPOINT para GUARDAR LEADS ---
@app.route('/save_lead', methods=['POST'])
def save_lead():
    """Recibe los datos del formulario y los guarda en PostgreSQL."""
    if not db_pool:
        # Si la conexión falló al inicio, devolvemos un 503 (Servicio no disponible)
        return jsonify({'error': 'El servicio de base de datos no está disponible.'}), 503
        
    data = request.get_json()
    
    # Validar campos esenciales
    nombre = data.get('nombre')
    telefono = data.get('telefono')
    servicio = data.get('servicio', 'No especificado')
    mensaje = data.get('mensaje', 'Contacto desde formulario web')
    
    if not nombre or not telefono:
        return jsonify({'error': 'Faltan campos obligatorios: nombre y teléfono.'}), 400

    conn = None
    try:
        # Obtener conexión del pool
        conn = db_pool.getconn()
        cur = conn.cursor()
        
        # Usar psycop2.sql para inyecciones SQL seguras
        insert_query = sql.SQL("""
            INSERT INTO leads (nombre, telefono, servicio, mensaje)
            VALUES (%s, %s, %s, %s)
        """)
        
        cur.execute(insert_query, (nombre, telefono, servicio, mensaje))
        conn.commit()
        
        cur.close()
        return jsonify({'success': True, 'message': 'Lead guardado con éxito.'}), 200

    except Exception as e:
        print(f"Error al guardar lead en DB: {e}")
        # Revertir cualquier cambio si hay un error
        if conn:
            conn.rollback()
        return jsonify({'error': 'Error interno del servidor al guardar el lead.', 'details': str(e)}), 500
        
    finally:
        # Devolver la conexión al pool
        if conn:
            db_pool.putconn(conn)


# --- ENDPOINT para el CHATBOT ---
@app.route('/chat', methods=['POST'])
def chat():
    """Recibe el historial de chat y devuelve una respuesta de Gemini."""
    data = request.get_json()
    contents = data.get('contents', [])
    system_instruction = data.get('systemInstruction')

    if not contents:
        return jsonify({'error': 'Faltan contenidos del chat.'}), 400
    
    try:
        # Llamar a la API de Gemini
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config={"system_instruction": system_instruction}
        )
        
        # Devolver la respuesta directamente
        return jsonify({
            'candidates': [
                {
                    'content': response.candidates[0].content.to_dict(),
                    'finish_reason': response.candidates[0].finish_reason.name
                }
            ]
        }), 200
    
    except APIError as e:
        # Manejo específico de errores de la API de Gemini
        print(f"Error de la API de Gemini: {e}")
        # Devolver 429 para Rate Limiting o 500 para otros errores de API
        status_code = 429 if 'RATE_LIMIT_EXCEEDED' in str(e) else 500
        return jsonify({'error': 'Error de la API de Gemini.', 'details': str(e)}), status_code
    
    except Exception as e:
        # Manejo de errores generales
        print(f"Error desconocido en la función de chat: {e}")
        return jsonify({'error': 'Error interno del servidor.', 'details': str(e)}), 500

# --- Inicio del Servidor ---
if __name__ == '__main__':
    # Usar el puerto proporcionado por Render o el 5000 por defecto
    port = int(os.environ.get('PORT', 5000))
    # En producción se debe usar Gunicorn (como se configura en Procfile), 
    # pero para pruebas locales, esto funciona.
    app.run(host='0.0.0.0', port=port, debug=True)