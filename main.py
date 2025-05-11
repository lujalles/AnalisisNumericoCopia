from flask import Flask, render_template, request, send_file
import os
from services.biseccion import biseccion
from services.regulafalsi import regula_falsi as generar_regula_falsi_pdf
from services.newton import newton_raphson
from services.gauss import gauss_elimination as gauss_elimination
import re
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/newton_inicio')
def newton_inicio():
    return render_template('formulario_newton.html')

# Definir la ruta para procesar los datos del formulario y ejecutar el método
@app.route('/newton_final', methods=['POST'])
def newton_final():
    # Este bloque try-except es para capturar errores generales durante el procesamiento de la solicitud.
    try:
        # Obtener los datos enviados desde el formulario HTML a través del método POST
        # Usamos request.form para acceder a los datos del formulario.
        func_str_input = request.form['funcion']
        df_str_input = request.form['derivada']
        # Convertir los valores numéricos a float o int según corresponda.
        # Incluimos un try-except interno para capturar errores si los inputs no son números válidos.
        try:
            x0 = float(request.form['x0'])
            tol = float(request.form['tolerancia'])
            max_iter = int(request.form['max_iter'])
            x_min = float(request.form['x_min'])
            x_max = float(request.form['x_max'])
        except ValueError:
            # Si hay un error de conversión (ej: el usuario ingresó texto en un campo numérico)
            return "Error en los datos de entrada. Por favor, ingrese valores numéricos válidos.", 400 # HTTP 400 Bad Request

        # Definir el nombre del archivo PDF que se generará
        nombre_pdf = "reporte_newton.pdf"

        # Llamar a la función newton_raphson con los datos obtenidos del formulario
        # La función newton_raphson devuelve un diccionario con los resultados o un mensaje de error.
        resultado = newton_raphson(func_str_input, df_str_input, x0, tol, max_iter, x_min, x_max, nombre_pdf)

        # Verificar si la función newton_raphson devolvió un error controlado
        if resultado and resultado.get("error"):
            # Si hay un error (ej: derivada cero, error de sintaxis en la función)
            # Devolvemos el mensaje de error al usuario. Podrías renderizar una plantilla de error más amigable.
            return f"Error en el cálculo: {resultado['error']}", 400 # HTTP 400 Bad Request

        # Si no hubo errores y la función generó un PDF exitosamente
        # Verificamos si el archivo PDF realmente existe antes de intentar enviarlo.
        if os.path.exists(nombre_pdf):
            # Usar send_file para enviar el archivo PDF al navegador del usuario.
            # as_attachment=True fuerza la descarga del archivo.
            return send_file(nombre_pdf, as_attachment=True)
        else:
            # Si la función no reportó un error pero el archivo PDF no se encuentra
            return "Error interno: No se pudo generar el archivo PDF.", 500 # HTTP 500 Internal Server Error

    except Exception as e:
        # Capturar cualquier otro error inesperado que pueda ocurrir durante el manejo de la solicitud.
        return f"Ocurrió un error inesperado en el servidor: {e}", 500 # HTTP 500 Internal Server Error

    
@app.route('/biseccion_inicio')
def biseccion_inicio():
    return render_template('formulario_biseccion.html')

@app.route('/regulafalsi_inicio')
def regulafalsi_inicio():
    return render_template('formulario_regulafalsi.html')

@app.route('/regulafalsi_final', methods=['POST'])
def regulafalsi_final():
    f = request.form['f']
    a = float(request.form['a'])
    b = float(request.form['b'])
    es = float(request.form['es'])
    imax = int(request.form['imax'])
    xi = float(request.form['xi'])
    xf = float(request.form['xf'])

    generar_regula_falsi_pdf(f, a, b, es, imax, xi, xf)

    return send_file("reporte_regulafalsi.pdf", as_attachment=True)

@app.route('/biseccion_final', methods=['POST'])
def biseccion_final():
    f = request.form['funcion']
    a = float(request.form['a'])
    b = float(request.form['b'])
    es = float(request.form['es'])
    imax = int(request.form['imax'])
    xi = float(request.form['xi'])
    xf = float(request.form['xf'])

    # Llama a tu función y genera el PDF
    biseccion(f, a, b, es, imax, xi, xf)

    # Devuelve el PDF generado
    return send_file('reporte_biseccion.pdf', as_attachment=True)

# Definir la ruta para mostrar el formulario de Eliminación Gaussiana
@app.route('/gauss_inicio')
def gauss_inicio():
    # Renderiza el archivo HTML del formulario.
    # Asume que el archivo se llama 'formulario_gauss.html' y está en una carpeta 'templates'.
    return render_template('formulario_gauss.html')

# Definir la ruta para procesar los datos del formulario y ejecutar el método
@app.route('/gauss_final', methods=['POST'])
def gauss_final():
    # Este bloque try-except es para capturar errores generales durante el procesamiento de la solicitud.
    try:
        # Obtener los datos enviados desde el formulario HTML a través del método POST
        # La matriz se espera como una cadena de texto en el campo 'matriz_aumentada'.
        # Los checkboxes envían 'on' si están marcados, o no envían nada si no lo están.
        matrix_string = request.form.get('matriz_aumentada', '').strip()
        use_pivoting = 'pivoteo' in request.form # Verifica si el checkbox 'pivoteo' fue enviado
        show_steps = 'mostrar_pasos' in request.form # Verifica si el checkbox 'mostrar_pasos' fue enviado

        # Validar que la cadena de la matriz no esté vacía
        if not matrix_string:
             return "Error: No se proporcionó la matriz aumentada.", 400 # HTTP 400 Bad Request

        # Definir el nombre del archivo PDF que se generará
        nombre_pdf = "reporte_gauss.pdf"

        # Llamar a la función gauss_elimination con los datos obtenidos del formulario
        # La función devuelve un diccionario con los resultados o un mensaje de error.
        resultado = gauss_elimination(matrix_string, use_pivoting=use_pivoting, show_steps=show_steps, nombre_pdf=nombre_pdf)

        # Verificar si la función gauss_elimination devolvió un error controlado
        if resultado and resultado.get("error"):
            # Si hay un error (ej: matriz singular, error de parseo)
            # Devolvemos el mensaje de error al usuario. Podrías renderizar una plantilla de error más amigable.
            return f"Error en el cálculo: {resultado['error']}", 400 # HTTP 400 Bad Request

        # Si no hubo errores y la función generó un PDF exitosamente
        # Verificamos si el archivo PDF realmente existe antes de intentar enviarlo.
        if os.path.exists(nombre_pdf):
            # Usar send_file para enviar el archivo PDF al navegador del usuario.
            # as_attachment=True fuerza la descarga del archivo.
            return send_file(nombre_pdf, as_attachment=True)
        else:
            # Si la función no reportó un error pero el archivo PDF no se encuentra
            return "Error interno: No se pudo generar el archivo PDF.", 500 # HTTP 500 Internal Server Error

    except Exception as e:
        # Capturar cualquier otro error inesperado que pueda ocurrir durante el manejo de la solicitud.
        return f"Ocurrió un error inesperado en el servidor: {e}", 500 # HTTP 500 Internal Server Error
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)