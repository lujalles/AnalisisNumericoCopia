from flask import Flask, render_template, request, send_file
import os
from services.biseccion import biseccion
from services.regulafalsi import regula_falsi as generar_regula_falsi_pdf
from services.NEWTON import newton_raphson  # importa tu función
from services.secante import generar_pdf_secante
from services.jacobi import calcular_jacobi_y_generar_pdf
from services.Gauss_seidel import calcular_gauss_seidel_y_generar_pdf
import ast

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

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
@app.route('/biseccion_inicio')
def biseccion_inicio():
    return render_template('formulario_biseccion.html')

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
    
@app.route('/newton_inicio')
def newton_inicio():
    return render_template('formulario_newton.html')

@app.route('/newton_final', methods=['POST'])
def newton_final():
    try:
        # Obtener datos del formulario
        f = request.form['funcion']
        df = request.form['derivada']
        x0 = float(request.form['x0'])
        tol = float(request.form['tolerancia'])
        imax = int(request.form['max_iter'])
        xi = float(request.form['x_min'])
        xf = float(request.form['x_max'])
        
        # Convertir ^ a ** si es necesario
        f = f.replace('^', '**')
        df = df.replace('^', '**')
        
        # Generar el reporte
        resultado = newton_raphson(f, df, x0, tol, imax, xi, xf)
        
        # Verificar que el archivo se creó
        if not os.path.exists(resultado['pdf']):
            raise FileNotFoundError(f"No se pudo generar el PDF en {resultado['pdf']}")
        
        # Enviar el archivo como respuesta
        return send_file(
            resultado['pdf'],
            as_attachment=True,
            mimetype='application/pdf',
            download_name="Reporte_Newton_Raphson.pdf"
        )
        
    except Exception as e:
        # Mensaje de error simple si no existe error.html
        error_message = f"Error al generar el reporte: {str(e)}"
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .error {{ color: #d9534f; background-color: #f8f9fa; 
                        padding: 20px; border-radius: 5px; border: 1px solid #d9534f; }}
                a {{ color: #337ab7; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <h1>Error en el sistema</h1>
            <div class="error">
                <p>{error_message}</p>
                <p><a href="/newton_inicio">Volver al formulario</a></p>
            </div>
        </body>
        </html>
        """, 500
    
@app.route('/secante_inicio')

def secante_inicio():
    return render_template('formulario_secante.html')

@app.route('/secante_final', methods=['POST'])
def secante_final():
    func_str = request.form["func_str"]
    x0 = float(request.form["x0"])
    x1 = float(request.form["x1"])
    tol = float(request.form["tol"])
    max_iter = int(request.form["max_iter"])
    x_min = float(request.form["x_min"])
    x_max = float(request.form["x_max"])

    pdf_path = generar_pdf_secante(func_str, x0, x1, tol, max_iter, x_min, x_max)
    return send_file(pdf_path, as_attachment=True)

@app.route('/jacobi_inicio')
def jacobi_inicio():
    return render_template('formulario_jacobi.html')

@app.route('/jacobi_final', methods=['POST'])
def jacobi_final():
    A_str = request.form['matriz']
    b_str = request.form['vector']
    x0_str = request.form.get('x0', '')
    tol = float(request.form['tol'])
    max_iter = int(request.form['max_iter'])

    # Convertir los strings a listas
    try:
        A = ast.literal_eval(A_str)
        b = ast.literal_eval(b_str)
        x0 = ast.literal_eval(x0_str) if x0_str.strip() else None
    except Exception as e:
        return f"❌ Error en el formato de los datos: {e}"

    # Llamar a la función
    pdf_path = calcular_jacobi_y_generar_pdf(A, b, x0, tol, max_iter)

    # Devolver el PDF
    return send_file(pdf_path, as_attachment=True)

@app.route('/gauss_inicio')
def gauss_inicio():
    return render_template('formulario_gauss.html')

@app.route('/gauss_final', methods=['POST'])
def gauss_final():
    # Tomamos la matriz aumentada
    try:
        matriz_aumentada_str = request.form['matriz_aumentada']
        matriz_aumentada = ast.literal_eval(matriz_aumentada_str)

        A = [fila[:-1] for fila in matriz_aumentada]
        b = [fila[-1] for fila in matriz_aumentada]

        x0 = None  # O podrías permitirlo desde el form
        tol = float(request.form.get('tol', 1e-6))
        max_iter = int(request.form.get('max_iter', 100))

    except Exception as e:
        return f"❌ Error en el formato de los datos: {e}"

    # Llamar a la función que crea el PDF
    pdf_path = calcular_gauss_seidel_y_generar_pdf(A, b, x0, tol, max_iter)

    return send_file(pdf_path, as_attachment=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)