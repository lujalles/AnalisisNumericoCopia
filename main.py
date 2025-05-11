from flask import Flask, render_template, request, send_file
import os
from services.biseccion import biseccion
from services.regulafalsi import regula_falsi as generar_regula_falsi_pdf

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/newton_inicio')
def newton_inicio():
    return render_template('formulario_newton.html')

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
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)