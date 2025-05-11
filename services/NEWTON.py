import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import symbols, sympify, lambdify, diff
from fpdf import FPDF
import os
import re

def newton_raphson(func_str_input, df_str_input, x0, tol, max_iter, x_min, x_max, nombre_pdf="resultado_newton.pdf"):

    # Normalizar función
    def normalizar_funcion(expr):
        expr = expr.replace('^', '')
        expr = re.sub(r'(?<=\d)(x)', r'*x', expr)
        expr = re.sub(r'(x)(?=\d)', r'x*', expr)
        expr = re.sub(r'(?<![\w)])x', '1*x', expr)
        return expr

    # Preparar funciones
    x = symbols('x')
    func_str = normalizar_funcion(func_str_input)
    df_str = normalizar_funcion(df_str_input)
    f_expr = sympify(func_str)
    df_expr = sympify(df_str)
    f = lambdify(x, f_expr, modules=['numpy'])
    df = lambdify(x, df_expr, modules=['numpy'])

    # Método de Newton-Raphson
    iteraciones = []
    xn = x0
    for i in range(1, max_iter + 1):
        fx = f(xn)
        dfx = df(xn)
        if dfx == 0:
            break
        xn_next = xn - fx / dfx
        error = abs(xn_next - xn)
        iteraciones.append([i, xn, fx, dfx, error])
        if error < tol:
            xn = xn_next
            break
        xn = xn_next

    raiz = xn
    df_iter = pd.DataFrame(iteraciones, columns=["Iteración", "x", "f(x)", "f'(x)", "Error"])

    # Gráfico
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = f(x_vals)
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label='f(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.plot(raiz, f(raiz), 'ro', label=f'xn final ≈ {raiz:.6f}')
    plt.title('Gráfico de la función')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plot_file = "grafico_newton.png"
    plt.savefig(plot_file)
    plt.close()

    # PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Método de Newton-Raphson", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.ln(5)
    pdf.cell(0, 8, f"Función: {func_str_input}", ln=True)
    pdf.cell(0, 8, f"Derivada: {df_expr}", ln=True)
    pdf.cell(0, 8, f"Valor inicial x0: {x0}", ln=True)
    pdf.cell(0, 8, f"Tolerancia: {tol}", ln=True)
    pdf.cell(0, 8, f"Iteraciones ejecutadas: {len(iteraciones)}", ln=True)
    pdf.cell(0, 8, f"Raíz aproximada: {raiz:.6f}", ln=True)

    pdf.ln(5)
    col_names = ["Iter", "x", "f(x)", "f'(x)", "Error"]
    col_widths = [15, 35, 35, 35, 35]
    for name, w in zip(col_names, col_widths):
        pdf.cell(w, 8, name, border=1)
    pdf.ln()
    pdf.set_font("Arial", '', 11)
    for fila in iteraciones:
        for i, dato in enumerate(fila):
            pdf.cell(col_widths[i], 8, f"{dato:.6f}", border=1)
        pdf.ln()
    pdf.ln(5)
    pdf.image(plot_file, x=10, w=190)
    pdf.output(nombre_pdf)
    os.remove(plot_file)

    return {
        "raiz": raiz,
        "iteraciones": df_iter,
        "pdf": nombre_pdf,
        "funcion_simbolica": f_expr,
        "derivada_simbolica": df_expr
    }