# archivo: secante.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import symbols, lambdify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from fpdf import FPDF
import os

transformations = standard_transformations + (implicit_multiplication_application,)

def generar_pdf_secante(func_str_input, x0, x1, tol, max_iter, x_min, x_max):
    func_str = func_str_input.replace('^', '**')
    x = symbols('x')
    f_expr = parse_expr(func_str, transformations=transformations)
    f = lambdify(x, f_expr, modules=['numpy'])

    iteraciones = []
    f0 = f(x0)
    f1 = f(x1)
    for i in range(1, max_iter + 1):
        if f1 - f0 == 0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        error = abs(x2 - x1)
        iteraciones.append([i, x0, x1, x2, f1, error])
        if error < tol:
            break
        x0, x1 = x1, x2
        f0, f1 = f1, f(x1)

    raiz = x2
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = f(x_vals)
    plt.figure(figsize=(8,5))
    plt.plot(x_vals, y_vals, label='f(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    if x_min <= raiz <= x_max:
        plt.plot(raiz, f(raiz), 'ro', label=f'Raíz ≈ {raiz:.6f}')
    plt.title('Método de la Secante')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plot_file = "grafico_secante.png"
    plt.savefig(plot_file)
    plt.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Método de la Secante", ln=True, align='C')

    pdf.set_font("Arial", '', 12)
    pdf.ln(5)
    pdf.cell(0, 8, f"Función: {func_str_input}", ln=True)
    pdf.cell(0, 8, f"x0 inicial: {iteraciones[0][1] if iteraciones else x0}", ln=True)
    pdf.cell(0, 8, f"x1 inicial: {iteraciones[0][2] if iteraciones else x1}", ln=True)
    pdf.cell(0, 8, f"Tolerancia: {tol}", ln=True)
    pdf.cell(0, 8, f"Iteraciones ejecutadas: {len(iteraciones)}", ln=True)
    pdf.cell(0, 8, f"Raíz aproximada: {raiz:.6f}", ln=True)

    pdf.ln(5)
    col_names = ["Iter", "x0", "x1", "x2", "f(x1)", "Error"]
    col_widths = [15, 30, 30, 30, 30, 30]
    for name, w in zip(col_names, col_widths):
        pdf.cell(w, 8, name, border=1)
    pdf.ln()

    pdf.set_font("Arial", '', 11)
    for fila in iteraciones:
        for idx, val in enumerate(fila):
            text = str(int(val)) if idx == 0 else f"{val:.6f}"
            pdf.cell(col_widths[idx], 8, text, border=1)
        pdf.ln()

    pdf.ln(5)
    pdf.image(plot_file, x=10, w=190)

    pdf_file = f"secante_{raiz:.6f}.pdf"
    pdf.output(pdf_file)
    os.remove(plot_file)
    return pdf_file