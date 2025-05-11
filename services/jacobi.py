# archivo: jacobi_solver.py
import numpy as np
from fpdf import FPDF
import os

def jacobi(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    x = x0.copy()
    history = []

    for k in range(1, max_iter + 1):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        history.append((k, x_new.copy()))
        if np.linalg.norm(np.dot(A, x_new) - b, np.inf) < tol:
            break
        x = x_new

    return x, history

def calcular_jacobi_y_generar_pdf(A, b, x0=None, tol=1e-6, max_iter=100, output_path="static/jacobi_resultado.pdf"):
    solucion, pasos = jacobi(A, b, x0, tol, max_iter)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Método de Jacobi - Resultado", ln=True, align="C")
    pdf.ln(10)

    # Solución
    sol_text = "x = [" + ", ".join(f"{xi:.6f}" for xi in solucion) + "]"
    pdf.cell(200, 10, txt="Solución final:", ln=True)
    pdf.multi_cell(0, 10, txt=sol_text)
    pdf.ln(5)

    # Iteraciones
    pdf.cell(200, 10, txt="Iteraciones:", ln=True)
    for k, xk in pasos:
        line = f"Iteración {k}: " + ", ".join(f"x{i+1} = {val:.6f}" for i, val in enumerate(xk))
        pdf.multi_cell(0, 10, txt=line)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)

    return output_path  # Para devolver al usuario
