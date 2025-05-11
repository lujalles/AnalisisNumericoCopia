import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from sympy import symbols, sympify, lambdify
import re

def newton_raphson(func_str_input, df_str_input, x0, tol, max_iter, x_min, x_max, nombre_pdf="reporte_newton.pdf"):
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
    errores = []
    xn = x0
    for i in range(1, max_iter + 1):
        fx = f(xn)
        dfx = df(xn)
        if dfx == 0:
            break
        xn_next = xn - fx / dfx
        error = abs(xn_next - xn)
        iteraciones.append([i, xn, fx, dfx, error])
        errores.append(error)
        if error < tol:
            xn = xn_next
            break
        xn = xn_next

    raiz = xn
    df_iter = pd.DataFrame(iteraciones, columns=["Iteración", "x", "f(x)", "f'(x)", "Error"])

    # Crear el PDF
    with PdfPages(nombre_pdf) as pdf:
        # Página de resumen
        resumen = (
            "REPORTE DEL MÉTODO DE NEWTON-RAPHSON\n\n"
            f"Función ingresada: f(x) = {func_str_input}\n"
            f"Derivada ingresada: f'(x) = {df_expr}\n"
            f"Valor inicial x0 = {x0}\n"
            f"Tolerancia = {tol}\n"
            f"Máx. iteraciones = {max_iter}\n"
            f"Raíz aproximada = {raiz:.6f}\n"
            f"Iteraciones realizadas: {len(iteraciones)}\n"
            f"Rango gráfico: x ∈ [{x_min}, {x_max}]\n"
        )
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0, 1, resumen, fontsize=12, verticalalignment='top')
        pdf.savefig()
        plt.close()

        # Página con tabla de iteraciones (dividida si es necesario)
        rows_per_page = 25
        for i in range(0, len(df_iter), rows_per_page):
            chunk = df_iter.iloc[i:i + rows_per_page]
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            table = ax.table(cellText=chunk.values, colLabels=chunk.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            ax.set_title(f"Newton-Raphson - Iteraciones (Página {i // rows_per_page + 1})", fontsize=12, pad=20)
            pdf.savefig()
            plt.close()

        # Gráfico de error vs iteración
        if errores:
            plt.figure()
            plt.plot(range(1, len(errores) + 1), errores, marker='o', label="Error absoluto")
            plt.xlabel("Iteración")
            plt.ylabel("Error")
            plt.title("Convergencia del método de Newton-Raphson")
            plt.grid(True)
            plt.legend()
            pdf.savefig()
            plt.close()

        # Gráfico de la función
        x_vals = np.linspace(x_min, x_max, 400)
        y_vals = f(x_vals)
        plt.figure()
        plt.plot(x_vals, y_vals, label="f(x)")
        plt.axhline(0, color='black', linestyle='--', label='y = 0')
        plt.plot(raiz, f(raiz), 'ro', label=f"Raíz ≈ {raiz:.6f}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Gráfico de la función f(x)")
        plt.grid(True)
        plt.legend()
        pdf.savefig()
        plt.close()

    return {
        "raiz": raiz,
        "iteraciones": df_iter,
        "pdf": nombre_pdf,
        "funcion_simbolica": f_expr,
        "derivada_simbolica": df_expr
    }
