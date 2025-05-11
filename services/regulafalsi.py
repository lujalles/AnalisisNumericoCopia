import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sympy import symbols, lambdify, sympify
from tabulate import tabulate
from math import *
import pandas as pd 


def regula_falsi(f, a, b, es, imax, xi, xf):
    x = symbols("x")
    fn = sympify(f)
    f = lambdify(x, fn)


    cv, eav, iv = [], [], []
    i = 0
    ea = 100  # Inicia 'ea' con un valor grande para que entre en el ciclo while
    bisec_table = []

    while f(a) * f(b) > 0:
        print("No hay cambio de signo en el intervalo")
        a = int(input("Ingrese el limite inferior: "))
        b = int(input("Ingrese el limite superior: "))

    denom = f(b) - f(a)
    if denom == 0:
        raise ZeroDivisionError("División por cero en la fórmula del método Regula Falsi.")

    c = a - (f(a)*(b - a)) / denom

    bisec_table.append([i, a, b, f(a), f(b), c, "--"])
    
    while ea > es and i <= imax:
        cv.append(c)
        i += 1
        iv.append(i)

        c1 = c

        if f(a) * f(c) < 0:
            b = c1
        elif f(a) * f(c) > 0:
            a = c1
        else:
            ea = 0

        denom = f(b) - f(a)
        if denom == 0:
            raise ZeroDivisionError("División por cero en la fórmula del método Regula Falsi.")

        c = a - (f(a)*(b - a)) / denom

        if c != 0:
            ea = abs(c - c1)
        else:
            ea = 0  

        eav.append(ea)

        bisec_table.append([i, a, b, f(a), f(b), c, f(c), ea])


    x_vals = np.linspace(xi, xf, 100)
    y_vals = [f(val) for val in x_vals]

    with PdfPages('reporte_regulafalsi.pdf') as pdf:
        resumen = (
            "REPORTE DEL MÉTODO DE REGULA FALSI\n\n"
            f"Función ingresada: f(x) = {fn}\n"
            f"Intervalo inicial: a = {a}, b = {b}\n"
            f"Error tolerado: es = {es}\n"
            f"Número máximo de iteraciones: {imax}\n"
            f"Número iteraciones realizadas: {i}\n"
            f"Rango del gráfico: x ∈ [{xi}, {xf}]\n"
        )

        plt.axis('off')
        plt.text(0, 1, resumen, fontsize=12, verticalalignment='top')
        pdf.savefig()
        plt.close()
    
        df = pd.DataFrame(bisec_table[1:], columns=["Iteración", "a", "b", "f(a)", "f(b)", "c", "f(c)", "e abs"])
        rows_per_page = 20

        for i in range(0, len(df), rows_per_page):
            chunk_df = df.iloc[i:i + rows_per_page]

            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            table = ax.table(cellText=chunk_df.values, colLabels=chunk_df.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            ax.set_title(f"Método de Regula Falsi - Página {i // rows_per_page + 1}", fontsize=12, pad=20)
            pdf.savefig()
            plt.close()

        # Gráfica del error
        plt.figure()
        plt.plot(iv, eav, label="e absoluto")
        plt.xlabel("Número de iteraciones")
        plt.ylabel("Error absoluto")
        plt.title("Gráfica de error absoluto vs iteraciones")
        plt.legend()
        pdf.savefig()
        plt.close()

        # Gráfica de la función
        plt.figure()
        plt.plot(x_vals, y_vals, label="f(x)")
        plt.axhline(0, color='black', linestyle='--', label="y = 0")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Gráfica de la función f(x)")
        plt.legend()
        pdf.savefig()
        plt.close()

    return()





