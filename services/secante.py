import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from sympy import symbols, sympify, lambdify
from datetime import datetime
from services.pdf_generator import generate_pdf_report

def secante(f_str, x0, x1, tol, max_iter):
    x = symbols('x')
    fn = sympify(f_str)
    f = lambdify(x, fn, modules=['numpy'])

    history = []
    f_x0 = f(x0)
    f_x1 = f(x1)
    x2 = x1 - (f_x1 * (x1 - x0)) / (f_x1 - f_x0)
    error = abs(x2 - x1)
    history.append([0, x0, x1, x2, f_x1, None])

    i = 0
    while error > tol and i < max_iter:
        i += 1
        x0, x1 = x1, x2
        f_x0 = f(x0)
        f_x1 = f(x1)
        denom = f_x1 - f_x0
        if denom == 0:
            raise ZeroDivisionError("División por cero en el método de la Secante")

        x2 = x1 - (f_x1 * (x1 - x0)) / denom
        error = abs(x2 - x1)
        history.append([i, x0, x1, x2, f_x1, error])

    return x2, history, f

def calcular_secante_y_generar_pdf(f_str, x0, x1, tol, max_iter, xi, xf):
    try:
        raiz, history, f = secante(f_str, x0, x1, tol, max_iter)
        fn_str = f_str

        params = {
            'x0': x0,
            'x1': x1,
            'es': tol,
            'imax': max_iter,
            'xi': xi,
            'xf': xf
        }

        last = history[-1]
        results = {
            'root': raiz,
            'f_root': f(raiz),
            'iterations': len(history) - 1,
            'final_error': last[5] if last[5] is not None else 0.0
        }

        df = pd.DataFrame(history, columns=["Iteración", "x0", "x1", "x2", "f(x1)", "Error"])
        for col in df.columns:
            if col != "Iteración":
                df[col] = df[col].apply(lambda x: f"{x:.6f}" if pd.notnull(x) else "")

        tables = [df.iloc[i:i + 12] for i in range(0, len(df), 12)]

        graphs = []

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        df_err = df[df["Error"] != ""].copy()
        df_err["Iteración"] = df_err["Iteración"].astype(int)
        df_err["Error"] = df_err["Error"].astype(float)
        ax1.plot(df_err["Iteración"], df_err["Error"], 'o-', color='#e74c3c')
        ax1.set_title("Evolución del error")
        ax1.set_xlabel("Iteración")
        ax1.set_ylabel("Error absoluto")
        ax1.grid(True)
        graphs.append((fig1, "Evolución del error por iteración"))

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        x_vals = np.linspace(xi, xf, 400)
        y_vals = f(x_vals)
        ax2.plot(x_vals, y_vals, label=f"f(x) = {f_str}", color='#3498db')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.plot(raiz, f(raiz), 'ro', label=f"Raíz ≈ {raiz:.6f}")
        ax2.set_title("Gráfico de la función")
        ax2.set_xlabel("x")
        ax2.set_ylabel("f(x)")
        ax2.grid(True)
        ax2.legend()
        graphs.append((fig2, "Representación gráfica de la función"))

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        x2_vals = df["x2"].replace("", "0").astype(float)
        ax3.plot(df["Iteración"], x2_vals, 'o-', color='#2ecc71')
        ax3.axhline(raiz, color='#e74c3c', linestyle='--', alpha=0.7, label=f'Raíz ({raiz:.6f})')
        ax3.set_title("Convergencia del punto x2")
        ax3.set_xlabel("Iteración")
        ax3.set_ylabel("x2")
        ax3.grid(True)
        ax3.legend()
        graphs.append((fig3, "Convergencia de x2 por iteración"))

        os.makedirs("static", exist_ok=True)
        output_path = os.path.join("static", "reporte_secante.pdf")

        generate_pdf_report(
            output_filename=output_path,
            method_name="Secante",
            fn_str=fn_str,
            params=params,
            results=results,
            tables=tables,
            graphs=graphs
        )

        return output_path

    except Exception as e:
        print(f"Error al generar PDF de Secante: {e}")
        raise
