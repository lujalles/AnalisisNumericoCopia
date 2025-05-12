import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from sympy import symbols, sympify, lambdify
from datetime import datetime
from services.pdf_generator import generate_pdf_report

def regula_falsi(f_str, a, b, tol, max_iter):
    x = symbols('x')
    fn = sympify(f_str)
    f = lambdify(x, fn, modules=['numpy'])

    if f(a) * f(b) > 0:
        raise ValueError("No hay cambio de signo en el intervalo dado.")

    history = []
    denom = f(b) - f(a)
    if denom == 0:
        raise ZeroDivisionError("División por cero")

    c = a - (f(a) * (b - a)) / denom
    error = abs(b - a)
    history.append([0, a, b, f(a), f(b), c, f(c), None])

    i = 0
    while error > tol and i < max_iter:
        i += 1
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

        denom = f(b) - f(a)
        if denom == 0:
            raise ZeroDivisionError("División por cero")

        c_new = a - (f(a) * (b - a)) / denom
        error = abs(c_new - c)
        c = c_new

        history.append([i, a, b, f(a), f(b), c, f(c), error])

    return c, history, f

def calcular_regula_falsi_y_generar_pdf(f_str, a, b, tol, max_iter, xi, xf):
    try:
        raiz, history, f = regula_falsi(f_str, a, b, tol, max_iter)

        fn_str = f_str

        params = {
            'a': a,
            'b': b,
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
            'final_error': last[7] if last[7] is not None else 0.0
        }

        # Tabla de iteraciones y formateo a 6 decimales
        df = pd.DataFrame(history, columns=["Iteración", "a", "b", "f(a)", "f(b)", "c", "f(c)", "Error"])
        for col in df.columns:
            if col != "Iteración":
                df[col] = df[col].apply(lambda x: f"{x:.6f}" if pd.notnull(x) else "")

        tables = [df.iloc[i:i + 12] for i in range(0, len(df), 12)]

        graphs = []

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        df_err = df[df["Error"] != ""]
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

        os.makedirs("static", exist_ok=True)
        output_path = os.path.join("static", f"reporte_regulafalsi.pdf")

        generate_pdf_report(
            output_filename=output_path,
            method_name="Regula Falsi",
            fn_str=fn_str,
            params=params,
            results=results,
            tables=tables,
            graphs=graphs
        )

        return output_path

    except Exception as e:
        print(f"Error al generar PDF de Regula Falsi: {e}")
        raise
