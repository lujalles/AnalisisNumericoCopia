import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import symbols, sympify, lambdify
from datetime import datetime
import os

def secante(f_str, x0, x1, tol, max_iter):
    """
    Implementación del método de la Secante
    Args:
        f_str: Función como string
        x0, x1: Valores iniciales
        tol: Tolerancia
        max_iter: Iteraciones máximas
    Returns:
        tuple: (raíz, historial de iteraciones, función evaluada)
    """
    x = symbols('x')
    fn = sympify(f_str)
    f = lambdify(x, fn, modules=['numpy'])

    history = []
    errores = []

    # Primera iteración
    f_x0 = f(x0)
    f_x1 = f(x1)
    x2 = x1 - (f_x1 * (x1 - x0)) / (f_x1 - f_x0)
    error = abs(x2 - x1)

    history.append([0, x0, x1, x2, f_x1, "--"])
    errores.append(error)

    # Iteraciones principales
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
        errores.append(error)

    return x2, history, f

def calcular_secante_y_generar_pdf(f_str, x0, x1, tol, max_iter, xi, xf):
    """
    Función principal que genera el PDF profesional para el método de la Secante
    Args:
        f_str: Función como string
        x0, x1: Valores iniciales
        tol: Tolerancia
        max_iter: Iteraciones máximas
        xi, xf: Rango para gráficos
    Returns:
        str: Ruta al archivo PDF generado
    """
    try:
        # Calcular solución
        raiz, history, f = secante(f_str, x0, x1, tol, max_iter)

        # Preparar datos para el PDF
        method_name = "Secante"

        # Parámetros del método
        params = {
            'x0_inicial': x0,
            'x1_inicial': x1,
            'tolerancia': f"{tol:.1e}",
            'max_iter': max_iter,
            'x_min': xi,
            'x_max': xf
        }

        # Resultados obtenidos
        final_error = history[-1][5] if history[-1][5] != "--" else 0.0
        results = {
            'iteraciones': len(history) - 1,
            'error_final': f"{final_error:.6f}",
            'raiz': f"{raiz:.8f}",
            'f_raiz': f"{f(raiz):.2e}"
        }

        # Preparar tabla de iteraciones
        df = pd.DataFrame(
            history,
            columns=["Iteración", "x0", "x1", "x2", "f(x1)", "Error"]
        )

        # Formatear números para mejor visualización
        for col in df.columns:
            if col != "Iteración":
                df[col] = df[col].apply(lambda x: f"{float(x):.6f}" if x != "--" else "--")

        # Dividir la tabla si es muy larga
        tables = []
        rows_per_page = 12
        for i in range(0, len(df), rows_per_page):
            tables.append(df.iloc[i:i + rows_per_page])

        # Crear gráficos
        graphs = []

        # Gráfico 1: Evolución del error
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        df_err = df[df["Error"] != "--"]
        df_err["Iteración"] = df_err["Iteración"].astype(int)
        df_err["Error"] = df_err["Error"].astype(float)
        ax1.plot(df_err["Iteración"], df_err["Error"], 'o-', color='#e74c3c', markersize=5, linewidth=1.5)
        ax1.set_xlabel("Iteración", fontweight='bold')
        ax1.set_ylabel("Error absoluto", fontweight='bold')
        ax1.set_title("Convergencia del error", pad=20, fontweight='bold')
        ax1.grid(True)
        ax1.set_facecolor('#f8f9fa')
        graphs.append((fig1, "Evolución del error por iteración"))

        # Gráfico 2: Función y raíz encontrada
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        x_vals = np.linspace(xi, xf, 400)
        y_vals = f(x_vals)
        ax2.plot(x_vals, y_vals, label=f"f(x) = {f_str}", color='#3498db', linewidth=2)
        ax2.axhline(0, color='#2c3e50', linestyle='--', label='y = 0', alpha=0.7)
        ax2.plot(raiz, f(raiz), 'ro', markersize=8, label=f"Raíz ≈ {raiz:.6f}")
        ax2.set_xlabel("x", fontweight='bold')
        ax2.set_ylabel("f(x)", fontweight='bold')
        ax2.set_title("Gráfico de la función", pad=20, fontweight='bold')
        ax2.grid(True)
        ax2.legend()
        ax2.set_facecolor('#f8f9fa')
        graphs.append((fig2, "Representación gráfica de la función"))

        # Gráfico 3: Convergencia de x2
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        x2_values = df["x2"].replace("--", "0").astype(float)
        ax3.plot(df["Iteración"], x2_values, 'o-', color='#2ecc71', markersize=5, linewidth=1.5)
        ax3.axhline(raiz, color='#e74c3c', linestyle='--', alpha=0.7, label=f'Raíz ({raiz:.6f})')
        ax3.set_xlabel("Iteración", fontweight='bold')
        ax3.set_ylabel("Valor de x2", fontweight='bold')
        ax3.set_title("Convergencia del punto x2", pad=20, fontweight='bold')
        ax3.grid(True)
        ax3.legend()
        ax3.set_facecolor('#f8f9fa')
        graphs.append((fig3, "Convergencia del punto x2 por iteración"))

        # Generar el PDF con nombre único
        os.makedirs('static', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"secante_report_{timestamp}.pdf"
        output_path = os.path.join('static', pdf_filename)

        generate_pdf_report(
            output_filename=output_path,
            method_name=method_name,
            fn_str=f"f(x) = {f_str}",
            params=params,
            results=results,
            tables=tables,
            graphs=graphs
        )

        return output_path

    except Exception as e:
        print(f"Error al generar PDF de Secante: {str(e)}")
        raise