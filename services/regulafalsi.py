import numpy as np
import matplotlib
matplotlib.use('Agg')  # Configuración importante para Flask
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import os
from sympy import symbols, sympify, lambdify
from datetime import datetime

def regula_falsi(f_str, a, b, tol, max_iter, xi, xf):
    """
    Método de Regula Falsi con generación de PDF profesional

    Args:
        f_str: Función como string
        a: Límite inferior del intervalo
        b: Límite superior del intervalo
        tol: Tolerancia
        max_iter: Iteraciones máximas
        xi, xf: Rango para gráficos

    Returns:
        dict: Resultados y ruta del PDF
    """
    try:
        plt.close('all')  # Limpiar figuras previas

        # Preparar función simbólica
        x = symbols('x')
        fn = sympify(f_str)
        f = lambdify(x, fn, modules=['numpy'])

        # Verificar cambio de signo
        if f(a) * f(b) > 0:
            raise ValueError("No hay cambio de signo en el intervalo dado. El método no puede aplicarse.")

        # Inicialización de variables
        iteraciones = []
        errores = []
        puntos_c = []

        # Primera iteración
        denom = f(b) - f(a)
        if denom == 0:
            raise ZeroDivisionError("División por cero en la fórmula del método Regula Falsi.")

        c = a - (f(a) * (b - a)) / denom
        error = abs(b - a)

        iteraciones.append([0, a, b, f(a), f(b), c, f(c), "--"])
        puntos_c.append(c)
        errores.append(error)

        # Iteraciones principales
        i = 0
        while error > tol and i < max_iter:
            i += 1

            if f(a) * f(c) < 0:
                b = c
            else:
                a = c

            denom = f(b) - f(a)
            if denom == 0:
                raise ZeroDivisionError("División por cero en la fórmula del método Regula Falsi.")

            c_new = a - (f(a) * (b - a)) / denom
            error = abs(c_new - c)
            c = c_new

            iteraciones.append([i, a, b, f(a), f(b), c, f(c), error])
            puntos_c.append(c)
            errores.append(error)

        raiz = c

        # Preparar datos para el PDF
        method_name = "Regula Falsi"

        # Parámetros del método
        params = {
            'a_inicial': iteraciones[0][1],
            'b_inicial': iteraciones[0][2],
            'tolerancia': tol,
            'max_iter': max_iter,
            'x_min': xi,
            'x_max': xf
        }

        # Resultados obtenidos
        results = {
            'iteraciones': i,
            'error_final': errores[-1] if errores else 0,
            'raiz': raiz,
            'f_raiz': f(raiz)
        }

        # Preparar tabla de iteraciones
        df_iter = pd.DataFrame(
            iteraciones,
            columns=["Iteración", "a", "b", "f(a)", "f(b)", "c", "f(c)", "Error"]
        )

        # Dividir la tabla si es muy larga
        tables = []
        rows_per_page = 12
        for i in range(0, len(df_iter), rows_per_page):
            tables.append(df_iter.iloc[i:i + rows_per_page])

        # Crear gráficos
        graphs = []

        # Gráfico 1: Evolución del error
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(range(1, len(errores)), errores[1:], 'o-', color='#e74c3c', markersize=5, linewidth=1.5)
        ax1.set_xlabel("Iteración", fontweight='bold')
        ax1.set_ylabel("Error absoluto", fontweight='bold')
        ax1.set_title("Convergencia del error", pad=20, fontweight='bold')
        ax1.grid(True)
        ax1.set_facecolor('#f8f9fa')
        fig1.tight_layout()
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
        fig2.tight_layout()
        graphs.append((fig2, "Representación gráfica de la función"))

        # Gráfico 3: Convergencia del punto c
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(range(len(puntos_c)), puntos_c, 'o-', color='#2ecc71', markersize=5, linewidth=1.5)
        ax3.axhline(raiz, color='#e74c3c', linestyle='--', alpha=0.7, label=f'Raíz ({raiz:.6f})')
        ax3.set_xlabel("Iteración", fontweight='bold')
        ax3.set_ylabel("Valor de c", fontweight='bold')
        ax3.set_title("Convergencia del punto c", pad=20, fontweight='bold')
        ax3.grid(True)
        ax3.legend()
        ax3.set_facecolor('#f8f9fa')
        fig3.tight_layout()
        graphs.append((fig3, "Convergencia del punto c por iteración"))

        # Generar el PDF con nombre único en el directorio static
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        os.makedirs(static_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"regula_falsi_report_{timestamp}.pdf"
        output_path = os.path.join(static_dir, pdf_filename)

        generate_pdf_report(
            output_filename=output_path,
            method_name=method_name,
            fn_str=f"f(x) = {f_str}",
            params=params,
            results=results,
            tables=tables,
            graphs=graphs
        )

        return {
            "raiz": raiz,
            "iteraciones": df_iter,
            "pdf": output_path,
            "funcion": fn
        }

    except Exception as e:
        plt.close('all')  # Asegurar que todas las figuras se cierren
        print(f"Error en Regula Falsi: {str(e)}")
        raise

def generate_pdf_report(output_filename, method_name, fn_str, params, results, tables, graphs):
    """
    Generador de PDF profesional unificado para todos los métodos

    Args:
        output_filename: Ruta completa del archivo PDF de salida
        method_name: Nombre del método numérico
        fn_str: Representación textual del problema
        params: Diccionario con parámetros del método
        results: Diccionario con resultados obtenidos
        tables: Lista de DataFrames con tablas de iteraciones
        graphs: Lista de tuplas (figura, título) con gráficos
    """
    with PdfPages(output_filename) as pdf:
        # Configuración de estilo profesional
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'figure.titlesize': 14,
            'figure.titleweight': 'bold',
            'figure.facecolor': 'white',
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.5
        })

        # ----------------------------
        # PORTADA DEL REPORTE
        # ----------------------------
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('MathXpert - Soluciones Numéricas', fontsize=16, fontweight='bold', y=0.95)

        # Diseño de la portada
        gs = GridSpec(3, 1, height_ratios=[0.3, 2, 0.2], hspace=0.3)

        # Encabezado
        ax0 = fig.add_subplot(gs[0])
        ax0.axis('off')
        ax0.text(0.5, 0.7, 'Reporte de Análisis Numérico', 
                fontsize=14, ha='center', va='center', fontweight='bold')

        # Contenido principal
        ax1 = fig.add_subplot(gs[1])
        ax1.axis('off')

        # Título del método
        ax1.text(0.5, 0.85, f'Método de {method_name}', 
                fontsize=18, ha='center', va='center', fontweight='bold', color='#2c3e50')

        # Línea decorativa
        ax1.axhline(y=0.82, xmin=0.1, xmax=0.9, color='#3498db', linewidth=2, alpha=0.7)

        # Resumen del análisis
        summary_text = (
            f"Función analizada:\n{fn_str}\n\n"
            f"Parámetros del método:\n"
            f"- Intervalo inicial: a = {params.get('a_inicial', 'N/A'):.6f}, b = {params.get('b_inicial', 'N/A'):.6f}\n"
            f"- Tolerancia: {params.get('tolerancia', 'N/A'):.1e}\n"
            f"- Iteraciones máximas: {params.get('max_iter', 'N/A')}\n"
            f"- Rango de análisis: x ∈ [{params.get('x_min', 'N/A'):.2f}, {params.get('x_max', 'N/A'):.2f}]\n\n"
            f"Resultados obtenidos:\n"
            f"- Iteraciones realizadas: {results.get('iteraciones', 'N/A')}\n"
            f"- Error final: {results.get('error_final', 'N/A'):.2e}\n"
            f"- Raíz aproximada: {results.get('raiz', 'N/A'):.8f}\n"
            f"- f(raíz): {results.get('f_raiz', 'N/A'):.2e}\n"
        )

        ax1.text(0.1, 0.5, summary_text, fontsize=11, 
                verticalalignment='top', linespacing=1.5,
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', 
                          edgecolor='#dee2e6', pad=1.0))

        # Pie de página
        ax2 = fig.add_subplot(gs[2])
        ax2.axis('off')
        ax2.text(0.5, 0.3, 'MathXpert © 2023 - Todos los derechos reservados', 
                fontsize=8, ha='center', va='center', alpha=0.7)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # ----------------------------
        # TABLAS DE ITERACIONES
        # ----------------------------
        for i, table in enumerate(tables):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')

            # Formatear números para mejor visualización
            formatted_table = table.copy()
            for col in formatted_table.columns:
                if pd.api.types.is_numeric_dtype(formatted_table[col]):
                    formatted_table[col] = formatted_table[col].apply(
                        lambda x: f"{float(x):.6f}" if pd.notnull(x) and col != "Iteración" else x
                    )

            table_title = f"Tabla de iteraciones - Método de {method_name}"
            if len(tables) > 1:
                table_title += f" (Parte {i+1})"

            ax.set_title(table_title, fontsize=12, pad=20, fontweight='bold')

            # Configurar anchos de columna
            num_cols = len(formatted_table.columns)
            col_widths = [0.1] + [0.9/(num_cols-1)]*(num_cols-1)

            tbl = ax.table(
                cellText=formatted_table.values,
                colLabels=formatted_table.columns,
                cellLoc='center',
                loc='center',
                colWidths=col_widths
            )

            # Ajustar estilo
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.3)

            # Coloración alternada
            for k, cell in tbl._cells.items():
                if k[0] == 0:  # Encabezado
                    cell.set_facecolor('#34495e')
                    cell.set_text_props(color='white', weight='bold')
                elif k[0] % 2 == 1:  # Filas impares
                    cell.set_facecolor('#f8f9fa')
                else:  # Filas pares
                    cell.set_facecolor('#e9ecef')

                cell.set_height(0.06)
                cell.PAD = 0.05

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # ----------------------------
        # GRÁFICOS
        # ----------------------------
        for fig, title in graphs:
            fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()