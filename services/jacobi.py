import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import os

def jacobi(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Implementación del método de Jacobi

    Args:
        A: Matriz de coeficientes
        b: Vector de términos independientes
        x0: Vector inicial (opcional)
        tol: Tolerancia para el criterio de parada
        max_iter: Número máximo de iteraciones

    Returns:
        tuple: (solución, historial de iteraciones)
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float).copy()
    history = []

    for k in range(1, max_iter + 1):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        error = np.linalg.norm(x_new - x, np.inf)
        history.append((k, x_new.copy(), error))
        if error < tol:
            break
        x = x_new

    return x_new, history

def generate_pdf_report(output_filename, method_name, fn_str, params, results, tables, graphs):
    """
    Generador de PDF profesional (mismo estilo que bisección y Gauss-Seidel)
    """
    with PdfPages(output_filename) as pdf:
        # Configuración general
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'figure.titlesize': 14,
            'figure.titleweight': 'bold'
        })

        # ----------------------------
        # PORTADA DEL REPORTE
        # ----------------------------
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('MathXpert - Soluciones Numéricas', fontsize=16, fontweight='bold', y=0.95)

        # Diseño de la portada
        gs = GridSpec(3, 1, height_ratios=[0.5, 2, 0.5], hspace=0.5)

        # Encabezado
        ax0 = fig.add_subplot(gs[0])
        ax0.axis('off')
        ax0.text(0.5, 0.5, 'Reporte de Análisis Numérico', 
                fontsize=14, ha='center', va='center', fontweight='bold')

        # Contenido principal
        ax1 = fig.add_subplot(gs[1])
        ax1.axis('off')

        # Título del método
        ax1.text(0.5, 0.9, f'Método de {method_name}', 
                fontsize=18, ha='center', va='center', fontweight='bold', color='#2c3e50')

        # Línea decorativa
        ax1.axhline(y=0.85, xmin=0.1, xmax=0.9, color='#3498db', linewidth=2)

        # Resumen del análisis
        summary_text = (
            f"Sistema de ecuaciones:\n{fn_str}\n\n"
            f"Parámetros del método:\n"
            f"- Tamaño de la matriz: {params.get('tamaño_matriz', 'N/A')}\n"
            f"- Tolerancia: {params.get('tolerancia', 'N/A')}\n"
            f"- Iteraciones máximas: {params.get('max_iter', 'N/A')}\n\n"
            f"Resultados obtenidos:\n"
            f"- Iteraciones realizadas: {results.get('iteraciones', 'N/A')}\n"
            f"- Error final: {results.get('error_final', 'N/A')}\n"
            f"- Solución: {results.get('solucion', 'N/A')}\n"
        )

        ax1.text(0.1, 0.6, summary_text, fontsize=11, 
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1, pad=1))

        # Pie de página
        ax2 = fig.add_subplot(gs[2])
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'MathXpert © 2023 - Todos los derechos reservados', 
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
                if col != "Iteración":
                    formatted_table[col] = formatted_table[col].apply(
                        lambda x: f"{float(x):.6f}" if pd.notnull(x) else ""
                    )

            table_title = f"Tabla de iteraciones - Método de {method_name}"
            if len(tables) > 1:
                table_title += f" (Parte {i+1})"

            ax.set_title(table_title, fontsize=12, pad=20, fontweight='bold')

            # Configurar anchos de columna
            num_cols = len(formatted_table.columns)
            col_widths = [0.08] + [0.92/(num_cols-1)]*(num_cols-1)

            tbl = ax.table(
                cellText=formatted_table.values,
                colLabels=formatted_table.columns,
                cellLoc='center',
                loc='center',
                colWidths=col_widths
            )

            # Ajustar estilo
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.2)

            # Coloración alternada
            for k, cell in tbl._cells.items():
                if k[0] == 0:  # Encabezado
                    cell.set_facecolor('#34495e')
                    cell.set_text_props(color='white', weight='bold')
                elif k[0] % 2 == 1:  # Filas impares
                    cell.set_facecolor('#f8f9fa')
                else:  # Filas pares
                    cell.set_facecolor('#e9ecef')

                cell.set_height(0.05)
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

def calcular_jacobi_y_generar_pdf(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Función principal que genera el PDF profesional
    Args:
        A: Matriz de coeficientes
        b: Vector de términos independientes
        x0: Vector inicial (opcional)
        tol: Tolerancia
        max_iter: Iteraciones máximas
    Returns:
        str: Ruta al PDF generado
    """
    try:
        # Calcular solución
        sol, history = jacobi(A, b, x0, tol, max_iter)

        # Preparar datos para el PDF
        method_name = "Jacobi"

        # Representación del sistema
        fn_str = ""
        for i in range(len(A)):
            equation = " + ".join([f"{A[i][j]:.2f}x{j+1}" for j in range(len(A[i]))])
            fn_str += f"{equation} = {b[i]:.2f}\n"

        # Parámetros del método
        params = {
            'tamaño_matriz': f"{len(A)}×{len(A)}",
            'tolerancia': f"{tol:.1e}",
            'max_iter': max_iter
        }

        # Resultados obtenidos
        final_error = history[-1][2] if history else 0
        results = {
            'iteraciones': len(history),
            'error_final': f"{final_error:.6f}",
            'solucion': "\n".join([f"x[{i+1}] = {val:.6f}" for i, val in enumerate(sol)])
        }

        # Preparar tabla de iteraciones
        iter_data = []
        for k, xv, error in history:
            row = [k] + list(xv) + [error]
            iter_data.append(row)

        columns = ["Iteración"] + [f"x[{i+1}]" for i in range(len(sol))] + ["Error"]
        df = pd.DataFrame(iter_data, columns=columns)

        # Dividir la tabla si es muy larga
        tables = []
        rows_per_page = 12
        for i in range(0, len(df), rows_per_page):
            tables.append(df.iloc[i:i + rows_per_page])

        # Crear gráficos
        graphs = []

        # Gráfico 1: Convergencia de variables
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        iterations = [h[0] for h in history]
        for var_idx in range(len(sol)):
            var_values = [h[1][var_idx] for h in history]
            ax1.plot(iterations, var_values, 'o-', markersize=4, label=f'x[{var_idx+1}]')

        ax1.set_xlabel("Iteración", fontweight='bold')
        ax1.set_ylabel("Valor", fontweight='bold')
        ax1.set_title("Convergencia de las variables", pad=20)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()
        graphs.append((fig1, "Convergencia de las variables por iteración"))

        # Gráfico 2: Evolución del error
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        errors = [h[2] for h in history]
        ax2.plot(iterations, errors, 'o-', color='red', markersize=4)
        ax2.set_xlabel("Iteración", fontweight='bold')
        ax2.set_ylabel("Error (norma infinito)", fontweight='bold')
        ax2.set_title("Evolución del error", pad=20)
        ax2.grid(True, linestyle='--', alpha=0.5)
        graphs.append((fig2, "Evolución del error absoluto"))

        # Generar el PDF
        os.makedirs('static', exist_ok=True)
        output_path = os.path.join('static', 'reporte_jacobi.pdf')

        generate_pdf_report(
            output_filename=output_path,
            method_name=method_name,
            fn_str=fn_str,
            params=params,
            results=results,
            tables=tables,
            graphs=graphs
        )

        return output_path

    except Exception as e:
        print(f"Error al generar PDF: {str(e)}")
        raise

# Ejemplo de uso directo
if __name__ == "__main__":
    # Sistema de prueba
    A = [[10, -1, 2], [-1, 11, -1], [2, -1, 10]]
    b = [6, 25, -11]

    # Generar PDF
    pdf_path = calcular_jacobi_y_generar_pdf(A, b)
    print(f"PDF generado en: {pdf_path}")