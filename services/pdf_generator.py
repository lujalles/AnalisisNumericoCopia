import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import pandas as pd

def generate_pdf_report(output_filename, method_name, fn_str, params, results, tables, graphs):
    """
    Genera un PDF profesional para métodos numéricos

    Args:
        output_filename (str): Nombre del archivo PDF a generar (ej. 'reporte_biseccion.pdf')
        method_name (str): Nombre del método (ej. "Bisección")
        fn_str (str): Representación string de la función
        params (dict): Parámetros del método
        results (dict): Resultados principales
        tables (list): Lista de DataFrames con tablas
        graphs (list): Lista de tuplas (fig, title) con gráficos
    """
    with PdfPages(output_filename) as pdf:
        # Configuración general de estilos
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

        # Diseño de la portada con GridSpec
        gs = GridSpec(3, 1, height_ratios=[0.5, 2, 0.5], hspace=0.5)

        # Logo/Encabezado
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
            f"Función analizada: $f(x) = {fn_str}$\n\n"
            f"Parámetros del método:\n"
            f"- Intervalo inicial: $a = {params.get('a', 'N/A')}$, $b = {params.get('b', 'N/A')}$\n"
            f"- Error tolerado: ${params.get('es', 'N/A')}$\n"
            f"- Iteraciones máximas: {params.get('imax', 'N/A')}\n"
            f"- Rango de gráfico: $x \in [{params.get('xi', 'N/A')}, {params.get('xf', 'N/A')}]$\n\n"
            f"Resultados obtenidos:\n"
            f"- Raíz encontrada: ${results.get('root', 'N/A'):.6f}$\n"
            f"- Valor de la función: ${results.get('f_root', 'N/A'):.6f}$\n"
            f"- Iteraciones realizadas: {results.get('iterations', 'N/A')}\n"
            f"- Error final: ${results.get('final_error', 'N/A'):.6f}$\n"
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

            # Título de la tabla
            table_title = f"Tabla de iteraciones - Método de {method_name}"
            if len(tables) > 1:
                table_title += f" (Parte {i+1})"

            ax.set_title(table_title, fontsize=12, pad=20, fontweight='bold')

            # Configuración de la tabla
            col_widths = [0.08] + [0.12]*(len(table.columns)-1)  # Ajuste de ancho de columnas

            tbl = ax.table(
                cellText=table.values, 
                colLabels=table.columns, 
                cellLoc='center', 
                loc='center',
                colWidths=col_widths
            )

            # Estilo de la tabla
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.3)

            # Colores alternados para mejor legibilidad
            for k, cell in tbl._cells.items():
                if k[0] == 0:  # Encabezado
                    cell.set_facecolor('#34495e')
                    cell.set_text_props(color='white', weight='bold')
                elif k[0] % 2 == 1:  # Filas impares
                    cell.set_facecolor('#f8f9fa')
                else:  # Filas pares
                    cell.set_facecolor('#e9ecef')

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # ----------------------------
        # GRÁFICOS
        # ----------------------------
        for fig, title in graphs:
            # Añadir título principal al gráfico
            fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)

            # Ajustar diseño y guardar
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()