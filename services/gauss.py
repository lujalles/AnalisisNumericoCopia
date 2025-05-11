
import numpy as np
from fpdf import FPDF # Se recomienda usar fpdf2: pip install fpdf2
import os
import io # Para leer la string como si fuera un archivo
import sys # Para capturar la salida de impresión si mostramos pasos
import re

def gauss_elimination(matrix_string, use_pivoting=True, show_steps=True, nombre_pdf="reporte_gauss.pdf"):
    """
    Resuelve un sistema de ecuaciones lineales utilizando el método de Eliminación Gaussiana.

    Args:
        matrix_string (str): Una cadena de texto representando la matriz aumentada [A|b].
                             Los números en cada fila deben estar separados por espacios o comas,
                             y las filas separadas por saltos de línea.
                             Ejemplo: "1 2 3 10\n4 5 6 32\n7 8 9 54"
        use_pivoting (bool): Si es True, aplica pivoteo parcial. Por defecto es True.
        show_steps (bool): Si es True, incluye los pasos detallados de la eliminación en el PDF. Por defecto es True.
        nombre_pdf (str): El nombre del archivo PDF a generar. Por defecto es "reporte_gauss.pdf".

    Returns:
        dict: Un diccionario que contiene:
            - "solution" (numpy.ndarray or None): El vector solución [x1, x2, ...], o None si hay error.
            - "steps" (str or None): Una cadena de texto con los pasos detallados si show_steps fue True y no hubo error.
            - "pdf" (str or None): El nombre del archivo PDF generado si tuvo éxito, o None.
            - "error" (str or None): Mensaje de error si ocurrió uno.
    """

    steps_output = io.StringIO() # Usamos StringIO para capturar la salida de los pasos
    original_stdout = sys.stdout # Guardar la salida estándar original

    try:
        sys.stdout = steps_output # Redirigir la salida estándar a StringIO

        # --- 1. Parsear la matriz de la cadena de texto ---
        # Convertir la cadena de entrada en una lista de listas de flotantes
        matrix_list = []
        rows = matrix_string.strip().split('\n') # Separar filas por salto de línea
        if not rows:
             raise ValueError("La cadena de la matriz está vacía.")

        num_cols = -1
        for i, row in enumerate(rows):
            # Intentar separar números por espacios o comas
            elements = re.split(r'[,\s]+', row.strip())
            # Filtrar elementos vacíos que puedan resultar de múltiples espacios o comas
            elements = [e for e in elements if e]
            if not elements:
                 raise ValueError(f"La fila {i+1} está vacía o no contiene números.")

            try:
                # Convertir elementos a flotantes
                row_data = [float(e) for e in elements]
            except ValueError:
                raise ValueError(f"La fila {i+1} contiene valores no numéricos.")

            if num_cols == -1:
                num_cols = len(row_data)
                if num_cols < 2:
                     raise ValueError("La matriz debe tener al menos 2 columnas (matriz A + vector b).")
            elif len(row_data) != num_cols:
                raise ValueError(f"La fila {i+1} tiene un número inconsistente de columnas ({len(row_data)} en lugar de {num_cols}).")

            matrix_list.append(row_data)

        # Convertir la lista de listas a un array de NumPy para facilitar las operaciones matriciales
        augmented_matrix = np.array(matrix_list, dtype=float)

        # Verificar las dimensiones de la matriz
        num_rows, current_num_cols = augmented_matrix.shape
        if current_num_cols != num_rows + 1:
            raise ValueError(f"La matriz aumentada debe ser cuadrada en su parte A, más una columna para b. Se esperaba una matriz de {num_rows}x{num_rows+1}, pero se obtuvo {num_rows}x{current_num_cols}.")

        n = num_rows # Dimensión del sistema (número de ecuaciones/variables)

        # Guardar la matriz original para el reporte
        original_matrix = np.copy(augmented_matrix)

        if show_steps:
            print("--- Método de Eliminación Gaussiana ---")
            print("\nMatriz aumentada inicial:")
            print(original_matrix)
            print("-" * 30)

        # --- 2. Fase de Eliminación hacia adelante ---
        for i in range(n): # Iterar sobre cada fila para hacer ceros debajo del pivote

            # --- Pivoteo Parcial (si está activado) ---
            if use_pivoting:
                # Encontrar la fila con el elemento más grande en valor absoluto en la columna actual (desde la fila i hacia abajo)
                pivot_row = i + np.argmax(np.abs(augmented_matrix[i:, i]))
                if pivot_row != i:
                    # Intercambiar la fila actual con la fila del pivote
                    augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]
                    if show_steps:
                        print(f"\nIntercambio de filas R{i+1} <-> R{pivot_row+1}")
                        print(augmented_matrix)
                        print("-" * 30)

            # --- Verificar si el pivote es cero ---
            # Si el elemento diagonal (pivote) es cero después del pivoteo (o sin él),
            # la matriz es singular y el sistema no tiene solución única.
            pivot_element = augmented_matrix[i, i]
            if abs(pivot_element) < 1e-12: # Usar una pequeña tolerancia para comparar con cero
                raise ValueError(f"El elemento pivote en la posición ({i+1}, {i+1}) es cero o muy cercano a cero. La matriz es singular o el sistema no tiene solución única.")

            # --- Hacer ceros debajo del pivote ---
            for j in range(i + 1, n): # Iterar sobre las filas debajo de la fila actual
                # Calcular el factor por el cual multiplicar la fila pivote
                factor = augmented_matrix[j, i] / pivot_element

                # Restar el múltiplo de la fila pivote a la fila actual para hacer cero el elemento en la columna i
                augmented_matrix[j, i:] = augmented_matrix[j, i:] - factor * augmented_matrix[i, i:]

                if show_steps:
                    print(f"\nOperación: R{j+1} = R{j+1} - ({factor:.4f}) * R{i+1}")
                    print(augmented_matrix)
                    print("-" * 30)

        if show_steps:
            print("\nMatriz después de la eliminación hacia adelante (forma triangular superior):")
            print(augmented_matrix)
            print("-" * 30)

        # --- 3. Fase de Sustitución hacia atrás ---
        # Inicializar el vector solución
        solution = np.zeros(n)

        # Iterar desde la última fila hacia arriba
        for i in range(n - 1, -1, -1):
            # El elemento diagonal es el coeficiente de la variable actual (xi)
            diagonal_element = augmented_matrix[i, i]

            # Sumar los términos conocidos (elementos de la derecha menos los productos de coeficientes por variables ya resueltas)
            # augmented_matrix[i, -1] es el elemento b_i
            # augmented_matrix[i, i+1:n] son los coeficientes de las variables x_{i+1} a x_n
            # solution[i+1:n] son los valores ya calculados de x_{i+1} a x_n
            sum_of_terms = np.sum(augmented_matrix[i, i+1:n] * solution[i+1:n])

            # Calcular el valor de la variable actual (xi)
            # xi = (b_i - sumatoria(a_ij * xj para j > i)) / a_ii
            solution[i] = (augmented_matrix[i, -1] - sum_of_terms) / diagonal_element

            if show_steps:
                print(f"\nSustitución hacia atrás para x{i+1}:")
                print(f"x{i+1} = ({augmented_matrix[i, -1]:.4f} - {sum_of_terms:.4f}) / {diagonal_element:.4f}")
                print(f"x{i+1} = {solution[i]:.6f}")
                print("-" * 30)


        if show_steps:
            print("\nVector solución X:")
            print(solution)
            print("-" * 30)

        # --- 4. Generación del Reporte PDF ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Reporte del Método de Eliminación Gaussiana", ln=True, align='C')
        pdf.ln(5)

        # Resumen
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Resumen:", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 7, f"Dimensión del sistema: {n}x{n}", ln=True)
        pdf.cell(0, 7, f"Uso de pivoteo parcial: {'Sí' if use_pivoting else 'No'}", ln=True)
        pdf.cell(0, 7, f"Mostrar pasos detallados: {'Sí' if show_steps else 'No'}", ln=True)
        pdf.ln(5)

        # Matriz Original
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Matriz Aumentada Inicial:", ln=True)
        pdf.set_font("Arial", '', 10)
        # Añadir la matriz original al PDF
        # Calculamos el ancho de las columnas basado en el número de columnas
        col_width = pdf.w / (current_num_cols + 2) # +2 para margen y separación
        for row in original_matrix:
            for j, element in enumerate(row):
                # Añadir una línea vertical antes de la última columna (vector b)
                border = 1
                if j == n - 1: # Si es la penúltima columna (última de A)
                     border = 'LR' # Borde izquierdo y derecho
                elif j == n: # Si es la última columna (vector b)
                     border = 'R' # Solo borde derecho
                else:
                     border = 'L' # Solo borde izquierdo

                pdf.cell(col_width, 8, f"{element:.4f}", border=border, align='C')
            pdf.ln() # Nueva línea al final de cada fila
        pdf.ln(5)


        # Pasos detallados (si se solicitaron)
        if show_steps:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Pasos de la Eliminación Gaussiana:", ln=True)
            pdf.set_font("Courier", '', 9) # Usar una fuente monoespaciada para los pasos
            steps_text = steps_output.getvalue() # Obtener el texto capturado
            # Dividir el texto en líneas y añadirlo al PDF, manejando saltos de página
            lines = steps_text.split('\n')
            for line in lines:
                 # Aproximación: si la línea es muy larga, podría necesitar envolverse,
                 # pero para la salida típica de numpy y mensajes, esto suele ser suficiente.
                 # Verificar si hay espacio para la línea actual
                 if pdf.get_y() + 5 > 280: # 280 es aprox el final de la página
                      pdf.add_page()
                      pdf.set_font("Courier", '', 9) # Re-establecer fuente
                 pdf.cell(0, 5, line, ln=True) # Añadir línea al PDF
            pdf.ln(5) # Espacio después de los pasos


        # Matriz Triangular Superior Final
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Matriz en Forma Triangular Superior:", ln=True)
        pdf.set_font("Arial", '', 10)
        # Añadir la matriz triangular al PDF (misma lógica de borde que la original)
        for row in augmented_matrix:
            for j, element in enumerate(row):
                border = 1
                if j == n - 1:
                     border = 'LR'
                elif j == n:
                     border = 'R'
                else:
                     border = 'L'
                pdf.cell(col_width, 8, f"{element:.4f}", border=border, align='C')
            pdf.ln()
        pdf.ln(5)


        # Vector Solución
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Vector Solución X:", ln=True)
        pdf.set_font("Arial", '', 11)
        # Formatear el vector solución como una cadena
        solution_str = "[" + ", ".join([f"{val:.6f}" for val in solution]) + "]"
        pdf.cell(0, 8, solution_str, ln=True)
        pdf.ln(5)


        # Guardar el PDF
        pdf.output(nombre_pdf)

        # Devolver los resultados
        return {
            "solution": solution,
            "steps": steps_output.getvalue() if show_steps else None,
            "pdf": nombre_pdf,
            "error": None
        }

    except ValueError as ve:
        # Capturar errores de validación o singularidad
        if show_steps:
            print(f"\nError: {ve}") # Imprimir el error en el output capturado también
        return {
            "solution": None,
            "steps": steps_output.getvalue() if show_steps else None,
            "pdf": None,
            "error": str(ve)
        }
    except Exception as e:
        # Capturar cualquier otro error inesperado
        if show_steps:
             print(f"\nError inesperado: {e}") # Imprimir el error en el output capturado
        return {
            "solution": None,
            "steps": steps_output.getvalue() if show_steps else None,
            "pdf": None,
            "error": f"Ocurrió un error inesperado: {e}"
        }
    finally:
        # Restaurar la salida estándar original
        sys.stdout = original_stdout
        # Asegurarse de que el objeto StringIO se cierre
        steps_output.close()


