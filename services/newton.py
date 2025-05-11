import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importamos los módulos necesarios de sympy, incluyendo SympifyError para manejo de errores de parseo
from sympy import symbols, sympify, lambdify, diff, SympifyError
# Importamos FPDF para la generación del reporte PDF. Se recomienda usar fpdf2.
from fpdf import FPDF
import os
import re

def newton_raphson(func_str_input, df_str_input, x0, tol, max_iter, x_min, x_max, nombre_pdf="resultado_newton.pdf"):
    """
    Implementa el método de Newton-Raphson para encontrar la raíz de una función.

    Args:
        func_str_input (str): La función f(x) como una cadena de texto (ej: "x**2 - 4").
        df_str_input (str): La derivada f'(x) como una cadena de texto (ej: "2*x").
        x0 (float): El valor inicial (punto de partida).
        tol (float): La tolerancia del error para el criterio de parada.
        max_iter (int): El número máximo de iteraciones permitidas.
        x_min (float): El límite inferior del rango para graficar.
        x_max (float): El límite superior del rango para graficar.
        nombre_pdf (str): El nombre del archivo PDF a generar (por defecto: "resultado_newton.pdf").

    Returns:
        dict: Un diccionario que contiene:
            - "raiz" (float or None): La raíz aproximada encontrada, o None si no se alcanzó la tolerancia.
            - "iteraciones" (pandas.DataFrame): Una tabla con los detalles de cada iteración.
            - "pdf" (str or None): El nombre del archivo PDF generado si tuvo éxito, o None.
            - "funcion_simbolica" (sympy expression or None): La expresión simbólica de f(x).
            - "derivada_simbolica" (sympy expression or None): La expresión simbólica de f'(x).
            - "error" (str or None): Mensaje de error si ocurrió uno (ej: derivada cero, error de sintaxis).
    """

    # Normalizar función: prepara la string para SymPy (ej: convierte ^ a **, añade * implícitos)
    # Esta función intenta convertir la entrada del usuario a un formato que SymPy pueda entender.
    def normalizar_funcion(expr):
        # Reemplaza el operador de potencia '^' por '**' que usa Python y SymPy
        expr = expr.replace('^', '**')
        # Añade '*' entre un dígito y 'x' (ej: 2x -> 2*x)
        expr = re.sub(r'(?<=\d)(x)', r'*x', expr)
        # Añade '*' entre un dígito y una letra que no sea 'x' (para funciones como 2sin -> 2*sin)
        expr = re.sub(r'(\d)([a-zA-Z_])', r'\1*\2', expr)
        # Elimina espacios en blanco
        expr = expr.replace(' ', '')
        return expr

    # Preparar funciones: Convertir las strings de función y derivada a objetos simbólicos y luego a funciones numéricas
    x = symbols('x')
    f_expr, df_expr = None, None # Inicializar expresiones simbólicas a None
    try:
        # Intentar normalizar y parsear la función ingresada por el usuario
        func_str = normalizar_funcion(func_str_input)
        f_expr = sympify(func_str)

        # Intentar normalizar y parsear la derivada ingresada por el usuario
        df_str = normalizar_funcion(df_str_input)
        df_expr = sympify(df_str)

        # Convertir las expresiones simbólicas a funciones numéricas usando numpy
        # Esto permite evaluar f(x) y f'(x) para valores numéricos de x.
        f = lambdify(x, f_expr, modules=['numpy'])
        df = lambdify(x, df_expr, modules=['numpy'])

    except SympifyError:
        # Capturar errores específicos de SymPy al parsear la string de la función o derivada.
        # Esto ocurre si la sintaxis matemática no es válida para SymPy.
        return {"error": "Error de sintaxis en la función o su derivada. Asegúrate de usar x como variable, ^ para potencias, y funciones como sin(x), cos(x), exp(x), log(x).",
                "raiz": None, "iteraciones": pd.DataFrame(), "pdf": None,
                "funcion_simbolica": None, "derivada_simbolica": None}
    except Exception as e:
        # Capturar cualquier otro error inesperado durante la preparación de funciones.
        return {"error": f"Ocurrió un error al preparar las funciones: {e}",
                "raiz": None, "iteraciones": pd.DataFrame(), "pdf": None,
                "funcion_simbolica": f_expr, "derivada_simbolica": df_expr}

    # --- Método de Newton-Raphson ---
    iteraciones = [] # Lista para almacenar los datos de cada iteración
    try:
        # Convertir el valor inicial a flotante.
        xn = float(x0)
    except ValueError:
        # Capturar error si x0 no es un número válido.
        return {"error": "El valor inicial (x0) debe ser un número válido.",
                "raiz": None, "iteraciones": pd.DataFrame(), "pdf": None,
                "funcion_simbolica": f_expr, "derivada_simbolica": df_expr}

    raiz = None # Variable para almacenar la raíz encontrada
    error_actual = float('inf') # Inicializar el error actual a infinito para asegurar que el bucle comience

    # Bucle principal del método de Newton-Raphson
    for i in range(1, max_iter + 1):
        try:
            # Evaluar la función y su derivada en el punto actual xn
            fx = f(xn)
            dfx = df(xn)
        except Exception as e:
            # Capturar errores si la evaluación de f(x) o f'(x) falla en algún punto durante las iteraciones.
            iteraciones.append([i, xn, f"Error: {e}", f"Error: {e}", "--"])
            df_iter = pd.DataFrame(iteraciones, columns=["Iteración", "x", "f(x)", "f'(x)", "Error"])
            return {"error": f"Error durante la evaluación de funciones en x = {xn} en la iteración {i}: {e}",
                    "raiz": None, "iteraciones": df_iter, "pdf": None,
                    "funcion_simbolica": f_expr, "derivada_simbolica": df_expr}

        # Verificar si la derivada es cercana a cero para evitar división por cero.
        if abs(dfx) < 1e-15: # Usamos una tolerancia pequeña en lugar de comparar directamente con 0.
            iteraciones.append([i, xn, fx, dfx, "--"])
            df_iter = pd.DataFrame(iteraciones, columns=["Iteración", "x", "f(x)", "f'(x)", "Error"])
            return {"error": f"La derivada es cercana a cero ({dfx:.2e}) en x = {xn} en la iteración {i}. El método falla.",
                    "raiz": None, "iteraciones": df_iter, "pdf": None,
                    "funcion_simbolica": f_expr, "derivada_simbolica": df_expr}

        # Calcular la siguiente aproximación de la raíz
        try:
            xn_next = xn - fx / dfx
            # Calcular el error absoluto entre la aproximación actual y la anterior
            error_actual = abs(xn_next - xn)
        except Exception as e:
             # Capturar errores si el cálculo de la siguiente iteración falla.
            iteraciones.append([i, xn, fx, dfx, f"Error: {e}"])
            df_iter = pd.DataFrame(iteraciones, columns=["Iteración", "x", "f(x)", "f'(x)", "Error"])
            return {"error": f"Error durante el cálculo de la siguiente iteración en i={i}: {e}",
                    "raiz": None, "iteraciones": df_iter, "pdf": None,
                    "funcion_simbolica": f_expr, "derivada_simbolica": df_expr}

        # Añadir los resultados de la iteración actual a la lista
        iteraciones.append([i, xn, fx, dfx, error_actual])

        # Criterio de parada: si el error actual es menor que la tolerancia, hemos encontrado una raíz aproximada.
        if error_actual < tol:
            raiz = xn_next # La raíz es la última aproximación calculada
            break # Salir del bucle

        # Actualizar xn para la siguiente iteración
        xn = xn_next

    # Si el bucle terminó porque se alcanzó el número máximo de iteraciones (y no por tolerancia),
    # la última xn_next calculada es la mejor aproximación encontrada.
    if raiz is None and len(iteraciones) > 0:
         # La raíz es el valor de 'x' en la última fila de la tabla de iteraciones.
        raiz = iteraciones[-1][1]

    # Crear un DataFrame de pandas con los resultados de las iteraciones para facilitar su manejo.
    df_iter = pd.DataFrame(iteraciones, columns=["Iteración", "x", "f(x)", "f'(x)", "Error"])

    # --- Generación de Reporte PDF ---
    plot_file = None # Inicializar la ruta del archivo de gráfico a None
    try:
        # Generar datos para el gráfico de la función en el rango especificado.
        x_vals = np.linspace(x_min, x_max, 400)
        y_vals = f(x_vals) # Evaluar la función en los puntos x_vals

        # Crear la figura y los ejes del gráfico
        plt.figure(figsize=(8, 5))
        # Dibujar la función
        plt.plot(x_vals, y_vals, label=f'f(x) = {f_expr}')
        # Dibujar el eje x (y=0) como referencia
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        # Graficar el punto de la raíz encontrada si existe
        if raiz is not None:
            try:
                # Evaluar la función en la raíz encontrada para graficar el punto (raiz, f(raiz))
                plt.plot(raiz, f(raiz), 'ro', label=f'Raíz aprox. ≈ {raiz:.6f}')
            except Exception:
                # Si falla la evaluación de f(raiz) (ej: dominio), simplemente no graficamos el punto.
                pass
        # Configurar título y etiquetas de los ejes
        plt.title('Gráfico de la función y la raíz')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        # Añadir cuadrícula
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # Mostrar leyenda
        plt.legend()

        # Guardar el gráfico en un archivo temporal PNG
        plot_file = "grafico_newton.png"
        plt.savefig(plot_file, bbox_inches='tight') # bbox_inches='tight' ayuda a que no se corten las etiquetas
        plt.close() # Cerrar la figura para liberar memoria

    except Exception as e:
        # Capturar errores durante la generación del gráfico.
        print(f"Advertencia: No se pudo generar el gráfico. Error: {e}")
        plot_file = None # Asegurarse de que plot_file es None si falla la generación

    # Crear el documento PDF
    pdf = FPDF()
    pdf.add_page() # Añadir una nueva página
    pdf.set_font("Arial", 'B', 16) # Configurar fuente para el título principal
    pdf.cell(0, 10, "Reporte del Método de Newton-Raphson", ln=True, align='C') # Añadir título centrado
    pdf.ln(5) # Añadir espacio

    # Sección de Resumen
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Resumen:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 7, f"Función: f(x) = {func_str_input}", ln=True)
    pdf.cell(0, 7, f"Derivada ingresada: f'(x) = {df_str_input}", ln=True) # Mostrar la derivada tal cual se ingresó
    pdf.cell(0, 7, f"Valor inicial x0: {x0}", ln=True)
    pdf.cell(0, 7, f"Tolerancia: {tol}", ln=True)
    pdf.cell(0, 7, f"Máximo de iteraciones permitido: {max_iter}", ln=True)
    pdf.cell(0, 7, f"Iteraciones ejecutadas: {len(iteraciones)}", ln=True)
    # Mostrar la raíz encontrada o un mensaje si no se encontró
    if raiz is not None:
        pdf.cell(0, 7, f"Raíz aproximada encontrada: {raiz:.6f}", ln=True)
    else:
        pdf.cell(0, 7, "Raíz aproximada encontrada: El método no convergió a la tolerancia deseada dentro del máximo de iteraciones, o falló antes.", ln=True)

    pdf.ln(10) # Añadir espacio

    # Sección de Tabla de Iteraciones
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Tabla de Iteraciones:", ln=True)
    pdf.ln(2)

    # Generar tabla en el PDF si hay iteraciones
    if iteraciones:
        # Formatear los datos de las iteraciones a strings para el PDF, usando notación científica para flotantes.
        formatted_iteraciones = []
        for fila in iteraciones:
            formatted_row = []
            for i, dato in enumerate(fila):
                # Verificar si el dato es numérico (float o numpy.float64)
                if isinstance(dato, (float, np.float64)):
                    formatted_row.append(f"{dato:.6e}") # Usar notación científica con 6 decimales
                else:
                    formatted_row.append(str(dato)) # Convertir otros tipos a string
            formatted_iteraciones.append(formatted_row)

        # Definir nombres y anchos de las columnas de la tabla
        col_names = ["Iter.", "x_n", "f(x_n)", "f'(x_n)", "Error Abs."]
        col_widths = [15, 35, 35, 35, 35] # Anchos en mm

        # Configurar fuente para los encabezados de la tabla
        pdf.set_font("Arial", 'B', 9)
        # Añadir encabezados de la tabla
        for name, w in zip(col_names, col_widths):
            pdf.cell(w, 7, name, border=1, align='C') # Celda con borde y centrada
        pdf.ln() # Nueva línea después de los encabezados

        # Configurar fuente para los datos de la tabla
        pdf.set_font("Arial", '', 8)
        # Añadir filas de datos a la tabla
        for fila in formatted_iteraciones:
            # Verificar si se necesita una nueva página antes de añadir la fila actual
            # Esto ayuda a evitar que la tabla se corte al final de una página.
            if pdf.get_y() + 7 > 280: # 280 es aproximadamente el final de una página A4 vertical en mm
                pdf.add_page() # Añadir nueva página
                pdf.set_font("Arial", 'B', 9) # Re-añadir encabezados en la nueva página
                for name, w in zip(col_names, col_widths):
                    pdf.cell(w, 7, name, border=1, align='C')
                pdf.ln()
                pdf.set_font("Arial", '', 8) # Volver a la fuente normal para los datos

            # Añadir datos de la fila actual
            for i, dato in enumerate(fila):
                pdf.cell(col_widths[i], 7, str(dato), border=1, align='C') # Celda con borde y centrada
            pdf.ln() # Nueva línea después de cada fila

    else:
        # Mensaje si no se realizaron iteraciones
        pdf.cell(0, 10, "No se realizaron iteraciones debido a un error inicial o convergencia inmediata.", ln=True, align='C')

    pdf.ln(10) # Añadir espacio

    # Sección de Gráfico
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Gráfico de la función:", ln=True)
    pdf.ln(2)

    # Incluir el gráfico en el PDF si se generó correctamente
    if plot_file and os.path.exists(plot_file):
        try:
            # Calcular la posición para centrar la imagen en la página
            img_width = 190 # Ancho deseado de la imagen en mm
            page_width = pdf.w - 2 * pdf.l_margin # Ancho útil de la página
            x_pos = (page_width - img_width) / 2 + pdf.l_margin # Posición x para centrar

            # Verificar si hay suficiente espacio en la página actual para el gráfico.
            # Estimamos la altura del gráfico basándonos en el ancho y un aspecto ratio típico (8x5).
            estimated_img_height = img_width * (5/8)
            if pdf.get_y() + estimated_img_height > 280: # Si no hay espacio, añadir nueva página
                pdf.add_page()
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, "Gráfico de la función (continuación):", ln=True)
                pdf.ln(2)

            # Incrustar la imagen del gráfico en el PDF
            pdf.image(plot_file, x=x_pos, w=img_width)
        except Exception as e:
            # Capturar errores al incrustar la imagen.
            print(f"Advertencia: No se pudo incrustar la imagen en el PDF. Error: {e}")
        finally:
            # Asegurarse de eliminar el archivo de gráfico temporal después de intentar incrustarlo.
            if os.path.exists(plot_file):
                os.remove(plot_file) # Eliminar archivo temporal

    else:
        # Mensaje si no se pudo generar o encontrar el archivo del gráfico.
        pdf.cell(0, 10, "No se pudo generar el gráfico o el archivo no existe.", ln=True, align='C')

    # Guardar el archivo PDF
    try:
        pdf.output(nombre_pdf)
    except Exception as e:
        # Capturar errores al guardar el archivo PDF.
        return {"error": f"Error al guardar el archivo PDF: {e}",
                "raiz": raiz, "iteraciones": df_iter, "pdf": None,
                "funcion_simbolica": f_expr, "derivada_simbolica": df_expr}

    # Devolver un diccionario con los resultados del cálculo.
    return {
        "raiz": raiz,
        "iteraciones": df_iter,
        "pdf": nombre_pdf, # Devolvemos el nombre del archivo PDF generado
        "funcion_simbolica": f_expr,
        "derivada_simbolica": df_expr,
        "error": None # Indicar que la función se ejecutó sin errores fatales
    }

# Ejemplo de cómo usar la función directamente (sin Flask)
