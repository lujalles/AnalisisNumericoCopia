import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import symbols, lambdify, sympify
from .pdf_generator import generate_pdf_report  # Importamos nuestro generador de PDFs

def biseccion(f, a, b, es, imax, xi, xf):
    """
    Implementación del método de bisección con generación de PDF profesional

    Args:
        f (str): Función en formato string
        a (float): Límite inferior del intervalo
        b (float): Límite superior del intervalo
        es (float): Error tolerado
        imax (int): Número máximo de iteraciones
        xi (float): Límite inferior para gráficos
        xf (float): Límite superior para gráficos

    Returns:
        float: Aproximación de la raíz
    """
    x = symbols("x")
    fn = sympify(f)
    f_lambda = lambdify(x, fn)

    # Verificar cambio de signo
    if f_lambda(a) * f_lambda(b) > 0:
        raise ValueError("No hay cambio de signo en el intervalo dado. El método no puede aplicarse.")

    # Inicialización de variables
    m = (a + b) / 2
    i = 0
    ea = 2 * es  # Error inicial

    # Almacenar datos para tabla y gráficos
    bisec_table = []
    iter_numbers = []
    errors = []

    # Primera iteración
    bisec_table.append([i, a, b, f_lambda(a), f_lambda(b), m, f_lambda(m), "--"])
    iter_numbers.append(i)
    errors.append(ea)

    # Iteraciones principales
    while ea > es and i < imax:
        i += 1

        # Determinar subintervalo
        if f_lambda(a) * f_lambda(m) < 0:
            b = m
        else:
            a = m

        # Calcular nuevo punto medio y error
        m_prev = m
        m = (a + b) / 2

        if m != 0:
            ea = abs(m - m_prev)

        # Registrar datos
        bisec_table.append([i, a, b, f_lambda(a), f_lambda(b), m, f_lambda(m), ea])
        iter_numbers.append(i)
        errors.append(ea)

    # ----------------------------
    # PREPARACIÓN DE DATOS PARA EL PDF
    # ----------------------------

    # 1. Parámetros del método
    params = {
        'a': a,
        'b': b,
        'es': es,
        'imax': imax,
        'xi': xi,
        'xf': xf
    }

    # 2. Resultados obtenidos
    results = {
        'root': m,
        'f_root': f_lambda(m),
        'iterations': i,
        'final_error': ea
    }

    # 3. Tabla de iteraciones (convertir a DataFrame)
    df = pd.DataFrame(
        bisec_table, 
        columns=["Iteración", "a", "b", "f(a)", "f(b)", "m", "f(m)", "Error abs."]
    )

    # Dividir la tabla si es muy larga (15 filas por página)
    tables = []
    rows_per_page = 15
    for i in range(0, len(df), rows_per_page):
        tables.append(df.iloc[i:i + rows_per_page])

    # 4. Gráficos
    graphs = []

    # Gráfico 1: Evolución del error
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(iter_numbers[1:], errors[1:], 'o-', color='#e74c3c', linewidth=2, markersize=6)
    plt.xlabel("Número de iteraciones", fontweight='bold')
    plt.ylabel("Error absoluto", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("Convergencia del error", pad=20)
    graphs.append((fig1, "Evolución del error absoluto por iteración"))

    # Gráfico 2: Función y raíz encontrada
    fig2 = plt.figure(figsize=(10, 5))
    x_vals = np.linspace(xi, xf, 500)
    y_vals = [f_lambda(x) for x in x_vals]

    plt.plot(x_vals, y_vals, color='#3498db', linewidth=2, label=f"$f(x) = {str(fn)}$")
    plt.axhline(0, color='#2c3e50', linestyle='--', alpha=0.7)
    plt.scatter([m], [f_lambda(m)], color='#e74c3c', s=100, zorder=5, 
               label=f"Raíz encontrada ($x = {m:.6f}$)")

    plt.xlabel("x", fontweight='bold')
    plt.ylabel("f(x)", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.title("Gráfica de la función y raíz encontrada", pad=20)
    graphs.append((fig2, "Análisis gráfico de la función"))

    # ----------------------------
    # GENERAR EL PDF
    # ----------------------------
    generate_pdf_report(
        output_filename='reporte_biseccion.pdf',
        method_name="Bisección",
        fn_str=str(fn),
        params=params,
        results=results,
        tables=tables,
        graphs=graphs
    )

    return m

# Ejemplo de uso
if __name__ == "__main__":
    # Parámetros de ejemplo
    funcion = "x**3 - x - 2"
    a = 1.0
    b = 2.0
    error_tolerado = 0.0001
    iteraciones_max = 100
    x_inicial = 0.0
    x_final = 3.0

    # Ejecutar método
    raiz = biseccion(funcion, a, b, error_tolerado, iteraciones_max, x_inicial, x_final)
    print(f"Raíz encontrada: {raiz}")