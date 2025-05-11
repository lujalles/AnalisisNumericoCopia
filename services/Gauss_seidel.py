import numpy as np
import matplotlib.pyplot as plt
import os

def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float).copy()
    history = []
    for k in range(1, max_iter + 1):
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]
        history.append((k, x.copy()))
        if np.linalg.norm(np.dot(A, x) - b, np.inf) < tol:
            break
    return x, history

def calcular_gauss_seidel_y_generar_pdf(A_list, b_list, x0=None, tol=1e-6, max_iter=100):
    A = np.array(A_list, dtype=float)
    b = np.array(b_list, dtype=float)
    sol, history = gauss_seidel(A, b, x0, tol, max_iter)

    lines = []
    lines.append("Gauss–Seidel Solver\n")
    lines.append(f"Matriz A ({len(A)}×{len(A)}):")
    for row in A:
        lines.append("  " + "  ".join(f"{val:.4f}" for val in row))
    lines.append(f"\nVector b:  " + "  ".join(f"{bi:.4f}" for bi in b))
    if x0 is not None:
        lines.append("x0: " + "  ".join(f"{xi:.4f}" for xi in x0))
    lines.append(f"Tolerancia: {tol}")
    lines.append(f"Máx. iteraciones: {max_iter}\n")
    lines.append("Iteraciones:")
    for k, xv in history:
        lines.append(f"  {k:>2} | " + "  ".join(f"{xi:.6f}" for xi in xv))
    lines.append("\nSolución aproximada:")
    for i, xi in enumerate(sol, start=1):
        lines.append(f"  x[{i}] = {xi:.6f}")

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    fig.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)

    output_path = os.path.join("static", "resultado_gauss_seidel.pdf")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
