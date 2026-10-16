# proyecto_kernel.py - Línea base serial pura del Filtro de Convolución 2D (5x5)
import json
import time
import numpy as np


def imagen_inicial(n):
    """Genera una imagen sintética determinística NxN con float64."""
    img = np.zeros((n, n), dtype=np.float64)
    a, b = n // 4, 3 * n // 4
    img[a:b, a:b] = 1.0
    return img


def convolucion2d_v0_serial(img):
    """
    Filtro de desenfoque de caja (Box Blur) 5x5 en Python puro (bucles anidados).
    Bordes en cero (Zero Padding / Dirichlet).
    """
    alto, ancho = img.shape
    salida = np.zeros_like(img)
    factor = 1.0 / 25.0

    # Recorrido omitiendo el borde de 2 píxeles para la ventana de 5x5
    for i in range(2, alto - 2):
        for j in range(2, ancho - 2):
            suma = 0.0
            for ki in range(-2, 3):
                for kj in range(-2, 3):
                    suma += img[i + ki, j + kj]
            salida[i, j] = suma * factor

    return salida


if __name__ == "__main__":
    N = 256  # Tamaño de prueba para la primera medición de línea base

    print(f"--- Probando convolucion2d_v0_serial (N={N}x{N}) ---")

    # Warmup / Calentamiento
    img_test = imagen_inicial(N)
    _ = convolucion2d_v0_serial(img_test)

    # Medición de tiempo limpio de la línea base (5 repeticiones)
    tiempos = []
    for _ in range(5):
        t0 = time.perf_counter()
        res = convolucion2d_v0_serial(img_test)
        tiempos.append(time.perf_counter() - t0)

    import statistics

    mediana = statistics.median(tiempos)
    print(f"Tiempo mediano (v0_serial, N={N}): {mediana:.4f} s")
    print(
        f"Mínimo: {min(tiempos):.4f} s | Máximo: {max(tiempos):.4f} s | Desv: {statistics.stdev(tiempos):.6f}"
    )