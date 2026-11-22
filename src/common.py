"""Datos de entrada compartidos por todas las implementaciones."""

import numpy as np


def imagen_inicial(n):
    """Genera una imagen sintética determinística NxN con float64."""
    img = np.zeros((n, n), dtype=np.float64)
    a, b = n // 4, 3 * n // 4
    img[a:b, a:b] = 1.0
    return img
