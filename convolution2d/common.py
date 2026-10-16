"""Contrato compartido por todas las implementaciones."""

import numpy as np

RADIO = 2  # ventana 5x5
VENTANA = 2 * RADIO + 1
FACTOR = 1.0 / (VENTANA * VENTANA)

KERNEL = "convolution2d"  # prefijo de las claves del JSON
N_BENCH = 256  # tamaño fijo de la medición (súbelo si v0 te queda muy rápido)


def imagen_inicial(n):
    """Genera una imagen sintética determinística NxN con float64."""
    img = np.zeros((n, n), dtype=np.float64)
    a, b = n // 4, 3 * n // 4
    img[a:b, a:b] = 1.0
    return img
