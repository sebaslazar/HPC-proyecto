"""v1: memoria compartida con Numba. Mismo Box Blur 5x5 que v0, con prange sobre las filas."""

import numba
import numpy as np
from numba import njit, prange

from .params import FACTOR, RADIO


@njit(parallel=True, cache=True)
def _nucleo(img, salida):
    alto, ancho = img.shape
    # Cada hilo recibe filas i distintas y escribe solo en salida[i, :]. La suma es un
    # escalar local de cada píxel (se declara dentro del bucle): no hay estado
    # compartido y, por tanto, no hay condiciones de carrera.
    for i in prange(RADIO, alto - RADIO):
        for j in range(RADIO, ancho - RADIO):
            suma = 0.0
            for ki in range(-RADIO, RADIO + 1):
                for kj in range(-RADIO, RADIO + 1):
                    suma += img[i + ki, j + kj]
            salida[i, j] = suma * FACTOR


def convolucion2d_v1_prange(img):
    """Box Blur 5x5 paralelo. Usa los hilos fijados con fijar_hilos()."""
    salida = np.zeros_like(img)
    _nucleo(img, salida)
    return salida


def hilos_disponibles():
    """Máximo de hilos que Numba permite fijar en esta máquina."""
    return numba.config.NUMBA_NUM_THREADS


def fijar_hilos(p):
    """Fija el número de hilos de las siguientes llamadas (p <= hilos_disponibles())."""
    numba.set_num_threads(p)
