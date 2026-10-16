"""Verifica v0_serial contra un modelo de referencia independiente (NumPy).

    python -m convolution2d.verificar
"""
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import params
from .common import imagen_inicial
from .versions.v0_serial import convolucion2d_v0_serial


def referencia(img):
    """Media de cada ventana 5x5 del interior; el borde de RADIO píxeles queda en 0."""
    r = np.zeros_like(img)
    r[params.RADIO:-params.RADIO, params.RADIO:-params.RADIO] = sliding_window_view(
        img, (params.VENTANA, params.VENTANA)).mean(axis=(2, 3))
    return r


def main():
    entradas = {
        "sintetica": imagen_inicial(64),
        "aleatoria": np.random.default_rng(0).random((64, 64)),   # semilla fija
    }
    ok = True
    for nombre, img in entradas.items():
        a, b = convolucion2d_v0_serial(img), referencia(img)
        bien = bool(np.allclose(a, b, rtol=params.RTOL, atol=params.ATOL))
        ok = ok and bien
        print(f"{nombre:10} coincide: {bien}   max|dif| = {np.abs(a - b).max():.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())