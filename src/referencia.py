"""Modelo de referencia independiente: media de cada ventana 5x5 con sliding_window_view.

No comparte código con las implementaciones: es el árbitro contra el que se verifican.
El borde de RADIO píxeles queda en 0.0 (Dirichlet), igual que en v0_serial.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import params
from .common import imagen_inicial


def entradas_verificacion():
    """Entradas de las pruebas de correctitud: las mismas en el test y en bench.py."""
    n = params.N_VERIFICACION
    return {
        "sintetica": imagen_inicial(n),
        "aleatoria": np.random.default_rng(params.SEMILLA).random((n, n)),
    }


def referencia(img):
    """Box Blur 5x5 vectorizado: interior calculado, borde en 0.0."""
    r = params.RADIO
    salida = np.zeros_like(img)
    ventanas = sliding_window_view(img, (params.VENTANA, params.VENTANA))
    salida[r:-r, r:-r] = ventanas.mean(axis=(2, 3))
    return salida


def comparar(obtenido, esperado, rtol=params.RTOL, atol=params.ATOL):
    """Devuelve (ok, error_maximo_absoluto, mensaje).

    Exige misma forma, mismo dtype y coincidencia dentro de la tolerancia.
    """
    obtenido = np.asarray(obtenido)
    esperado = np.asarray(esperado)
    if obtenido.shape != esperado.shape:
        return False, float("inf"), f"forma {obtenido.shape} != {esperado.shape}"
    if obtenido.dtype != esperado.dtype:
        return False, float("inf"), f"dtype {obtenido.dtype} != {esperado.dtype}"
    err = float(np.max(np.abs(obtenido - esperado)))
    ok = bool(np.allclose(obtenido, esperado, rtol=rtol, atol=atol))
    return ok, err, f"max|dif| = {err:.3e} (rtol={rtol}, atol={atol})"
