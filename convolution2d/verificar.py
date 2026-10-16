"""
verificar.py - Árbitro del proyecto: comprueba v0_serial contra un modelo de referencia
independiente en NumPy (sliding_window_view) y expone la función de comparación.

Uso:
    uv run python -m convolution2d.verificar
"""

import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import params
from .common import imagen_inicial
from .versions.v0_serial import convolucion2d_v0_serial


def modelo_referencia(img):
    """Modelo independiente: calcula la media de la ventana (5x5) usando
    vistas de ventana deslizante vectorizadas de NumPy.
    """
    salida = np.zeros_like(img)
    r = params.RADIO
    vistas = sliding_window_view(img, (params.VENTANA, params.VENTANA))
    salida[r:-r, r:-r] = vistas.mean(axis=(2, 3))
    return salida


def comparar(obtenido, esperado, rtol=params.RTOL, atol=params.ATOL):
    """Compara un resultado paralelo/vectorizado contra la referencia.

    Devuelve (bool, mensaje). Exige coincidencia de forma y que la diferencia
    absoluta máxima respete la tolerancia declarada.
    """
    obtenido = np.asarray(obtenido)
    esperado = np.asarray(esperado)

    if obtenido.shape != esperado.shape:
        return (
            False,
            f"Las formas no coinciden: obtenido {obtenido.shape}, esperado {esperado.shape}",
        )

    err_max = float(np.max(np.abs(obtenido - esperado)))
    correcto = bool(np.allclose(obtenido, esperado, rtol=rtol, atol=atol))

    if correcto:
        return True, f"Correcto: error máximo {err_max:.3e} (<= atol={atol})"
    else:
        idx = np.unravel_index(
            int(np.argmax(np.abs(obtenido - esperado))), obtenido.shape
        )
        return (
            False,
            f"ERROR: máximo {err_max:.3e} en {idx}. Obtenido {obtenido[idx]!r}, esperado {esperado[idx]!r}",
        )


def main():
    print("=== Verificando v0_serial contra el Modelo Numérico de Referencia ===")

    entradas = {
        "sintetica": imagen_inicial(64),
        "aleatoria": np.random.default_rng(0).random(
            (64, 64)
        ),  # Semilla fija para reproducibilidad
    }

    todo_ok = True
    for nombre, img in entradas.items():
        esperado = modelo_referencia(img)
        obtenido = convolucion2d_v0_serial(img)

        ok, msg = comparar(obtenido, esperado)
        todo_ok = todo_ok and ok
        err_max = np.max(np.abs(obtenido - esperado))
        print(
            f"  Imagen {nombre:10} | Coincide: {ok} | max|dif| = {err_max:.3e} -> {msg}"
        )

    return 0 if todo_ok else 1


if __name__ == "__main__":
    sys.exit(main())
