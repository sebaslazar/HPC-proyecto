"""
test_correctitud.py - Verifica v0_serial y v1_prange contra src/referencia.py.

    uv run python test_correctitud.py

Cada prueba imprime PASS o FAIL con el error máximo absoluto. v1_prange se prueba
con 1, 2 y 4 hilos. El código de salida es 0 solo si TODAS pasan; es 1 si alguna
falla o si no se pudo importar una versión. Un conteo de hilos que la máquina no
permite se marca SKIP y no cuenta como fallo.
"""

import importlib
import sys

import numpy as np

from src import params
from src.common import imagen_inicial
from src.referencia import comparar, referencia


def entradas():
    n = params.N_VERIFICACION
    return {
        "sintetica": imagen_inicial(n),
        "aleatoria": np.random.default_rng(params.SEMILLA).random((n, n)),
    }


def correr():
    """Ejecuta todas las pruebas. Devuelve la lista de resultados (True/False)."""
    imgs = entradas()
    esperados = {nombre: referencia(img) for nombre, img in imgs.items()}
    resultados = []

    for nombre in params.VERSIONES:
        try:
            mod = importlib.import_module(f"src.{nombre}")
        except ImportError as e:
            print(f"[FAIL] {nombre}: no se pudo importar ({e})")
            resultados.append(False)
            continue

        fn = getattr(mod, f"convolucion2d_{nombre}")
        fijar = getattr(mod, "fijar_hilos", None)
        conteos = [None] if fijar is None else list(params.HILOS)

        for p in conteos:
            etiqueta = nombre if p is None else f"{nombre} p={p}"
            if p is not None:
                if p > mod.hilos_disponibles():
                    print(
                        f"[SKIP] {etiqueta}: la máquina permite "
                        f"{mod.hilos_disponibles()} hilos"
                    )
                    continue
                fijar(p)
            for entrada, img in imgs.items():
                ok, _, msg = comparar(fn(img), esperados[entrada])
                veredicto = "PASS" if ok else "FAIL"
                print(f"[{veredicto}] {etiqueta:15} | {entrada:9} | {msg}")
                resultados.append(ok)
    return resultados


def main():
    print(f"=== Correctitud contra src/referencia.py (N={params.N_VERIFICACION}) ===")
    resultados = correr()
    pasadas = sum(resultados)
    total = len(resultados)
    if total == 0 or pasadas != total:
        print(f"\nRESULTADO: FAIL ({pasadas}/{total} pruebas pasaron)")
        return 1
    print(f"\nRESULTADO: PASS ({pasadas}/{total} pruebas pasaron)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
