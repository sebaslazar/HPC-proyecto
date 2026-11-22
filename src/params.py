"""Parámetros únicos del proyecto. Nada de esto debe repetirse en otro archivo."""

import pathlib

# --- Rutas -------------------------------------------------------------
RAIZ = pathlib.Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "resultados"

# --- Problema ----------------------------------------------------------
KERNEL = "convolution2d"  # prefijo de las claves del JSON
RADIO = 2  # ventana 5x5
VENTANA = 2 * RADIO + 1
FACTOR = 1.0 / (VENTANA * VENTANA)

# --- Tamaños -----------------------------------------------------------
N_BENCH = 1024  # tamaño de la medición
N_CALENTAMIENTO = 32  # imagen de calentamiento (compila el JIT), fuera del reloj
N_VERIFICACION = 64  # imagen de las pruebas de correctitud

# --- Medición ----------------------------------------------------------
VERSIONES = ("v0_serial", "v1_prange")  # nombre del módulo en src/
LINEA_BASE = "v0_serial"  # define el speedup absoluto
REPETICIONES = 5
HILOS = (1, 2, 4)  # barrido de trabajadores de v1_prange
TIEMPO_MIN_FIABLE_S = 0.005  # por debajo, bench.py avisa: el ruido pesa en la mediana

# --- Correctitud -------------------------------------------------------
RTOL = 1e-12
ATOL = 1e-12
SEMILLA = 0  # imagen aleatoria reproducible
