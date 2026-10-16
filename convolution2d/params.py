"""Parámetros únicos del proyecto. Nada de esto debe repetirse en otro archivo."""

import pathlib

# --- Rutas -------------------------------------------------------------
RAIZ = pathlib.Path(__file__).resolve().parent.parent  # proyecto/
RESULTADOS = RAIZ / "results"

# --- Problema ----------------------------------------------------------
KERNEL = "convolution2d"  # prefijo de las claves del JSON
RADIO = 2  # ventana 5x5
VENTANA = 2 * RADIO + 1
FACTOR = 1.0 / (VENTANA * VENTANA)
N_BENCH = 1024  # tamaño de la medición
N_CALENTAMIENTO = 32  # tamaño de la imagen de calentamiento

# --- Medición (benchmark) ---------------------------------------------
REFERENCIA = "v0_serial"  # define la línea base del speedup
REPETICIONES = 5
REPETICIONES_LINEA_BASE = 5  # v0 es lenta; igual que el harness del profesor
CONTEOS_TRABAJADORES = (1, 2, 4, 8, 16, 32, 64)
RTOL = 1e-12  # verificación de correctitud contra la referencia
ATOL = 1e-12

# --- Profiling ---------------------------------------------------------
N_PROFILING = N_BENCH
REPETICIONES_PROFILING = 3  # llamadas al kernel dentro del perfilador
CPROFILE_ORDEN = "cumulative"  # clave de ordenamiento de pstats
CPROFILE_TOP = 40  # filas del reporte
PYSPY_TASA_HZ = 200  # muestras por segundo
PYSPY_FORMATO = "raw"  # pilas colapsadas en texto
SCALENE_SOLO = "convolution2d"  # Scalene solo perfila rutas que contengan esto
