"""
benchmark.py - Harness de medición de la convolución 2D 5x5.

    python -m convolution2d.benchmark --info
    python -m convolution2d.benchmark
    python -m convolution2d.benchmark --solo v0_serial
    python -m convolution2d.benchmark --solo v1_numpy,v2_prange

Reglas del curso:
  1. perf_counter(), nunca time.time() (monótono, mayor resolución).
  2. Cinco repeticiones y la MEDIANA, no la media.
  3. La preparación de los datos va fuera del reloj.
  4. Se llama a calentar() antes de medir (compilación JIT, cachés).
  5. La máquina se declara sola (entorno.describir()).

Salida: proyecto/results/<version>.json
"""

import argparse
import importlib
import json
import os
import pathlib
import re
import statistics
import sys
import time

import numpy as np

from . import entorno
from .common import KERNEL, N_BENCH, imagen_inicial
from .registry import REGISTRO

RESULTADOS = pathlib.Path(__file__).resolve().parent.parent / "results"

# Añade aquí cada versión nueva (nombre del módulo = nombre registrado)
VERSIONES = ["v0_serial"]
REFERENCIA = "v0_serial"

REPETICIONES = 5
# v0 en Python puro es lenta: se mide con menos repeticiones (como en el
# harness del profesor). Es la línea base, no necesita precisión de tres cifras.
REPETICIONES_LINEA_BASE = 5


def cargar(nombres):
    """Importa los módulos; si falta una dependencia, devuelve el motivo y sigue."""
    motivos = {}
    for nombre in nombres:
        try:
            importlib.import_module(f".{nombre}", package=__package__)
        except ImportError as e:
            motivos[nombre] = str(e)
    return motivos


def cronometrar(fn, argumentos, repeticiones):
    """Devuelve (mediana, minimo, maximo, desviacion) en segundos.

    Se hacen repeticiones+1 corridas y se descarta la primera: aunque se haya
    llamado a calentar(), la primera paga fallos de caché en frío.
    """
    tiempos = []
    for _ in range(repeticiones + 1):
        t0 = time.perf_counter()
        fn(**argumentos)
        tiempos.append(time.perf_counter() - t0)

    utiles = tiempos[1:] if len(tiempos) > 1 else tiempos
    return (
        statistics.median(utiles),
        min(utiles),
        max(utiles),
        statistics.stdev(utiles) if len(utiles) > 1 else 0.0,
    )


def _medicion(
    clave, valor, unidad, repeticiones, mn=None, mx=None, sd=None, trabajadores=1
):
    m = {
        "clave": clave,
        "valor": round(float(valor), 6),
        "unidad": unidad,
        "repeticiones": repeticiones,
        "trabajadores": trabajadores,
    }
    if mn is not None:
        m["minimo"] = round(float(mn), 6)
        m["maximo"] = round(float(mx), 6)
        m["desviacion"] = round(float(sd), 6)
    return m


def _n_de_comando(comando):
    m = re.search(r"--n[ =](\d+)", comando)
    return int(m.group(1)) if m else N_BENCH


def _base_guardada(n):
    """Tiempo de v0_serial de una corrida anterior, si es del mismo N."""
    ruta = RESULTADOS / f"{REFERENCIA}.json"
    if not ruta.exists():
        return None
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        if _n_de_comando(datos["comando"]) != n:
            return None
        for m in datos["mediciones"]:
            if m["clave"] == f"{KERNEL}/{REFERENCIA}/tiempo":
                return m["valor"]
    except (OSError, ValueError, KeyError):
        pass
    return None


def medir_version(v, argumentos, repeticiones, conteos, base):
    """Mide una versión. Devuelve (mediciones, base)."""
    v.calentar()
    reps = REPETICIONES_LINEA_BASE if v.nombre == REFERENCIA else repeticiones
    k = f"{KERNEL}/{v.nombre}"
    med = []

    if not v.paralela:
        mediana, mn, mx, sd = cronometrar(v.funcion, argumentos, reps)
        if v.nombre == REFERENCIA:
            base = mediana
        speedup = base / mediana if base else None
        print(
            f"  {v.nombre:22} {mediana:9.4f} s  [{mn:.4f}-{mx:.4f}]  "
            + (f"{speedup:7.2f}x" if speedup else "      -")
        )
        med.append(_medicion(f"{k}/tiempo", mediana, "s", reps, mn, mx, sd))
        if speedup:
            med.append(_medicion(f"{k}/speedup", speedup, "x", reps))
        return med, base

    # Versión paralela: barrido sobre el número de trabajadores.
    for p in conteos:
        extra = v.fijar_trabajadores(p)
        mediana, mn, mx, sd = cronometrar(v.funcion, {**argumentos, **extra}, reps)
        speedup = base / mediana if base else None
        print(
            f"  {v.nombre:22} {mediana:9.4f} s  [{mn:.4f}-{mx:.4f}]  "
            + (f"{speedup:7.2f}x" if speedup else "      -")
            + f"   p={p}"
        )
        med.append(
            _medicion(
                f"{k}/tiempo_p{p}", mediana, "s", reps, mn, mx, sd, trabajadores=p
            )
        )
        if speedup:
            med.append(
                _medicion(f"{k}/speedup_p{p}", speedup, "x", reps, trabajadores=p)
            )
    # El speedup "a secas" de una versión paralela es el de todos los núcleos.
    ultimo = [m for m in med if m["clave"] == f"{k}/speedup_p{conteos[-1]}"]
    if ultimo:
        med.append(
            _medicion(
                f"{k}/speedup", ultimo[0]["valor"], "x", reps, trabajadores=conteos[-1]
            )
        )
    return med, base


def escribir(mediciones, version, comando):
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = RESULTADOS / f"{version}.json"
    destino.write_text(
        json.dumps(
            {
                "maquina": entorno.describir(),
                "comando": comando,
                "mediciones": mediciones,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  Escrito {destino} con {len(mediciones)} mediciones")


def barrido(
    n,
    repeticiones,
    nombres,
    trabajadores_max=None,
    verificar=True,
    comando="",
    escribir_json=True,
):
    motivos = cargar(list(dict.fromkeys([REFERENCIA, *nombres])))
    # La referencia va primero: define la línea base del speedup.
    orden = sorted(dict.fromkeys(nombres), key=lambda x: x != REFERENCIA)

    # Preparación de datos: FUERA del reloj.
    argumentos = {"img": imagen_inicial(n)}

    nucleos = trabajadores_max or os.cpu_count() or 1
    conteos = [p for p in (1, 2, 4, 8, 16, 32, 64) if p <= nucleos]
    if nucleos not in conteos:
        conteos.append(nucleos)

    base = None
    ref = None
    if REFERENCIA not in nombres:
        base = _base_guardada(n)
        if base is None:
            print(
                f"  AVISO: no hay {REFERENCIA}.json para N={n}; no se calcula speedup.\n"
                f"  Corre primero: python -m convolution2d.benchmark --solo {REFERENCIA}"
            )
        if verificar and REFERENCIA in REGISTRO:
            ref = REGISTRO[REFERENCIA].funcion(**argumentos)
    # (si se mide v0 no hay nada contra qué verificarlo)

    print(f"\n{'=' * 68}\n  {KERNEL}  N={n}\n{'=' * 68}")

    for nombre in orden:
        if nombre not in REGISTRO:
            print(f"  {nombre:22} SALTADO   {motivos.get(nombre, 'no registrada')}")
            continue
        v = REGISTRO[nombre]

        if ref is not None and nombre != REFERENCIA:
            extra = v.fijar_trabajadores(1)
            if not np.allclose(
                v.funcion(**argumentos, **extra), ref, rtol=1e-12, atol=1e-12
            ):
                print(f"  {nombre:22} ERROR     no coincide con {REFERENCIA}")
                continue

        med, base = medir_version(v, argumentos, repeticiones, conteos, base)
        if escribir_json:
            escribir(med, nombre, comando)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark Convolución 2D 5x5")
    ap.add_argument("--n", type=int, default=N_BENCH)
    ap.add_argument("--repeticiones", type=int, default=REPETICIONES)
    ap.add_argument(
        "--solo",
        metavar="v0_serial,v1_numpy",
        help="medir solo estas versiones, separadas por comas",
    )
    ap.add_argument("--trabajadores-max", type=int, default=None)
    ap.add_argument("--sin-escribir", action="store_true")
    ap.add_argument("--sin-verificar", action="store_true")
    ap.add_argument("--info", action="store_true", help="solo describir la máquina")
    args = ap.parse_args(argv)

    if args.info:
        return entorno.informe()

    if args.repeticiones < REPETICIONES:
        print(
            f"  AVISO: {args.repeticiones} repeticiones es menos del mínimo "
            f"del curso ({REPETICIONES}). El resultado no es entregable."
        )

    # Comando exacto guardado en el JSON para que la medición sea reproducible.
    comando = "python -m convolution2d.benchmark " + " ".join(
        sys.argv[1:] if argv is None else argv
    )
    nombres = (
        [s.strip() for s in args.solo.split(",") if s.strip()]
        if args.solo
        else VERSIONES
    )

    barrido(
        args.n,
        args.repeticiones,
        nombres,
        args.trabajadores_max,
        not args.sin_verificar,
        comando.strip(),
        not args.sin_escribir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
