"""
bench.py - Harness de medición de la convolución 2D 5x5.

    uv run python bench.py                     # v0_serial y v1_prange
    uv run python bench.py --solo v0_serial
    uv run python bench.py --solo v1_prange
    uv run python bench.py --info              # solo describir la máquina

Reglas del curso:
  1. perf_counter(), nunca time.time() (monótono, mayor resolución).
  2. Cinco repeticiones y la MEDIANA, no la media.
  3. La preparación de los datos va fuera del reloj.
  4. Se calienta antes de medir: la primera llamada a una función @njit compila.
  5. La máquina se declara sola (src/entorno.py).

Salida: resultados/<version>.json, con las claves maquina, comando y mediciones.
El comando guardado lleva siempre --n explícito, así el tamaño del problema queda
registrado y la medición es reproducible.
"""

import argparse
import importlib
import json
import re
import statistics
import sys
import time

import numpy as np

from src import entorno, params
from src.common import imagen_inicial
from src.referencia import comparar, referencia


def cargar(nombre):
    """Importa src/<nombre>.py. Devuelve (módulo, motivo); motivo es None si salió bien."""
    try:
        return importlib.import_module(f"src.{nombre}"), None
    except ImportError as e:
        return None, str(e)


def cronometrar(fn, img, repeticiones):
    """Devuelve (mediana, minimo, maximo, desviacion) en segundos.

    Se hacen repeticiones+1 corridas y se descarta la primera: aunque ya se haya
    calentado, la primera corrida paga fallos de caché en frío.
    """
    tiempos = []
    for _ in range(repeticiones + 1):
        t0 = time.perf_counter()
        fn(img)
        tiempos.append(time.perf_counter() - t0)

    utiles = tiempos[1:]
    return (
        statistics.median(utiles),
        min(utiles),
        max(utiles),
        statistics.stdev(utiles) if len(utiles) > 1 else 0.0,
    )


def calentar(fn):
    """Llamada previa FUERA del reloj: compila el JIT y llena cachés."""
    fn(imagen_inicial(params.N_CALENTAMIENTO))


def verificar(fn):
    """Comprueba la versión contra la referencia antes de medirla (fuera del reloj)."""
    n = params.N_VERIFICACION
    img = np.random.default_rng(params.SEMILLA).random((n, n))
    ok, _, msg = comparar(fn(img), referencia(img))
    return ok, msg


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


def _imprimir(nombre, mediana, mn, mx, speedup, p=None):
    sp = f"{speedup:7.2f}x" if speedup else "      -"
    sufijo = f"   p={p}" if p else ""
    print(f"  {nombre:12} {mediana:9.4f} s  [{mn:.4f}-{mx:.4f}]  {sp}{sufijo}")


def _n_de_comando(comando):
    m = re.search(r"--n[ =](\d+)", comando)
    return int(m.group(1)) if m else params.N_BENCH


def _base_guardada(n):
    """Mediana de la línea base de una corrida anterior, solo si fue con el mismo N."""
    ruta = params.RESULTADOS / f"{params.LINEA_BASE}.json"
    if not ruta.exists():
        return None
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        if _n_de_comando(datos["comando"]) != n:
            return None
        for m in datos["mediciones"]:
            if m["clave"] == f"{params.KERNEL}/{params.LINEA_BASE}/tiempo":
                return m["valor"]
    except (OSError, ValueError, KeyError):
        pass
    return None


def medir(nombre, mod, img, repeticiones, base):
    """Mide una versión. Devuelve (mediciones, mediana_de_la_linea_base)."""
    fn = getattr(mod, f"convolucion2d_{nombre}")
    fijar = getattr(mod, "fijar_hilos", None)
    k = f"{params.KERNEL}/{nombre}"
    med = []

    if fijar is None:  # versión serial
        calentar(fn)
        mediana, mn, mx, sd = cronometrar(fn, img, repeticiones)
        if nombre == params.LINEA_BASE:
            base = mediana
        speedup = base / mediana if base else None
        _imprimir(nombre, mediana, mn, mx, speedup)
        med.append(_medicion(f"{k}/tiempo", mediana, "s", repeticiones, mn, mx, sd))
        if speedup:
            med.append(_medicion(f"{k}/speedup", speedup, "x", repeticiones))
        return med, base

    # Versión paralela: barrido sobre el número de hilos.
    tope = mod.hilos_disponibles()
    conteos = [p for p in params.HILOS if p <= tope]
    omitidos = [p for p in params.HILOS if p > tope]
    if omitidos:
        print(f"  AVISO: esta máquina permite {tope} hilos; se omiten p={omitidos}")

    t1 = None
    for p in conteos:
        fijar(p)
        calentar(fn)  # tras fijar los hilos: arranca el pool antes del reloj
        mediana, mn, mx, sd = cronometrar(fn, img, repeticiones)
        if t1 is None:
            t1 = mediana  # primer conteo (p=1): referencia del speedup relativo
        speedup = base / mediana if base else None
        _imprimir(nombre, mediana, mn, mx, speedup, p)
        med.append(
            _medicion(
                f"{k}/tiempo_p{p}", mediana, "s", repeticiones, mn, mx, sd, trabajadores=p
            )
        )
        if speedup:
            med.append(
                _medicion(f"{k}/speedup_p{p}", speedup, "x", repeticiones, trabajadores=p)
            )
        med.append(
            _medicion(
                f"{k}/speedup_relativo_p{p}", t1 / mediana, "x", repeticiones, trabajadores=p
            )
        )

    # El speedup "a secas" de una versión paralela es el del mayor número de hilos.
    ultimo = [m for m in med if m["clave"] == f"{k}/speedup_p{conteos[-1]}"]
    if ultimo:
        med.append(
            _medicion(
                f"{k}/speedup",
                ultimo[0]["valor"],
                "x",
                repeticiones,
                trabajadores=conteos[-1],
            )
        )
    return med, base


def escribir(mediciones, nombre, comando):
    params.RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = params.RESULTADOS / f"{nombre}.json"
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
    print(f"  Escrito {destino.relative_to(params.RAIZ)} con {len(mediciones)} mediciones")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark Convolución 2D 5x5")
    ap.add_argument("--n", type=int, default=params.N_BENCH)
    ap.add_argument("--repeticiones", type=int, default=params.REPETICIONES)
    ap.add_argument(
        "--solo",
        metavar="v0_serial,v1_prange",
        help="medir solo estas versiones, separadas por comas",
    )
    ap.add_argument("--sin-escribir", action="store_true")
    ap.add_argument("--info", action="store_true", help="solo describir la máquina")
    args = ap.parse_args(argv)

    if args.info:
        return entorno.informe()

    pedidas = (
        [s.strip() for s in args.solo.split(",") if s.strip()]
        if args.solo
        else list(params.VERSIONES)
    )
    desconocidas = [x for x in pedidas if x not in params.VERSIONES]
    if desconocidas:
        ap.error(
            f"versión desconocida: {', '.join(desconocidas)}. "
            f"Disponibles: {', '.join(params.VERSIONES)}"
        )
    # La línea base va primero: define el speedup absoluto.
    nombres = sorted(dict.fromkeys(pedidas), key=lambda x: x != params.LINEA_BASE)

    if args.repeticiones < params.REPETICIONES:
        print(
            f"  AVISO: {args.repeticiones} repeticiones es menos del mínimo del curso "
            f"({params.REPETICIONES}). El resultado no es entregable."
        )

    # Comando canónico guardado en el JSON: el tamaño N queda registrado.
    comando = f"uv run python bench.py --n {args.n} --repeticiones {args.repeticiones}"
    if args.solo:
        comando += f" --solo {','.join(pedidas)}"

    # Preparación de datos: FUERA del reloj.
    img = imagen_inicial(args.n)

    base = None
    if params.LINEA_BASE not in nombres:
        base = _base_guardada(args.n)
        if base is None:
            print(
                f"  AVISO: no hay {params.LINEA_BASE}.json para N={args.n}; no se calcula "
                f"speedup absoluto.\n  Corre primero: uv run python bench.py --n {args.n} "
                f"--solo {params.LINEA_BASE}"
            )

    print(f"\n{'=' * 68}\n  {params.KERNEL}  N={args.n}\n{'=' * 68}")

    fallos = 0
    for nombre in nombres:
        mod, motivo = cargar(nombre)
        if mod is None:
            print(f"  {nombre:12} ERROR     no se pudo importar: {motivo}")
            fallos += 1
            continue

        fijar = getattr(mod, "fijar_hilos", None)
        if fijar:
            fijar(1)
        ok, msg = verificar(getattr(mod, f"convolucion2d_{nombre}"))
        if not ok:
            print(f"  {nombre:12} ERROR     no coincide con la referencia: {msg}")
            fallos += 1
            continue

        med, base = medir(nombre, mod, img, args.repeticiones, base)
        if not args.sin_escribir:
            escribir(med, nombre, comando)

    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
