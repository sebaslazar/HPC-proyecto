"""
Programa que perfilan las herramientas: prepara datos, calienta y ejecuta el
kernel N veces. Se puede correr como script (lo que necesitan Scalene y py-spy)
o importar `preparar` (lo que usa cProfile en proceso).

    python convolution2d/profiling/objetivo.py --version v0_serial --n 256
"""
import argparse
import pathlib
import sys

if __package__ in (None, ""):          # ejecución como script
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from convolution2d import params, registry
from convolution2d.common import imagen_inicial


def preparar(version, n, trabajadores=None):
    """Devuelve una función sin argumentos que ejecuta el kernel una vez.

    Datos y calentamiento ocurren AQUÍ, antes de que arranque el perfilador
    (los que lo envuelven por fuera ven import y preparación, pero pesan poco).
    """
    v = registry.cargar(version)
    img = imagen_inicial(n)
    extra = v.fijar_trabajadores(trabajadores) if trabajadores else {}
    v.calentar()
    return lambda: v.funcion(img, **extra)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--n", type=int, default=params.N_PROFILING)
    ap.add_argument("--repeticiones", type=int, default=params.REPETICIONES_PROFILING)
    ap.add_argument("--trabajadores", type=int, default=None)
    args = ap.parse_args(argv)

    correr = preparar(args.version, args.n, args.trabajadores)
    for _ in range(args.repeticiones):
        correr()


if __name__ == "__main__":
    main()