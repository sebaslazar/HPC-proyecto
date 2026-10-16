"""
Perfila versiones con cProfile, py-spy y Scalene. Salida: results/<version>__<herramienta>.txt

    python -m convolution2d.profiling --listar
    python -m convolution2d.profiling --version v0_serial
    python -m convolution2d.profiling --version v0_serial --herramienta cprofile,scalene
    python -m convolution2d.profiling --version v0_serial v1_numpy --herramienta pyspy --sudo
    python -m convolution2d.profiling --version todas --repeticiones 5
    python -m convolution2d.profiling --version v2_prange --trabajadores 4
"""
import argparse
import sys

from .. import params, registry
from . import herramientas
from .base import PERFILADORES, ErrorPerfilador, Solicitud


def main(argv=None):
    herramientas.descubrir()

    ap = argparse.ArgumentParser(description="Profiling de versiones de convolution2d")
    ap.add_argument("--version", nargs="+",
                    help="versiones a perfilar, o 'todas'")
    ap.add_argument("--herramienta", default="todas",
                    help=f"lista separada por comas, o 'todas' ({', '.join(PERFILADORES)})")
    ap.add_argument("--n", type=int, default=params.N_PROFILING)
    ap.add_argument("--repeticiones", type=int, default=params.REPETICIONES_PROFILING,
                    help="llamadas al kernel dentro del perfilador")
    ap.add_argument("--trabajadores", type=int, default=None,
                    help="hilos para versiones paralelas (se añade _pN al nombre)")
    ap.add_argument("--sudo", action="store_true",
                    help="ejecutar py-spy con sudo (macOS / permisos de ptrace)")
    ap.add_argument("--listar", action="store_true",
                    help="mostrar versiones y herramientas disponibles")
    args = ap.parse_args(argv)

    motivos = registry.descubrir()

    if args.listar:
        print("Versiones:")
        for nombre in sorted(registry.REGISTRO):
            print(f"  {nombre}")
        for modulo, motivo in motivos.items():
            print(f"  {modulo}  (no disponible: {motivo})")
        print("Herramientas:")
        for p in PERFILADORES.values():
            motivo = p.motivo_no_disponible()
            print(f"  {p.nombre:10} {p.descripcion}" + (f"  [NO DISPONIBLE: {motivo}]" if motivo else ""))
        return 0

    if not args.version:
        ap.error("indique --version (o --listar)")

    versiones = (sorted(registry.REGISTRO) if args.version == ["todas"]
                 else list(dict.fromkeys(args.version)))
    if args.herramienta == "todas":
        nombres_h = list(PERFILADORES)
    else:
        nombres_h = [h.strip() for h in args.herramienta.split(",") if h.strip()]
        desconocidas = [h for h in nombres_h if h not in PERFILADORES]
        if desconocidas:
            ap.error(f"herramienta desconocida: {', '.join(desconocidas)}. "
                     f"Disponibles: {', '.join(PERFILADORES)}")

    hechos, fallos = [], []
    for version in versiones:
        try:
            registry.cargar(version)
        except (ImportError, KeyError) as e:
            print(f"  {version:14} SALTADA   {motivos.get(version, e)}")
            fallos.append((version, "-"))
            continue

        for nombre_h in nombres_h:
            p = PERFILADORES[nombre_h]
            motivo = p.motivo_no_disponible()
            if motivo:
                print(f"  {version:14} {nombre_h:9} SALTADO   {motivo}")
                fallos.append((version, nombre_h))
                continue

            sol = Solicitud(version, args.n, args.repeticiones, args.trabajadores,
                            {"sudo": args.sudo})
            destino = sol.destino(nombre_h)
            print(f"  {version:14} {nombre_h:9} perfilando...", flush=True)
            try:
                p.ejecutar(sol, destino)
            except ErrorPerfilador as e:
                print(f"  {version:14} {nombre_h:9} ERROR\n{e}")
                fallos.append((version, nombre_h))
            else:
                print(f"  {version:14} {nombre_h:9} -> {destino}")
                hechos.append(destino)

    print(f"\n  {len(hechos)} reportes escritos, {len(fallos)} omitidos o fallidos")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())