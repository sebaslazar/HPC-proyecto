"""Declaración de la máquina: se escribe en cada JSON de resultados."""

import json
import os
import platform
import subprocess
from importlib import metadata


def _cmd(*args):
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _version(*distribuciones):
    for d in distribuciones:
        try:
            return metadata.version(d)
        except metadata.PackageNotFoundError:
            continue
    return None


def _psutil():
    try:
        import psutil

        return psutil
    except ImportError:
        return None


def _cpu():
    sistema = platform.system()
    if sistema == "Darwin":
        r = _cmd("sysctl", "-n", "machdep.cpu.brand_string")
        if r:
            return r
    elif sistema == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for linea in f:
                    if linea.startswith("model name"):
                        return linea.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _nucleos_fisicos():
    ps = _psutil()
    if ps and ps.cpu_count(logical=False):
        return ps.cpu_count(logical=False)
    if platform.system() == "Darwin":
        r = _cmd("sysctl", "-n", "hw.physicalcpu")
        if r and r.isdigit():
            return int(r)
    return os.cpu_count()  # sin psutil en Linux/Windows: se asume = lógicos


def _memoria_gb():
    total = None
    ps = _psutil()
    if ps:
        total = ps.virtual_memory().total
    elif platform.system() == "Darwin":
        r = _cmd("sysctl", "-n", "hw.memsize")
        total = int(r) if r and r.isdigit() else None
    else:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            pass
    return round(total / 2**30, 1) if total else None


def _gpu():
    r = _cmd("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
    return r.splitlines()[0].strip() if r else None


def describir():
    return {
        "cpu": _cpu(),
        "nucleos_fisicos": _nucleos_fisicos(),
        "nucleos_logicos": os.cpu_count(),
        "memoria_gb": _memoria_gb(),
        "gpu": _gpu(),
        "sistema": f"{platform.system()} {platform.release()}",
        "arquitectura": platform.machine(),
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "numba": _version("numba"),
        "mpi4py": _version("mpi4py"),
        "dask": _version("dask"),
        "cupy": _version("cupy", "cupy-cuda12x", "cupy-cuda11x", "cupy-cuda13x"),
    }


def informe():
    print(json.dumps(describir(), indent=2, ensure_ascii=False))
    return 0
