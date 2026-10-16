"""Contrato y registro de perfiladores."""

import datetime
import pathlib
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .. import entorno, params

OBJETIVO = pathlib.Path(__file__).with_name("objetivo.py")
PERFILADORES = {}


class ErrorPerfilador(RuntimeError):
    """La herramienta falló; el mensaje explica por qué."""


@dataclass
class Solicitud:
    version: str
    n: int
    repeticiones: int
    trabajadores: Optional[int] = None
    opciones: dict = field(default_factory=dict)  # p. ej. {"sudo": True}

    def destino(self, herramienta):
        sufijo = f"_p{self.trabajadores}" if self.trabajadores else ""
        return params.RESULTADOS / f"{self.version}{sufijo}__{herramienta}.txt"

    def comando_objetivo(self):
        """Comando que ejecuta el kernel; lo envuelven las herramientas externas."""
        cmd = [
            sys.executable,
            str(OBJETIVO),
            "--version",
            self.version,
            "--n",
            str(self.n),
            "--repeticiones",
            str(self.repeticiones),
        ]
        if self.trabajadores:
            cmd += ["--trabajadores", str(self.trabajadores)]
        return cmd


class Perfilador(ABC):
    """Una herramienta de profiling. Para añadir una: subclase + @perfilador."""

    nombre = ""  # aparece en el nombre del archivo
    descripcion = ""

    def motivo_no_disponible(self):
        """None si se puede usar; si no, el motivo en texto."""
        return None

    @abstractmethod
    def ejecutar(self, sol, destino):
        """Corre la herramienta y deja el reporte de texto en `destino`."""


def perfilador(clase):
    PERFILADORES[clase.nombre] = clase()
    return clase


def correr(cmd, **kw):
    """Ejecuta un subproceso; si falla, lanza ErrorPerfilador con su stderr."""
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise ErrorPerfilador(
            f"código de salida {r.returncode}: {' '.join(cmd)}\n{r.stderr.strip()}"
        )
    return r


def encabezado(sol, herramienta, comando):
    m = entorno.describir()
    lineas = [
        f"herramienta  : {herramienta}",
        f"version      : {sol.version}",
        f"n            : {sol.n}",
        f"repeticiones : {sol.repeticiones}",
        f"trabajadores : {sol.trabajadores or '-'}",
        f"fecha        : {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"maquina      : {m['cpu']} | {m['nucleos_fisicos']}f/{m['nucleos_logicos']}l "
        f"núcleos | {m['memoria_gb']} GB | {m['sistema']}",
        f"python/numpy : {m['python']} / {m['numpy']}",
        f"comando      : {comando}",
    ]
    return "".join(f"# {l}\n" for l in lineas) + "\n"


def escribir(destino, sol, herramienta, comando, cuerpo):
    """Escribe encabezado reproducible + cuerpo del reporte."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(encabezado(sol, herramienta, comando) + cuerpo, encoding="utf-8")
