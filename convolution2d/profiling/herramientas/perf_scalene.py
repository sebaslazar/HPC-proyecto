"""Scalene: perfilador de muestreo por línea (CPU y memoria)."""
import importlib.util
import shlex
import sys
import tempfile
from importlib import metadata

from ... import params
from ..base import Perfilador, correr, escribir, perfilador


def _es_scalene_2():
    try:
        return int(metadata.version("scalene").split(".")[0]) >= 2
    except (metadata.PackageNotFoundError, ValueError):
        return False


@perfilador
class Scalene(Perfilador):
    nombre = "scalene"
    descripcion = "muestreo por línea: CPU, memoria y copias"

    def motivo_no_disponible(self):
        if importlib.util.find_spec("scalene") is None:
            return "scalene no está instalado (pip install scalene)"
        return None

    def ejecutar(self, sol, destino):
        script, *args = sol.comando_objetivo()[1:]
        py = sys.executable
        # Sin --profile-only, Scalene ignora el código fuera de la carpeta del script
        # (y el kernel vive en versions/, no en profiling/).
        filtro = ["--program-path", str(params.RAIZ / "convolution2d"), "--cpu-only"]

        with tempfile.TemporaryDirectory() as tmp:   # el JSON intermedio no queda en results/
            if _es_scalene_2():
                run = [py, "-m", "scalene", "run", *filtro, script, "---", *args]
                ver = [py, "-m", "scalene", "view", "--cli"]
                correr(run, cwd=tmp)
                salida = correr(ver, cwd=tmp).stdout
                comando = f"{shlex.join(run)}  &&  {shlex.join(ver)}"
            else:
                cmd = [py, "-m", "scalene", "--cli", *filtro, script, "---", *args]
                salida = correr(cmd, cwd=tmp).stdout
                comando = shlex.join(cmd)

        escribir(destino, sol, self.nombre, comando, salida)