"""py-spy: perfilador de muestreo externo (no modifica el programa)."""
import platform
import shutil
import sys

from ... import params
from ..base import ErrorPerfilador, Perfilador, correr, perfilador


@perfilador
class PySpy(Perfilador):
    nombre = "pyspy"
    descripcion = "muestreo externo; salida en pilas colapsadas (formato raw)"

    def motivo_no_disponible(self):
        if shutil.which("py-spy") is None:
            return "py-spy no está instalado (pip install py-spy)"
        return None

    def ejecutar(self, sol, destino):
        destino.parent.mkdir(parents=True, exist_ok=True)
        cmd = [shutil.which("py-spy"), "record",
               "--format", params.PYSPY_FORMATO,
               "--rate", str(params.PYSPY_TASA_HZ),
               "--output", str(destino),
               "--", *sol.comando_objetivo()]
        if sol.opciones.get("sudo"):
            cmd.insert(0, "sudo")

        try:
            correr(cmd)
        except ErrorPerfilador as e:
            raise ErrorPerfilador(f"{e}\n{self._pista()}") from None

        if not destino.exists():
            raise ErrorPerfilador("py-spy terminó sin escribir el archivo de salida")

    @staticmethod
    def _pista():
        if platform.system() == "Darwin":
            return "Pista: en macOS py-spy exige privilegios; repite con --sudo."
        if platform.system() == "Linux":
            return ("Pista: puede faltar permiso de ptrace. Repite con --sudo, o "
                    "'sudo sysctl kernel.yama.ptrace_scope=0'; en Docker usa "
                    "--cap-add=SYS_PTRACE.")
        return "Pista: revisa los permisos para adjuntar a procesos."