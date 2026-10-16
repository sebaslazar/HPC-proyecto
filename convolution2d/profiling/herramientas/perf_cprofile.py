"""cProfile: perfilador determinista (cuenta cada llamada a función)."""
import cProfile
import io
import pstats

from ... import params
from ..base import Perfilador, escribir, perfilador


@perfilador
class CProfile(Perfilador):
    nombre = "cprofile"
    descripcion = "perfilador determinista por función (stdlib)"

    def ejecutar(self, sol, destino):
        from ..objetivo import preparar

        correr = preparar(sol.version, sol.n, sol.trabajadores)  # fuera del perfil

        pr = cProfile.Profile()
        pr.enable()
        for _ in range(sol.repeticiones):
            correr()
        pr.disable()

        buf = io.StringIO()
        stats = pstats.Stats(pr, stream=buf).strip_dirs().sort_stats(params.CPROFILE_ORDEN)
        stats.print_stats(params.CPROFILE_TOP)

        comando = (f"cProfile.Profile() sobre objetivo.preparar("
                   f"{sol.version!r}, n={sol.n}) x{sol.repeticiones}")
        escribir(destino, sol, self.nombre, comando, buf.getvalue())