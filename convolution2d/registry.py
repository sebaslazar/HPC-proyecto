"""Registro de implementaciones: nombre -> Version, con autodescubrimiento."""
import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass

from . import versions as _paquete
from .common import imagen_inicial
from .params import N_CALENTAMIENTO


@dataclass
class Version:
    nombre: str
    funcion: Callable
    paralela: bool = False
    fijar: Callable[[int], dict] | None = None  # fija p trabajadores; devuelve kwargs extra

    def fijar_trabajadores(self, p):
        return self.fijar(p) if self.fijar else {}

    def calentar(self):
        """Llamada previa fuera del reloj (compila @njit, llena cachés)."""
        self.funcion(imagen_inicial(N_CALENTAMIENTO))


REGISTRO = {}


def registrar(nombre, paralela=False, fijar=None):
    def decorador(fn):
        REGISTRO[nombre] = Version(nombre, fn, paralela, fijar)
        return fn
    return decorador


def cargar(nombre):
    """Importa versions/<nombre>.py y devuelve su Version. Propaga ImportError."""
    importlib.import_module(f"{_paquete.__name__}.{nombre}")
    return REGISTRO[nombre]


def descubrir():
    """Importa todos los módulos de versions/.

    Devuelve {módulo: motivo} de los que no se pudieron importar
    (típicamente una dependencia opcional que falta).
    """
    motivos = {}
    for info in pkgutil.iter_modules(_paquete.__path__):
        try:
            importlib.import_module(f"{_paquete.__name__}.{info.name}")
        except ImportError as e:
            motivos[info.name] = str(e)
    return motivos