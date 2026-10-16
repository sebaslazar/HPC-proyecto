"""Registro de implementaciones: nombre -> Version."""

from collections.abc import Callable
from dataclasses import dataclass

from .common import imagen_inicial


@dataclass
class Version:
    nombre: str
    funcion: Callable
    paralela: bool = False
    fijar: Callable[[int], dict] | None = (
        None  # fija p trabajadores; devuelve kwargs extra
    )

    def fijar_trabajadores(self, p):
        return self.fijar(p) if self.fijar else {}

    def calentar(self):
        """Llamada previa fuera del reloj (compila @njit, llena cachés)."""
        self.funcion(imagen_inicial(32))


REGISTRO = {}


def registrar(nombre, paralela=False, fijar=None):
    def decorador(fn):
        REGISTRO[nombre] = Version(nombre, fn, paralela, fijar)
        return fn

    return decorador
