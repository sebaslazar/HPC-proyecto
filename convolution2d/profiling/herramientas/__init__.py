"""Cada módulo de esta carpeta registra un perfilador."""
import importlib
import pkgutil


def descubrir():
    for info in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{info.name}")