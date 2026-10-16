"""v0: línea base serial en Python puro (bucles anidados)."""
import numpy as np

from ..params import FACTOR, RADIO
from ..registry import registrar


@registrar("v0_serial")
def convolucion2d_v0_serial(img):
    """
    Filtro de desenfoque de caja (Box Blur) 5x5 con bucles anidados.
    Solo se calcula el interior (borde de RADIO píxeles); el borde de la salida queda en 0.
    """
    alto, ancho = img.shape
    salida = np.zeros_like(img)

    for i in range(RADIO, alto - RADIO):
        for j in range(RADIO, ancho - RADIO):
            suma = 0.0
            for ki in range(-RADIO, RADIO + 1):
                for kj in range(-RADIO, RADIO + 1):
                    suma += img[i + ki, j + kj]
            salida[i, j] = suma * FACTOR

    return salida


if __name__ == "__main__":
    from ..benchmark import main
    main(["--solo", "v0_serial"])