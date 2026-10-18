from convolution2d.common import imagen_inicial
from convolution2d.versions.v0_serial import convolucion2d_v0_serial

img = imagen_inicial(256)
perfilada = profile(convolucion2d_v0_serial)  # noqa: F821
perfilada(img)
