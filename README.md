# Convolución 2D 5x5 (Box Blur) — Proyecto HPC

Proyecto final de Computación de Alto Desempeño (UTP 2026-2). El kernel es un filtro de
desenfoque de caja 5×5 sobre una imagen `float64` de N×N: cada píxel interior se reemplaza por
la media de su ventana de 25 vecinos, y el borde de 2 píxeles de la salida queda en 0,0
(Dirichlet). El interior es `[2, N-2)`.

Todos los comandos se ejecutan **desde la raíz del proyecto** (la carpeta que contiene
`reproducir.sh`).

| Pieza | Archivo | Qué es |
|---|---|---|
| Referencia | `src/referencia.py` | Modelo independiente en NumPy (`sliding_window_view`). Es el árbitro de la correctitud. |
| v0 | `src/v0_serial.py` | Línea base en Python puro, cuatro bucles anidados. |
| v1 | `src/v1_prange.py` | Memoria compartida con Numba: `@njit(parallel=True, cache=True)` y `prange` sobre las filas de salida. |

## Autores y roles

| Integrante | Cuenta de Git | Propuesta y Taller 4 | Entrega final (roles rotados) |
|---|---|---|---|
| Sebastián Cacante Salazar | `sebaslazar` | Implementador | Medidor |
| Jhosuar Andrés Suárez | `Jhosuar1` | Medidor | Implementador |

Cada integrante registra sus cambios con commits desde su propia cuenta.

## Requisitos

- [`uv`](https://docs.astral.sh/uv/). Es lo único que hay que instalar a mano: crea el entorno
  exacto a partir de `uv.lock` (Python 3.12, fijado en `.python-version`).
- `bash` (Linux o macOS).

---

## 1. Cómo reproducir todo

Desde la raíz del proyecto, un solo comando:

```bash
bash reproducir.sh
```

Hace, sin intervención manual, tres cosas y se detiene con error si alguna falla:

1. **Entorno exacto**: `uv sync --frozen` instala las versiones fijadas en `uv.lock`.
2. **Correctitud**: `uv run --frozen python test_correctitud.py` verifica `v0_serial` y
   `v1_prange` (este con 1, 2 y 4 hilos; entradas sintética y aleatoria, N=64) contra
   `src/referencia.py`. Imprime `PASS` o `FAIL` por prueba y sale con código 0 solo si todas
   pasan.
3. **Mediciones**: `uv run --frozen python bench.py` mide ambas versiones y regenera
   `resultados/v0_serial.json` y `resultados/v1_prange.json`.

---

## 2. Ejecutar por partes

```bash
uv run python bench.py --info                  # solo describe la máquina
uv run python test_correctitud.py              # solo la correctitud
uv run python bench.py --solo v0_serial        # solo la línea base
uv run python bench.py --solo v1_prange        # solo la versión paralela (usa el v0 guardado)
uv run python bench.py --n 512                 # otro tamaño (el speedup exige el mismo N)
```

| Flag | Qué hace |
|---|---|
| `--info` | Imprime la descripción de la máquina (JSON) y sale. |
| `--solo a,b` | Mide solo esas versiones, separadas por comas. Sin él se miden las dos actuales. |
| `--n N` | Tamaño de la imagen NxN. Por defecto 1024 (`params.N_BENCH`). |
| `--repeticiones k` | Repeticiones por versión. Por defecto 5, mínimo del curso. |
| `--sin-escribir` | No escribe el JSON: solo imprime en pantalla. |

**Salida:** `resultados/<version>.json`, con las claves `maquina`, `comando` y `mediciones`.

**Speedup:** se calcula contra `v0_serial`. Si no lo mides en la misma corrida, se lee de
`resultados/v0_serial.json`, pero solo si fue medido con el mismo `--n`.

---

## 3. Metodología de medición

- **Reloj:** `time.perf_counter()` (monótono), nunca `time.time()`.
- **Estadística:** 5 repeticiones y **mediana**; también se guardan mínimo, máximo y desviación.
  Se hace una corrida adicional que se descarta (cachés en frío).
- **Preparación:** la creación de la imagen va fuera del reloj.
- **Calentamiento:** antes de medir se calienta con una imagen de 32×32 (compila el JIT de
  Numba). En `v1_prange` se vuelve a calentar tras fijar cada número de hilos.
- **Verificación:** antes de medir cada versión se comprueba su resultado contra la referencia.
- **Máquina:** se declara sola en cada JSON (`src/entorno.py`): CPU, núcleos físicos y lógicos,
  memoria y versiones de las librerías.
- **Hilos:** barrido de `v1_prange` con p ∈ {1, 2, 4}. Si la máquina permite menos hilos, se
  omiten los que no caben y se avisa.

---

## 4. Formato de `resultados/*.json`

Claves de primer nivel: `maquina`, `comando` y `mediciones`. Cada medición lleva `clave`,
`valor`, `unidad`, `repeticiones` y `trabajadores` (y `minimo`, `maximo`, `desviacion` en los
tiempos).

| Clave | Significado |
|---|---|
| `convolution2d/v0_serial/tiempo` | Mediana en segundos de la línea base. |
| `convolution2d/v0_serial/speedup` | 1,0 por definición. |
| `convolution2d/v1_prange/tiempo_p{p}` | Mediana en segundos con `p` hilos. |
| `convolution2d/v1_prange/speedup_p{p}` | Speedup absoluto: mediana de `v0_serial` / mediana con `p` hilos. |
| `convolution2d/v1_prange/speedup_relativo_p{p}` | Speedup relativo: mediana de `v1_prange` con 1 hilo / mediana con `p` hilos. |
| `convolution2d/v1_prange/speedup` | El `speedup_p` del mayor número de hilos medido. |

El tamaño del problema queda registrado en el campo `comando` (`--n 1024`).
El speedup absoluto solo se calcula si la línea base se midió con el mismo N.

---

## 5. Estructura

```
.
├── README.md
├── .python-version        # fija Python 3.12
├── pyproject.toml         # entorno del curso; uv.lock fija las versiones exactas
├── uv.lock
├── reproducir.sh          # entorno + correctitud + resultados, con un comando
├── bench.py               # harness de medición
├── test_correctitud.py    # PASS/FAIL contra la referencia
├── src/
│   ├── __init__.py
│   ├── params.py          # parámetros únicos
│   ├── common.py          # imagen_inicial(n)
│   ├── entorno.py         # metadatos de la máquina
│   ├── referencia.py      # modelo de referencia (NumPy)
│   ├── v0_serial.py
│   └── v1_prange.py
└── resultados/
    ├── v0_serial.json     # los genera bench.py
    ├── v1_prange.json
    └── perfil_taller4/    # evidencia del perfil previo (Taller 4)
        ├── v0_serial__cprofile.txt
        └── v0_serial__lineas.txt
```
