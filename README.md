# Convolución 2D 5x5 (Box Blur) — Proyecto HPC

Kernel de desenfoque de caja 5x5 con varias implementaciones (serial, NumPy, Numba `prange`, …),
un harness de medición y un conjunto de perfiladores (cProfile, py-spy, Scalene).

Todos los comandos se ejecutan **desde la raíz del proyecto** (la carpeta que contiene `convolution2d/`),
con el entorno virtual activado:

```bash
source .venv/bin/activate
```

## Estructura

```
proyecto/
├── results/                          # JSON de benchmark y TXT de profiling
└── convolution2d/
    ├── params.py                     # TODOS los parámetros (tamaños, repeticiones, tasas…)
    ├── common.py                     # imagen_inicial(n)
    ├── registry.py                   # registro de versiones + autodescubrimiento
    ├── entorno.py                    # declaración de la máquina
    ├── benchmark.py                  # harness de medición
    ├── versions/                     # UNA implementación por archivo
    │   ├── v0_serial.py
    │   └── ...
    └── profiling/
        ├── __main__.py               # CLI de profiling
        ├── base.py                   # contrato Perfilador
        ├── objetivo.py               # programa que perfilan las herramientas
        └── herramientas/             # una herramienta por archivo
            ├── perf_cprofile.py
            ├── perf_pyspy.py
            └── perf_scalene.py
```

## Instalación

```bash
uv sync
```

---

## 1. Benchmark

Mide las versiones registradas siguiendo las reglas del curso: `perf_counter`, 5 repeticiones y
mediana, datos preparados fuera del reloj, calentamiento previo, máquina declarada en el JSON.

```bash
uv run python -m convolution2d.benchmark --info
uv run python -m convolution2d.benchmark --solo v0_serial
uv run python -m convolution2d.benchmark --solo v1_numpy,v2_prange
uv run python -m convolution2d.benchmark
```

| Comando / flag | Qué hace |
|---|---|
| `--info` | Imprime la descripción de la máquina (CPU, núcleos, memoria, versiones de librerías) y sale. |
| `--solo a,b` | Mide solo esas versiones. Sin este flag mide todas las que haya en `versions/`. |
| `--n 512` | Tamaño de la imagen NxN. Por defecto `params.N_BENCH`. |
| `--repeticiones 5` | Repeticiones por versión. Menos de 5 produce un aviso: el resultado no es entregable. |
| `--trabajadores-max 8` | Tope de hilos para el barrido de versiones paralelas (1, 2, 4, 8, … hasta el tope). |
| `--sin-verificar` | Omite la comprobación de correctitud contra `v0_serial`. |
| `--sin-escribir` | No escribe el JSON (solo imprime en pantalla). |

**Salida:** `results/<version>.json`, con las claves `maquina`, `comando` y `mediciones`.

**Speedup:** se calcula contra `v0_serial`. Si no lo mides en la misma corrida, se lee de
`results/v0_serial.json`, pero solo si fue medido con el mismo `--n`. Por eso conviene medir primero la
línea base:

```bash
uv run python -m convolution2d.benchmark --solo v0_serial
uv run python -m convolution2d.benchmark --solo v2_prange
```

---

## 2. Profiling con las herramientas del proyecto

```bash
uv run python -m convolution2d.profiling --listar
uv run python -m convolution2d.profiling --version v0_serial
uv run python -m convolution2d.profiling --version v0_serial --herramienta cprofile,scalene
uv run python -m convolution2d.profiling --version v0_serial --herramienta pyspy --sudo
uv run python -m convolution2d.profiling --version todas --repeticiones 5
uv run python -m convolution2d.profiling --version v2_prange --trabajadores 4
```

| Flag | Qué hace |
|---|---|
| `--listar` | Muestra versiones y herramientas disponibles (y el motivo si alguna no lo está). |
| `--version v0 v1 …` | Versiones a perfilar; `todas` las incluye todas. |
| `--herramienta a,b` | `cprofile`, `pyspy`, `scalene` o `todas` (por defecto). |
| `--n 1024` | Tamaño de la imagen. Por defecto `params.N_PROFILING`. |
| `--repeticiones 5` | Llamadas al kernel dentro del perfilador. Súbelo con versiones rápidas. |
| `--trabajadores 4` | Hilos para versiones paralelas; añade `_p4` al nombre del archivo. |
| `--sudo` | Ejecuta py-spy con `sudo` (necesario en macOS; a veces en Linux). |

**Salida:** `results/<version>__<herramienta>.txt`

```
results/v0_serial__cprofile.txt
results/v0_serial__pyspy.txt
results/v0_serial__scalene.txt
results/v2_prange_p4__scalene.txt
```

cProfile y Scalene llevan un encabezado con la versión, el tamaño, la máquina y el comando exacto.
El de py-spy no, porque su formato `raw` no admite comentarios; sus parámetros están en el nombre del archivo.

---

## 3. Usar las herramientas directamente desde la terminal

No necesitas el CLI del proyecto para perfilar. Las tres herramientas perfilan el mismo programa:
`convolution2d/profiling/objetivo.py`, que prepara los datos, calienta y llama al kernel.

```bash
python convolution2d/profiling/objetivo.py --version v0_serial --n 1024 --repeticiones 5
```

Opciones de `objetivo.py`: `--version` (obligatoria), `--n`, `--repeticiones`, `--trabajadores`.

### 3.1 cProfile puro

Perfilador determinista de la biblioteca estándar: cuenta cada llamada a función.

```bash
# 1) Guardar el perfil binario
uv run python -m cProfile -o results/v0_serial__cprofile.prof \
    convolution2d/profiling/objetivo.py --version v0_serial --n 1024 --repeticiones 5

# 2) Convertirlo a texto ordenado por tiempo acumulado
uv run python -c "import pstats; pstats.Stats('results/v0_serial__cprofile.prof').strip_dirs().sort_stats('tottime').print_stats(40)" \
    | tee results/v0_serial__cprofile.txt

# 3) O explorarlo de forma interactiva
uv run python -m pstats results/v0_serial__cprofile.prof
```

Atajo (mezcla la salida del programa con la del perfil):

```bash
uv run python -m cProfile -s cumulative convolution2d/profiling/objetivo.py --version v0_serial > salida.txt
```

Claves de ordenamiento útiles: `cumulative` (incluye subllamadas), `tottime` (solo la función), `calls`.

Limitación: cProfile no ve dentro de un bucle sin llamadas a función. Sobre `v0_serial` verás casi todo
el tiempo en una sola fila. Una función `@njit` de Numba también aparece como una única llamada.
Además, este modo perfila también los imports y la preparación de datos.

### 3.2 py-spy

Perfilador de **muestreo** externo: observa el proceso desde fuera, con muy poco overhead.

```bash
# Registrar en formato texto (pilas colapsadas)
py-spy record --format raw --rate 200 \
    --output results/v0_serial__pyspy.txt \
    -- python convolution2d/profiling/objetivo.py --version v0_serial --n 1024 --repeticiones 5

# macOS (y a veces Linux): privilegios
sudo py-spy record --format raw --rate 200 --output results/v0_serial__pyspy.txt \
    -- python convolution2d/profiling/objetivo.py --version v0_serial --n 1024 --repeticiones 5
```

Otras formas útiles:

```bash
py-spy record --format speedscope -o perfil.json -- python …   # abrir en https://www.speedscope.app
py-spy record -o flame.svg -- python …                          # flamegraph SVG (formato por defecto)
py-spy top  --pid <PID>                                         # vista en vivo tipo "top"
py-spy dump --pid <PID>                                         # instantánea de las pilas
```

Flags adicionales: `--native` (incluye frames de código nativo), `--subprocesses`, `--threads`.

El formato `raw` son líneas `pila;pila;pila N` (N = número de muestras). Se puede convertir a SVG
con `flamegraph.pl` (proyecto FlameGraph).

Notas:
- Si la corrida dura menos de ~1 s hay pocas muestras. Aumenta `--repeticiones` o `--n`.
- En Linux, si falla por permisos: `sudo sysctl kernel.yama.ptrace_scope=0` o usa `sudo`.
  En Docker: `--cap-add=SYS_PTRACE`.

### 3.3 Scalene

Perfilador de muestreo **por línea**: CPU (Python vs. nativo), memoria y copias.

Con Scalene 2.x (`scalene run` / `scalene view`):

```bash
# 1) Perfilar (escribe scalene-profile.json en el directorio actual)
uv run python -m scalene run --program-path convolution2d --cpu-only \
    convolution2d/profiling/objetivo.py --- --version v0_serial --n 1024 --repeticiones 5

# 2) Ver el reporte en terminal
uv run python -m scalene view --cli

# 3) O verlo en el navegador
uv run python -m scalene view
```

Con Scalene 1.x:

```bash
uv run python -m scalene --cli --program-path convolution2d --cpu-only \
    convolution2d/profiling/objetivo.py --- --version v0_serial --n 1024 --repeticiones 5 \
    > results/v0_serial__scalene.txt
```

Puntos clave:
- `---` separa las opciones de Scalene de los argumentos del programa perfilado.
- **`--program-path convolution2d` es imprescindible.** Por defecto Scalene solo perfila los archivos que
  están dentro de la carpeta del script (`convolution2d/profiling/`) o en su carpeta padre. El kernel vive
  en `convolution2d/versions/`, que queda fuera: su tiempo no se reparte por líneas, sino que se
  atribuye a la línea de `objetivo.py` que lo llama, y el reporte sale inservible. (`--profile-only` no
  lo arregla: solo restringe más, nunca amplía.)
- `--cpu-only` desactiva el perfilado de memoria. Sin él, el `import numpy` infla el reporte (en una
  corrida de prueba llegó a ser el 68 % del tiempo) y el reporte pierde la señal del kernel. Las columnas
  de CPU (Python / native / system) se conservan.
- Cómo comprobar que salió bien: el archivo `v0_serial.py` debe ser el que más `% of time` tenga, y las
  líneas del bucle interno (`suma += img[i + ki, j + kj]`) deben traer números en las columnas de CPU.
- Para ver las versiones exactas de las opciones: `uv run python -m scalene run --help-advanced`.
- Mira la versión instalada con `pip show scalene`.

---

## 4. Cómo se lee cada reporte

| Herramienta | Qué responde | Dónde mirar |
|---|---|---|
| cProfile | ¿Qué función consume el tiempo? | `cumtime` y `tottime` de cada fila |
| py-spy | ¿En qué pilas pasa el tiempo realmente? | Pilas con N más alto (último número de cada línea) |
| Scalene | ¿Qué **línea** pesa y si el tiempo es Python o nativo? | Columnas "Python %" / "native %" por línea |

---

## 5. Extender el proyecto

**Nueva versión:** crea `convolution2d/versions/v1_numpy.py` con una función `f(img) -> np.ndarray`
decorada con `@registrar("v1_numpy")`. No hay que editar ninguna lista: se descubre sola, y ya funciona
con `benchmark` y con `profiling`. Para versiones paralelas:
`@registrar("v2_prange", paralela=True, fijar=funcion_que_fija_hilos)`.

**Nueva herramienta de profiling:** crea `convolution2d/profiling/herramientas/perf_<nombre>.py` con una
subclase de `Perfilador` decorada con `@perfilador` e implementa `ejecutar(sol, destino)`.

**Parámetros:** todo se cambia en `convolution2d/params.py`.

---

## 6. Problemas frecuentes

**`No module named convolution2d.profiling.__main__; 'convolution2d.profiling' is a package and cannot be directly executed`**
Falta el archivo `convolution2d/profiling/__main__.py` o tiene otro nombre (`main.py`, `_main_.py`…).
El nombre exacto lleva **dos** guiones bajos a cada lado. Comprueba con:

```bash
ls convolution2d/profiling/
```

**`attempted relative import with no known parent package`**
Ejecutaste un módulo como script. Usa siempre `uv run python -m convolution2d.…` desde la raíz del proyecto
(la excepción es `objetivo.py`, que sí está preparado para ejecutarse como script).

**`No module named convolution2d`**
No estás en la raíz del proyecto, o falta `convolution2d/__init__.py`.

**py-spy: `Permission denied` / `Operation not permitted`**
Repite con `--sudo` (o con `sudo` si lo ejecutas a mano). Ver las notas de la sección 3.2.

**Scalene no muestra líneas del kernel (el archivo de `versions/` sale con 0 %)**
Falta `--program-path convolution2d`: sin él, Scalene no perfila `versions/` y atribuye todo el tiempo
a la línea de `objetivo.py` que llama al kernel. Si aun así sale vacío, la corrida es demasiado corta:
sube `--repeticiones` o `--n`.

**`SALTADO` en una versión**
Falta una dependencia opcional (por ejemplo Numba). El motivo se imprime junto al aviso;
`uv run python -m convolution2d.profiling --listar` lo muestra también.