# NecLab

Software para el análisis automatizado de imágenes de microscopía y para la visualización/análisis de series de datos, desarrollado en el laboratorio LansBiodyt de la Facultad de Ciencias, UNAM.

📖 **Manual de usuario completo:** [docs/MANUAL_USUARIO.md](docs/MANUAL_USUARIO.md) (Español) · [docs/USER_MANUAL.md](docs/USER_MANUAL.md) (English)

---

## Descripción

NecLab es una aplicación de escritorio (Tkinter) con dos grandes flujos de trabajo:

1. **Procesamiento de imágenes de microscopía**: carga de pilas OME-TIFF, ajuste de brillo/contraste/threshold, 7 métodos de análisis de variabilidad temporal, detección de células por clustering espacial, y extracción/exportación de series temporales por célula.
2. **Visualización y análisis de datos numéricos**: carga de archivos `.npy`/`.csv` o de múltiples libros de Excel, detección de picos (7 métodos), suavizado, normalización, correlaciones (Pearson/Kendall/Spearman), dendrogramas de clustering jerárquico y exportación a CSV/XLSX.

Funcionalidades principales:

- Cargar y visualizar imágenes OME-TIFF, con navegación por frames
- Ajustar brillo, contraste y threshold de binarización
- Aplicar 7 métodos de análisis de variabilidad temporal (Rango, Varianzas, Desviaciones Estándar, Coeficiente de Variación, IQR)
- Detectar y segmentar células mediante clustering espacial (básico, avanzado y descomposición de clusters grandes)
- Extraer series temporales de células individuales y analizar sus correlaciones, con vista 3D de la superficie de variabilidad
- Cargar datos `.npy`/`.csv` individuales o **múltiples archivos Excel a la vez** ("Datos Múltiples"), con suavizado, 4 modos de normalización, mapa de calor y clasificación de hojas
- Detectar picos con 7 métodos distintos (Elliptic Envelope, Peak Caller, Local Outlier Factor, Isolation Forest, modelos lineales, etc.)
- Generar dendrogramas de clustering jerárquico y matrices de correlación, o cargar una matriz de correlación ya calculada
- Exportar resultados en CSV/XLSX (series de tiempo, picos, correlaciones, clasificaciones, clusters) y guardar imágenes de cada gráfica
- Revisar actualizaciones del código directamente desde el menú **Help → Check for Updates**

---

## Requisitos del sistema

- Sistema operativo: Windows, macOS o Linux
- Python 3.11 (recomendado; el proyecto se compila y prueba con 3.11)
- Miniconda o Anaconda (para la instalación recomendada)

---

## Instalación

Hay tres formas de obtener NecLab. Para la mayoría de los usuarios del laboratorio se recomienda la **Opción A** (ejecutable precompilado); para desarrollo o para plataformas sin ejecutable publicado, use la **Opción B** (conda).

### Opción A: Usar el ejecutable precompilado (sin instalar Python)

Cada vez que se publica una versión etiquetada (`v1.0`, `v1.1`, etc.) en GitHub, se generan automáticamente ejecutables para macOS y Windows mediante GitHub Actions.

1. Ir a la pestaña **Releases** del repositorio: https://github.com/SergioCruzUnamIA/NecLabUNAM/releases
2. Descargar `NecLab-mac.zip` (macOS) o `NecLab.exe` (Windows) de la versión más reciente
3. **macOS**: descomprimir y arrastrar `NecLab.app` a la carpeta Aplicaciones. Como la app no está firmada por Apple, la primera vez debe hacer clic derecho → "Abrir" (en vez de doble clic) y confirmar en el diálogo de seguridad
4. **Windows**: ejecutar `NecLab.exe` directamente. Si Windows Defender SmartScreen muestra una advertencia, seleccione "Más información" → "Ejecutar de todas formas"

Si no hay ninguna versión etiquetada reciente, también puede descargar el último build automático desde la pestaña **Actions** del repositorio (artefactos `NecLab-mac` / `NecLab-windows`), aunque estos requieren una cuenta de GitHub para descargarse.

### Opción B: Instalación con Conda (recomendada para desarrollo)

#### Paso 1: Instalar Miniconda

Si no tiene Miniconda instalado, descárguelo desde:
https://docs.conda.io/en/latest/miniconda.html

Siga las instrucciones de instalación para su sistema operativo.

#### Paso 2: Descargar el repositorio

Opción A - Clonar con Git:
```
git clone https://github.com/SergioCruzUnamIA/NecLabUNAM.git
```

Opción B - Descargar ZIP:
1. Ir a la página del repositorio en GitHub
2. Dar clic en el botón verde "Code"
3. Seleccionar "Download ZIP"
4. Extraer el archivo en la ubicación deseada

#### Paso 3: Crear el ambiente virtual

Abra una terminal y navegue a la carpeta del proyecto:
```
cd ruta/a/NecLabUNAM
```

Cree el ambiente virtual con las dependencias:
```
conda env create -f environment.yml
```

Este proceso descargará todas las bibliotecas necesarias (incluyendo `customtkinter`, instalado vía pip dentro del propio ambiente conda, ya que no está disponible en conda-forge). Puede tomar varios minutos dependiendo de su conexión a internet.

#### Paso 4: Activar el ambiente virtual

```
conda activate neclab_env
```

Una vez activado, verá `(neclab_env)` al inicio de la línea de comandos.

### Opción C: Instalación con pip / venv

`requirements.txt` es una captura completa (`pip freeze`) del ambiente de *desarrollo y empaquetado*, no solo de ejecución: además de las bibliotecas que usa la aplicación, incluye herramientas para compilar ejecutables (`pyinstaller`, `cx_Freeze`, `dmgbuild`, `mac-alias`) y el stack de Jupyter usado por los notebooks de prototipado. Instalarlo funciona, pero es más pesado de lo necesario solo para ejecutar la app.

```
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Ejecución

Con el ambiente virtual activado, ejecute:

```
python interface3.py
```

La primera ejecución puede tardar entre 1 y 2 minutos mientras se cargan las bibliotecas.

---

## Guía de uso rápida

Esta sección es solo un resumen. Para instrucciones detalladas paso a paso de cada menú, pestaña y función de exportación, consulte el **manual de usuario completo**: [docs/MANUAL_USUARIO.md](docs/MANUAL_USUARIO.md) (Español) o [docs/USER_MANUAL.md](docs/USER_MANUAL.md) (English).

La aplicación se organiza en pestañas:

- **Procesamiento de Imágenes**: abrir una imagen OME-TIFF (`Archivo → Abrir OME-TIFF`, o `Ctrl+O`), ajustar brillo/contraste/threshold, y desde el menú **Análisis de Variabilidad** elegir uno de los 7 métodos para abrir la ventana de detección y clustering de células.
- **Visualización de Datos**: abrir un archivo `.npy`/`.csv` (`Archivo → Abrir Datos`) para buscar picos, aplicar suavizado, calcular correlaciones y generar dendrogramas o series de tiempo.
- **Datos Múltiples** (aparece al usar `Archivo → Abrir Multiples Archivos (.xls)`): cargar y comparar varias hojas de Excel a la vez, con suavizado, normalización, mapa de calor y clasificación de hojas.
- **Dendograma** y **Time Series**: se crean automáticamente al usar las opciones correspondientes del menú **Visualizacion**, una vez que hay datos cargados.

También puede cargar una matriz de correlación ya calculada con `Archivo → Cargar Matriz de Correlacion`.

---

## Actualizaciones desde la aplicación

`Help → Check for Updates` compara la versión local contra el último commit de la rama `main` del repositorio en GitHub y, si hay una versión más reciente, ofrece descargar los archivos `.py` actualizados y reiniciar la aplicación automáticamente. Requiere conexión a internet.

`Help → About NecLab` muestra información de contacto y del repositorio.

---

## Estructura del proyecto

```
NecLabUNAM/
├── interface3.py               # Interfaz gráfica principal (punto de entrada)
├── variability_functions.py    # Métodos de variabilidad y clustering de células
├── peak_functions.py           # Detección de picos (7 métodos) y suavizado
├── corr_dendo_functions.py     # Correlaciones, dendrogramas y series de tiempo
├── multi_xls_helpers.py        # Carga y procesamiento para la pestaña Datos Múltiples
├── visualization_helpers.py    # Funciones auxiliares de visualización de datos
├── image_loader.py             # Carga de imágenes OME-TIFF
├── image_processing.py         # Procesamiento de imágenes
├── NecLab.spec                 # Especificación de PyInstaller (ejecutables Mac/Windows)
├── build_mac.sh                # Script para compilar NecLab.app localmente
├── build_windows.bat           # Script para compilar NecLab.exe localmente
├── .github/workflows/build.yml # Compilación automática de ejecutables (GitHub Actions)
├── environment.yml             # Dependencias del proyecto (conda)
├── requirements.txt            # Dependencias completas de desarrollo/empaquetado (pip)
├── cell_detection_complete.ipynb  # Notebook de prototipado (no forma parte de la app)
├── signal_processing.ipynb        # Notebook de prototipado (no forma parte de la app)
├── docs/
│   ├── MANUAL_USUARIO.md       # Manual de usuario completo (Español)
│   └── USER_MANUAL.md          # Full user manual (English)
└── README.md                   # Este archivo
```

---

## Compilar ejecutables standalone

Los ejecutables se generan con PyInstaller a partir de `NecLab.spec`, que produce un `.app` en macOS o un `.exe` en Windows según la plataforma donde se ejecute.

**Localmente:**
```
pip install pyinstaller
./build_mac.sh          # macOS → dist/NecLab.app
build_windows.bat       # Windows → dist\NecLab.exe
```

**Automáticamente (GitHub Actions):** el workflow `.github/workflows/build.yml` compila ambos ejecutables al hacer push de un tag `v*` (y los publica en la Release correspondiente), o al hacer push a las ramas de desarrollo configuradas en ese archivo. También puede lanzarse manualmente desde la pestaña Actions ("Run workflow").

---

## Solución de problemas

**El programa no inicia / `ModuleNotFoundError: No module named 'customtkinter'`:**
- Verifique que el ambiente virtual esté activado (debe ver `(neclab_env)` en la terminal)
- Si instaló con conda antes de esta corrección, `customtkinter` no se instala vía `conda install` porque no existe en conda-forge; ejecute `pip install customtkinter` dentro del ambiente activado, o recree el ambiente con `conda env create -f environment.yml`

**Error de biblioteca no encontrada:**
- Ejecute: `pip install nombre_de_la_biblioteca`
- O reinstale el ambiente: `conda env remove -n neclab_env` y repita la instalación

**La imagen no se visualiza correctamente:**
- Verifique que el archivo sea formato OME-TIFF válido
- Intente ajustar el brillo y contraste en el panel de controles

**"Check for Updates" no encuentra nada / falla:**
- Requiere conexión a internet y acceso a `api.github.com`; si su red bloquea GitHub, actualice manualmente con `git pull` o descargando un nuevo ZIP/ejecutable

**macOS dice que la app está dañada o de un desarrollador no identificado:**
- La app no está firmada ni notarizada por Apple. Use clic derecho → "Abrir" en vez de doble clic la primera vez

**Cargar muchos archivos Excel en "Datos Múltiples" es lento:**
- La carga y el guardado corren en un hilo en segundo plano con una ventana de progreso, pero archivos muy grandes o con muchas hojas seguirán tardando; espere a que la ventana de progreso se cierre antes de interactuar con la pestaña

---

## Datos de prueba

Para probar el software puede descargar una imagen de ejemplo desde:
https://drive.google.com/file/d/1EP7TQMWQglbhgoRdJm2bY-10s2cYMDa3/view

---

## Créditos

Desarrollado en el laboratorio LansBiodyt, Facultad de Ciencias, UNAM.

Asesor: Dr. Sergio Rodolfo Cruz Gómez

---

## Licencia

Este proyecto está bajo la licencia MIT incluida en el archivo [LICENSE](LICENSE).
