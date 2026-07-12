# Manual de Usuario — NecLab

Manual de usuario completo de NecLab, la herramienta de análisis de imágenes de microscopía y visualización de datos del laboratorio LansBiodyt, Facultad de Ciencias, UNAM.

Para instrucciones de instalación vea el [README](../README.md). Este manual asume que ya tiene la aplicación instalada y corriendo (`python interface3.py` o el ejecutable precompilado).

> 🇬🇧 English version: [USER_MANUAL.md](USER_MANUAL.md)

---

## Tabla de contenidos

1. [Visión general de la aplicación](#1-visión-general-de-la-aplicación)
2. [Barra de menús](#2-barra-de-menús)
3. [Pestaña: Procesamiento de Imágenes](#3-pestaña-procesamiento-de-imágenes)
4. [Análisis de Variabilidad y detección de células](#4-análisis-de-variabilidad-y-detección-de-células)
5. [Pestaña: Visualización de Datos](#5-pestaña-visualización-de-datos)
6. [Pestaña: Datos Múltiples](#6-pestaña-datos-múltiples)
7. [Pestaña: Dendograma](#7-pestaña-dendograma)
8. [Pestaña: Time Series (Series de Tiempo)](#8-pestaña-time-series-series-de-tiempo)
9. [Cargar una matriz de correlación precalculada](#9-cargar-una-matriz-de-correlación-precalculada)
10. [Atajos de teclado](#10-atajos-de-teclado)
11. [Actualizaciones automáticas](#11-actualizaciones-automáticas)
12. [Glosario de métodos](#12-glosario-de-métodos)
13. [Solución de problemas](#13-solución-de-problemas)

---

## 1. Visión general de la aplicación

NecLab tiene dos flujos de trabajo principales, organizados en pestañas dentro de una sola ventana:

- **Procesamiento de imágenes**: cargar una pila OME-TIFF, ajustar su visualización y analizar la variabilidad temporal de cada píxel para detectar y segmentar células como clusters espaciales.
- **Visualización y análisis de datos**: cargar series numéricas (un archivo `.npy`/`.csv`, o varios archivos Excel a la vez) para detectar picos, suavizar señales, normalizar, calcular correlaciones y construir dendrogramas.

Ambos flujos comparten la misma ventana principal, que tiene una barra de menús arriba y un cuaderno de pestañas (`Notebook`) debajo. Algunas pestañas están presentes desde el inicio (Procesamiento de Imágenes, Visualización de Datos); otras se crean automáticamente la primera vez que se usa la función correspondiente (Datos Múltiples, Dendograma, Time Series).

---

## 2. Barra de menús

### Archivo

| Opción | Acción |
|---|---|
| **Abrir OME-TIFF** (`Ctrl+O`) | Carga una pila de imágenes OME-TIFF, cambia a la pestaña "Procesamiento de Imágenes" y habilita los menús Imagen y Análisis de Variabilidad. |
| **Abrir Datos (.npy / .csv)** | Carga un archivo de datos numéricos en la pestaña "Visualización de Datos". Si es CSV, se muestra un diálogo para elegir la columna inicial (ROI); si es `.npy`, las columnas se nombran automáticamente "Column 1", "Column 2", etc. |
| **Cargar Matriz de Correlacion** | Carga una matriz de correlación ya calculada (CSV/XLSX, debe ser cuadrada) y la muestra como dendrograma + mapa de calor. Ver [sección 9](#9-cargar-una-matriz-de-correlación-precalculada). |
| **Abrir Multiples Archivos (.xls)** | Abre el flujo de "Datos Múltiples": selección de varios archivos Excel y de las hojas a cargar de cada uno. Ver [sección 6](#6-pestaña-datos-múltiples). |
| **Salir** | Cierra la aplicación (pide confirmación y cierra todas las ventanas de gráficas abiertas). |

### Imagen

Deshabilitado hasta cargar una imagen OME-TIFF.

| Opción | Acción |
|---|---|
| **Auto Contraste** | Aplica autocontraste automático a cada frame de la pila. |
| **Histogram** | Muestra un histograma de la varianza por píxel a lo largo de toda la pila, en una ventana aparte. |
| **Binarize** | Abre un diálogo de umbral (threshold) y binariza la imagen de varianza. |
| **Restaurar Original** | Revierte cualquier ajuste y vuelve a la imagen original tal como se cargó. |

### Análisis de Variabilidad

Deshabilitado hasta cargar una imagen. Se llena dinámicamente con los 7 métodos de variabilidad (ver [sección 4](#4-análisis-de-variabilidad-y-detección-de-células)). Cada opción abre una ventana completa "Análisis Completo — &lt;método&gt;" para ese método.

### Visualizacion

| Opción | Acción |
|---|---|
| **Dendograma** | Deshabilitado hasta cargar datos con `Archivo → Abrir Datos`. Una vez cargados, abre/activa la pestaña permanente "Dendograma" (sección 7). |
| **Series de tiempo** | Deshabilitado hasta cargar datos con `Archivo → Abrir Datos`. Una vez cargados, abre la pestaña "Time Series" (sección 8). |

> **Nota:** estas dos opciones solo se activan si los datos se cargaron con `Archivo → Abrir Datos (.npy / .csv)`. Si solo usó "Abrir Multiples Archivos (.xls)" o "Cargar Matriz de Correlacion", estas opciones permanecerán deshabilitadas — use los flujos propios de esas pestañas para dendrogramas y correlaciones.

### Help

| Opción | Acción |
|---|---|
| **Check for Updates** | Compara la versión local contra el último commit de `main` en GitHub y ofrece actualizar los archivos `.py` de la aplicación. Ver [sección 11](#11-actualizaciones-automáticas). |
| **About NecLab** | Muestra un cuadro con el nombre de la aplicación, el repositorio y el correo de contacto (sergio.cruz@ciencias.unam.mx). |

---

## 3. Pestaña: Procesamiento de Imágenes

Pestaña visible desde el inicio. Se compone de un panel de imagen (izquierda) y un panel de controles (derecha):

**NAVEGACIÓN**
- **Capa (Frame)**: control deslizante para recorrer los frames/slices de la pila cargada. Muestra "Frame: n / N".

**AJUSTES DE IMAGEN**
- **Brillo** y **Contraste**: controles deslizantes de -100 a 100.
- **Auto Contraste**: aplica autocontraste automáticamente.
- **Resetear Ajustes**: regresa brillo/contraste a sus valores por defecto.

**PROCESAMIENTO**
- **Threshold**: casilla "Aplicar" para activar la binarización y un control deslizante de 0 a 255 para el valor del umbral.

**INFORMACIÓN**
- Panel de solo lectura con las dimensiones, número de frames y tipo de dato de la imagen cargada.

### Flujo típico

1. `Archivo → Abrir OME-TIFF` (o `Ctrl+O`) y seleccione el archivo.
2. Recorra los frames con el control "Capa" para inspeccionar la pila.
3. Ajuste brillo/contraste, o use "Auto Contraste", para facilitar la inspección visual.
4. Opcionalmente aplique un threshold desde el menú Imagen o el panel de Procesamiento.
5. Pase al menú **Análisis de Variabilidad** para iniciar la detección de células.

---

## 4. Análisis de Variabilidad y detección de células

Al elegir un método en el menú **Análisis de Variabilidad**, se abre una ventana nueva de 1400×800, "Análisis Completo — &lt;método&gt;", con su propia barra de menús.

### 4.1 Métodos de variabilidad disponibles

| # | Método | Descripción | Umbral por defecto |
|---|---|---|---|
| 1 | Rango | Máximo − mínimo por píxel a través del tiempo | 100 |
| 2 | Varianza Poblacional | `np.var` con `ddof=0` | 120 |
| 3 | Varianza Muestral | `np.var` con `ddof=1` | 200 |
| 4 | Desviación Estándar Poblacional | `ddof=0` | 12 |
| 5 | Desviación Estándar Muestral | `ddof=1` | 5 |
| 6 | Coeficiente de Variación | `std(ddof=1) / media × 100` | 5 |
| 7 | Rango Intercuartílico (IQR) | Q3 − Q1 | 20 |

### 4.2 Controles de la ventana de análisis

- **Threshold** (1–1000): umbral de binarización de la imagen de variabilidad; empieza en el valor por defecto del método elegido.
- **Min Size** / **Max Size** (1–500 / 1–1000): tamaño mínimo y máximo (en píxeles) de los clusters a conservar.

### 4.3 Menú Clustering

| Opción | Acción |
|---|---|
| **Procesar Cluster (Básico)** | Detecta clusters espaciales usando solo el threshold. |
| **Procesar Cluster (Avanzado)** | Igual que el básico, pero filtra por Min Size / Max Size. |
| **Descomponer Clusters Grandes** | Divide clusters que excedan Max Size en sub-clusters, respetando Min Size/Max Size. |

### 4.4 Selección de clusters

- Haga clic sobre los clusters en el gráfico para seleccionarlos (se resaltan en rojo).
- **Selección por Región**: mantenga el botón derecho y arrastre sobre el gráfico para seleccionar/quitar varios clusters a la vez dentro de un rectángulo; use el conmutador Añadir/Quitar para elegir el modo.
- Menú **Selección**: `Seleccionar Todos`, `Limpiar Selección`.
- La lista de clusters seleccionados aparece en el panel lateral, con un botón "Remover" por cada uno.

### 4.5 Visualización 3D

Menú **Visualización → Vista 3D**: abre una superficie 3D (matplotlib) de la imagen de variabilidad, que puede rotarse con el mouse.

### 4.6 Exportar

Menú **Exportar**:
- **Guardar Imagen**: exporta la imagen del análisis actual.
- **Guardar .npy**: exporta las series temporales promedio de los clusters seleccionados como un arreglo `(frames, 1 + n_clusters)`, donde la primera columna es el índice de frame.
- **Usar Seleccionados (Correlaciones)**: abre la ventana "Análisis de Correlaciones — Clusters Seleccionados", con botones para calcular correlación Pearson, Kendall o Spearman entre las series de los clusters elegidos.

---

## 5. Pestaña: Visualización de Datos

Pestaña visible desde el inicio, pero solo se activa completamente al cargar datos con `Archivo → Abrir Datos (.npy / .csv)`.

### 5.1 Menú local de la pestaña

En la parte superior de la pestaña hay una barra de menú propia:

- **Vista**
  - `Smoothing`: activa/desactiva el suavizado por envolvente convexa sobre la señal.
  - `Show points`: muestra/oculta los puntos de la señal.
  - `Show Labels (Correlación)`: muestra/oculta las etiquetas de los ejes en el mapa de calor de correlación.
- **Guardar**
  - `Save Data Image...`: guarda la imagen de la gráfica de datos.
  - `Save Correlation...`: guarda la imagen del mapa de calor de correlación.
  - `Save Correlation Data...`: exporta la matriz de correlación (CSV/XLSX).
  - `Save Peaks CSV...`: exporta los picos detectados (ver 5.4).

### 5.2 Panel lateral

- **COLUMNAS DE DATOS**: lista (selección múltiple) con todas las columnas del archivo cargado.
- **PEAK FINDER**: combobox con los 7 métodos de detección de picos (ver [glosario](#12-glosario-de-métodos)) y un spinbox "Points:" (2–50) para el número de puntos usados por el suavizado de envolvente convexa.
- **CORRELACIÓN**: combobox para elegir el método (`pearson`, `kendall`, `spearman`).
- **SELECCIÓN**: lista de selección única, con botones "Add to Selection" / "Remove from Selection". Se necesitan al menos 2 columnas en la selección para dibujar el mapa de calor de correlación.

### 5.3 Área de gráficas

Al hacer clic en una columna, se construyen tres paneles apilados:

1. **Superior**: la señal original de la columna elegida, con marcadores de picos (si hay un método de detección activo) y la línea base de suavizado (si "Smoothing" está activado).
2. **Medio**: vista "procesada" — la señal suavizada sola, o el diagnóstico propio del método de detección de picos elegido.
3. **Inferior**: mapa de calor de correlación (Pearson/Kendall/Spearman) de las columnas en la lista de Selección.

### 5.4 Exportar picos detectados

`Guardar → Save Peaks CSV...` ejecuta el método de detección de picos activo sobre cada columna en la Selección y guarda un CSV con una columna `TIME` y una columna 0/1 por cada columna seleccionada (1 = pico detectado en ese instante). Requiere haber ejecutado antes un método de detección y tener al menos una columna en Selección.

---

## 6. Pestaña: Datos Múltiples

Se crea la primera vez que usa `Archivo → Abrir Multiples Archivos (.xls)`. Permite cargar y comparar varias hojas de uno o más archivos Excel a la vez.

### 6.1 Cargar archivos

1. `Archivo → Abrir Multiples Archivos (.xls)`.
2. Seleccione uno o más archivos Excel en el diálogo.
3. Para cada archivo, aparecerá un diálogo "Seleccionar Hojas a Cargar" con casillas por hoja (más los botones "Seleccionar Todo"/"Deseleccionar Todo").
4. Las hojas se leen **sin fila de encabezado** (las columnas se identifican solo por posición: Column 1, Column 2, ...). Solo se muestran las columnas que existen en **todas** las hojas cargadas.
5. La carga corre en segundo plano con una ventana de progreso; espere a que se cierre antes de interactuar con la pestaña.

### 6.2 Menú local de la pestaña

- **Vista**
  - `Mostrar Nombres de Datos` (por defecto apagado): muestra/oculta las etiquetas de hoja en el eje X de ambas gráficas.
  - `Smoothing` (por defecto **encendido**): aplica suavizado por envolvente convexa a la gráfica de líneas y al mapa de calor.
  - `Puntos de Smoothing...`: abre un diálogo con un spinbox (2–50, por defecto 2) para el número de puntos del suavizado.
  - `Escala de Color Compartida (mapa de calor)` (por defecto apagado): usa una sola escala de color para todas las hojas, en vez de autoescalar cada una por separado.
  - **Normalización** (elija una):
    | Modo | Descripción |
    |---|---|
    | Por Columna | divide entre el mínimo de esa columna en cada hoja (por defecto) |
    | Por Hoja | divide entre el mínimo de toda la hoja |
    | Por Columna en Todas las Hojas | divide entre el mínimo de esa columna combinando todas las hojas cargadas |
    | Global | divide entre el mínimo de todo el conjunto de datos cargado |
- **Gráfica**
  - `Límites de Ejes (Gráfica Superior)...`: fija manualmente los límites X/Y de la gráfica de líneas (botón "Auto" para revertir).
  - `Límites de Color (Heatmap)...`: fija manualmente el rango de color del mapa de calor, mostrando el rango actualmente en uso (botón "Auto" para revertir).
  - `Save Plot Image...`: guarda la gráfica de líneas (PNG/PDF/TIFF/SVG/EPS).
  - `Save Heatmap Image...`: guarda el mapa de calor.
  - `Save Smoothed Data (XLSX, Multiple Sheets)...`: exporta **todas** las columnas comunes de **todas** las hojas cargadas, procesadas igual que en pantalla (interpoladas, normalizadas según el modo activo, suavizadas si aplica), en un solo archivo `.xlsx` con un solo sheet, cada hoja de origen separada por 20 filas en blanco. Corre en segundo plano con ventana de progreso.
- **Datos**
  - `Editar Clasificaciones...`: agregar, renombrar o eliminar etiquetas de clasificación (para etiquetar cada hoja cargada); los cambios se propagan a todas las columnas.
  - `Save Classifications...` / `Load Classifications...`: guardar/cargar, por columna de datos, qué clasificación tiene asignada cada hoja (XLSX/CSV).

### 6.3 Distribución de la pestaña

- Panel izquierdo: lista "DATOS (CELDAS)" con las columnas comunes disponibles.
- Divisor horizontal arrastrable entre el panel izquierdo y el área de gráficas (ancho mínimo forzado).
- Divisor vertical arrastrable entre:
  - **Panel superior**: gráfica de líneas combinada — la columna elegida, graficada para cada hoja cargada, con offset horizontal y separadores punteados entre hojas; comboboxes de clasificación alineados sobre cada segmento; botón "▶" para avanzar a la siguiente columna.
  - **Panel inferior**: un mosaico de mapa de calor por hoja (columnas = columnas de datos, filas = muestras).

Ambos paneles se redibujan para llenar su espacio al cambiar el tamaño de la ventana o al soltar los divisores.

---

## 7. Pestaña: Dendograma

Se crea la primera vez que hay datos cargados (vía `Archivo → Abrir Datos`) y se usa `Visualizacion → Dendograma`.

- Panel lateral: lista "COLUMNAS DE DATOS" (igual que en Visualización de Datos), lista "SELECCIÓN", botones "Add to Selection"/"Remove from Selection", y botones "Save Dendrogram Image" / "Save Dendrogram CSV".
- Área de gráfica: panel superior con la señal cruda de la columna elegida; panel inferior con el dendrograma de las columnas en Selección (mínimo 2), calculado con `AgglomerativeClustering` (`distance_threshold=0, n_clusters=None`) sobre la selección transpuesta, y dibujado con `scipy.cluster.hierarchy.dendrogram`.
- "Save Dendrogram CSV" exporta las etiquetas de cluster junto con la matriz de enlace (linkage: pasos de fusión, hijos, distancias).

---

## 8. Pestaña: Time Series (Series de Tiempo)

Se crea al usar `Visualizacion → Series de tiempo` (requiere datos cargados vía `Archivo → Abrir Datos`).

- Lista de columnas de datos y lista de Selección, propias de esta pestaña.
- Casilla "Show Labels".
- Vista previa de una sola señal (arriba) y superposición de varias señales con offset vertical por señal (abajo), para comparar visualmente varias columnas a la vez.
- Botones: "Save Image", "Save CSV", "Close Tab".

---

## 9. Cargar una matriz de correlación precalculada

`Archivo → Cargar Matriz de Correlacion` permite cargar una matriz de correlación ya calculada externamente (CSV o XLSX). Debe ser una matriz cuadrada; si contiene valores fuera del rango [-1, 1] se muestra una advertencia.

Se renderiza directamente en el área principal de la pestaña "Visualización de Datos" como un dendrograma (arriba) más un mapa de calor de correlación (abajo), con sus propios botones: "Save Dendrogram", "Save Correlation Matrix" (CSV), "Save Correlation Matrix Image", "Save All".

---

## 10. Atajos de teclado

| Atajo | Acción |
|---|---|
| `Ctrl+O` | Abrir OME-TIFF |

No hay soporte de arrastrar-y-soltar (drag-and-drop); todas las cargas de archivo se hacen a través de los diálogos de archivo del sistema operativo.

---

## 11. Actualizaciones automáticas

`Help → Check for Updates`:

1. Consulta en segundo plano el último commit de la rama `main` del repositorio `sergiocruzunamia/neclabunam` en GitHub, y lo compara contra la versión local registrada.
2. Si hay una versión más reciente, ofrece descargar las versiones actuales de los 8 archivos `.py` principales (`interface3.py`, `peak_functions.py`, `visualization_helpers.py`, `corr_dendo_functions.py`, `variability_functions.py`, `image_loader.py`, `image_processing.py`, `multi_xls_helpers.py`) y sobrescribirlos localmente.
3. Ofrece reiniciar la aplicación automáticamente tras actualizar.

Requiere conexión a internet. Esta función **no** actualiza dependencias (bibliotecas de Python) ni el propio ejecutable empaquetado — solo el código fuente `.py`, por lo que solo aplica a instalaciones que corren desde el código fuente (Opción B/C de instalación), no a los ejecutables precompilados.

---

## 12. Glosario de métodos

### Detección de picos (Peak Finder)

| Método | Técnica | Parámetros (valor por defecto) |
|---|---|---|
| Elliptic Envelope | `sklearn.covariance.EllipticEnvelope` con detrend por `ElasticNet` | Contamination (0.01) |
| Peak Caller | Heurística de subida/bajada por porcentaje | Rise % (5), Fall % (5), Max Lookback (10 pts), Max Lookahead (10 pts) |
| Local Outlier Factor | `sklearn.neighbors.LocalOutlierFactor` | N Neighbors (20) |
| Peak Function 4 | Elliptic Envelope + SVR | Contamination (0.01) |
| Isolation Forest | `sklearn.ensemble.IsolationForest` | Contamination (0.05) |
| Linear Model | `SGDOneClassSVM` | Nu (0.131) |
| Peak Function 7 | Lasso + Local Outlier Factor | N Neighbors (20) |

El suavizado (independiente del método de picos) usa una envolvente convexa: construye una línea base por tramos a partir de los puntos de menor "convexidad local" genuina de la señal (parámetro "Points", 2–50) y la resta de la señal original. Se usa tanto en Visualización de Datos como en Datos Múltiples.

### Métodos de variabilidad

Ver tabla en la [sección 4.1](#41-métodos-de-variabilidad-disponibles).

### Métodos de correlación

`pearson` (lineal), `kendall` (rangos, tau), `spearman` (rangos, rho) — disponibles en Visualización de Datos, Datos Múltiples (implícitamente vía dendrograma) y en las correlaciones de clusters del Análisis de Variabilidad.

---

## 13. Solución de problemas

Ver la sección "Solución de problemas" del [README](../README.md#solución-de-problemas) para errores de instalación y arranque.

**Problemas específicos de uso:**

- **"Save Peaks CSV..." está deshabilitado**: ejecute primero un método de detección de picos sobre una columna y agregue al menos una columna a la lista de Selección.
- **El mapa de calor de correlación no aparece**: necesita al menos 2 columnas en la lista de Selección.
- **"Cargar Matriz de Correlacion" rechaza mi archivo**: la matriz debe ser cuadrada (mismo número de filas y columnas, con los mismos encabezados).
- **En Datos Múltiples no aparecen algunas columnas de mis hojas**: solo se muestran las columnas presentes en **todas** las hojas cargadas (comparadas por posición, ya que se leen sin encabezado). Si una hoja tiene menos columnas que las demás, las columnas "extra" de las otras hojas no se mostrarán.
- **Visualizacion → Dendograma / Series de tiempo aparecen deshabilitados**: solo se activan tras cargar datos con `Archivo → Abrir Datos (.npy / .csv)`; no se activan al usar "Abrir Multiples Archivos (.xls)" ni "Cargar Matriz de Correlacion".

Si el problema persiste, contacte a sergio.cruz@ciencias.unam.mx o revise/abra un issue en el repositorio de GitHub.
