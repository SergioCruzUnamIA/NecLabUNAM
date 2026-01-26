# NecLab

Software para el análisis automatizado de imágenes de microscopía, diseñado para la detección y segmentación de células en imágenes de fluorescencia. Desarrollado en el laboratorio LansBiodyt de la Facultad de Ciencias, UNAM.

---

## Descripción

NecLab permite a investigadores realizar análisis cuantitativos de imágenes de microscopía de manera eficiente. El software incluye herramientas para:

- Cargar y visualizar imágenes en formato OME-TIFF
- Ajustar brillo y contraste de las imágenes
- Aplicar diferentes métodos de análisis de variabilidad temporal
- Detectar y segmentar células mediante clustering espacial
- Extraer y analizar series temporales de células individuales
- Calcular correlaciones entre comportamientos celulares
- Exportar resultados en formatos CSV y generar reportes

---

## Requisitos del sistema

- Sistema operativo: Windows, macOS o Linux
- Python 3.9 o superior
- Miniconda o Anaconda

---

## Instalación

### Paso 1: Instalar Miniconda

Si no tiene Miniconda instalado, descárguelo desde:
https://docs.conda.io/en/latest/miniconda.html

Siga las instrucciones de instalación para su sistema operativo.

### Paso 2: Descargar el repositorio

Opción A - Clonar con Git:
```
git clone https://github.com/SergioCruzUnamIA/NecLab.git
```

Opción B - Descargar ZIP:
1. Ir a la página del repositorio en GitHub
2. Dar clic en el botón verde "Code"
3. Seleccionar "Download ZIP"
4. Extraer el archivo en la ubicación deseada

### Paso 3: Crear el ambiente virtual

Abra una terminal y navegue a la carpeta del proyecto:
```
cd ruta/a/NecLab
```

Cree el ambiente virtual con las dependencias:
```
conda env create -f environment.yml
```

Este proceso descargará todas las bibliotecas necesarias. Puede tomar varios minutos dependiendo de su conexión a internet.

### Paso 4: Activar el ambiente virtual

```
conda activate neclab_env
```

Una vez activado, verá `(neclab_env)` al inicio de la línea de comandos.

---

## Ejecución

Con el ambiente virtual activado, ejecute:

```
python interface3_clean.py
```

La primera ejecución puede tardar entre 1 y 2 minutos mientras se cargan las bibliotecas.

---

## Guía de uso

### Abrir una imagen

1. Ir a Archivo > Abrir OME-TIFF
2. Seleccionar el archivo de imagen a analizar
3. La imagen se mostrará en el panel principal

### Ajustar la visualización

En el panel derecho encontrará los controles:

- **Capa (Frame)**: Permite navegar entre los diferentes frames de la imagen
- **Brillo**: Ajusta el brillo de la visualización
- **Contraste**: Ajusta el contraste de la visualización
- **Threshold**: Aplica un umbral para binarizar la imagen (marque "Aplicar" para activar)

### Análisis de variabilidad

1. Ir a Imagen > Análisis de Variabilidad
2. Seleccionar uno de los 7 métodos disponibles:
   - Rango
   - Varianza Poblacional
   - Varianza Muestral
   - Desviación Estándar Poblacional
   - Desviación Estándar Muestral
   - Coeficiente de Variación
   - Rango Intercuartílico (IQR)

### Detectar células

Una vez en la ventana de análisis de variabilidad:

1. Ajustar el valor de Threshold según la imagen
2. Dar clic en "Aplicar Binarización"
3. Dar clic en "Encontrar Clusters"
4. Ajustar Min Size y Max Size para filtrar por tamaño
5. Dar clic en "Procesar Clusters (Básico)" o "(Avanzado)"

### Seleccionar células

- Haga clic sobre los clusters en el gráfico inferior derecho para seleccionarlos
- Los clusters seleccionados se mostrarán en rojo
- La lista de clusters seleccionados aparece en el panel lateral

### Analizar correlaciones

1. Seleccione los clusters de interés haciendo clic sobre ellos
2. Dar clic en "Usar Seleccionados"
3. Se abrirá la ventana de análisis de correlaciones con:
   - Series temporales de los clusters
   - Matriz de correlación
   - Opciones para exportar datos

### Visualización 3D

Dar clic en "Vista 3D" para ver la superficie de variabilidad en tres dimensiones. Puede rotar la vista con el mouse.

### Exportar resultados

En la ventana de correlaciones:
- "Exportar Series Temporales": Guarda las series en formato CSV
- "Exportar Coordenadas": Guarda las posiciones de los clusters
- "Generar Reporte": Crea un resumen del análisis

---

## Estructura del proyecto

```
NecLab/
├── interface3_clean.py      # Interfaz gráfica principal
├── variability_functions.py # Métodos de variabilidad y clustering
├── peak_functions.py        # Detección de picos
├── corr_dendo_functions.py  # Correlaciones y dendrogramas
├── visualization_helpers.py # Funciones auxiliares de visualización
├── image_loader.py          # Carga de imágenes OME-TIFF
├── image_processing.py      # Procesamiento de imágenes
├── environment.yml          # Dependencias del proyecto
└── README.md                # Este archivo
```

---

## Solución de problemas

**El programa no inicia:**
- Verifique que el ambiente virtual esté activado (debe ver `(neclab_env)` en la terminal)
- Cierre la terminal, abra una nueva y active el ambiente nuevamente

**Error de biblioteca no encontrada:**
- Ejecute: `pip install nombre_de_la_biblioteca --break-system-packages`
- O reinstale el ambiente: `conda env remove -n neclab_env` y repita la instalación

**La imagen no se visualiza correctamente:**
- Verifique que el archivo sea formato OME-TIFF válido
- Intente ajustar el brillo y contraste en el panel de controles

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

Este proyecto está bajo la licencia incluida en el archivo LICENSE.
