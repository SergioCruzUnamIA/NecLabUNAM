# NecLab

Cell_detection_complete - Programa que extrae los centros de las celulas y extrae las señales (Necesita el archivo test.tif, bajarlo del siguiente link https://drive.google.com/file/d/1EP7TQMWQglbhgoRdJm2bY-10s2cYMDa3/view?usp=drive_link)

Signal_processing - Programa para procesar las señales y encontrar picos

Interface3 - Programa que crea una simple interfaz para el uso de los programas anteriores

## Instalación

### Con Anaconda

1. Descargar el instalador de miniconda desde esta página web: https://www.anaconda.com/download/success.

2. Instalar miniconda con el instalador

3. Descargar el repositorio (Dar click en el botón verde que dice Code, posteriormente dar click en download zip)

4. Extraer el zip que contiene el repositorio

5. Abrir una terminal en la ruta donde se extrajo el repositorio

6. En la terminal escribir

    > $ conda env create -f environment.yml

    Esto descargará todas las bibliotecas necesarias para el funcionamiento del programa

7. Una vez se halla completado el paso anterior, escribir en la terminal

    > $ conda activate .neclabconda

    Esto activará el ambiente virtual con las bibliotecas ya descargadas

8. Una vez se halla activado el ambiente virtual, se verá en la terminal (.neclabconda) en donde se escriben los comandos, esto indica que el ambiente virtual ha sido creado exitosamente.

9. Ahora navegamos a la carpeta src, escribir en la terminal
    > $ cd src

10. Para correr el programa escribir
    > $ python3 interface3.py

**NOTA:** Puede ser que después de instalar y activar el ambiente virtual (paso 7) el programa marque que no se ha instalado una biblioteca. En este caso se recomienda cerrar la terminal y abrir una nueva en la carpeta del programa y repetir los pasos 7 en adelante.



