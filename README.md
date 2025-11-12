# Herramienta para la caracterización del comportamiento del cangrejo azul

**Autora:** Olivia Castiñeiras Queijo  
**Directores:** Álvaro Rodríguez Tajes, Elena Ortega Jiménez

Este repositorio contiene el código fuente de la aplicación de escritorio desarrollada para el Trabajo de Fin de Grado "Herramienta para la caracterización del comportamiento del cangrejo azul" (Universidade da Coruña, 2025).

La aplicación es una herramienta de análisis de alto nivel que permite a los investigadores cargar datos de trayectoria, aplicar calibración de perspectiva, configurar un pipeline de post-procesado (suavizado, filtrado) y generar métricas etológicas (velocidad, sinuosidad, tigmotaxis, etc.), análisis comparativos y reportes en PDF.

## Contenido del Repositorio

* `/app.py`: El script principal que lanza la interfaz gráfica de usuario (GUI) construida con CustomTkinter.
* `/analisis.py`: El motor de análisis científico (backend lógico) que contiene todas las funciones de procesamiento (Pandas) y calibración (OpenCV).
* `/datos_csv/`: Carpeta que contiene los 17 archivos `.csv` con los datos de trayectoria brutos (en píxeles) utilizados en el experimento del Capítulo 8.
* `/Resultados_Resumen_Metricas.csv`: La tabla resumen con las métricas finales calculadas del experimento "Día vs. Noche".

## Estructura de Carpetas e Instalación

Para que la aplicación (`app.py`) funcione correctamente, es **fundamental** replicar la siguiente estructura de carpetas en la raíz del proyecto, ya que el código depende de estas rutas relativas:
TFG-Cangrejo-Analisis/
│
├── app.py                      (Script principal de la interfaz)
├── analisis.py                 (Motor de análisis científico)
├── docker-hub-docker-compose.yaml  (Orquestador del backend)
│
├── config/                     (¡Carpeta obligatoria!)
│   └── run_conf.yaml           (Fichero de configuración del tracker)
│
├── datos_csv/
│   ├── (Aquí van los 17 .csv del experimento Día/Noche)
│   └── Resultados_Resumen_Metricas.csv
│
├── results/                    (¡Carpeta obligatoria!)
│   ├── stats/                  (Debe existir, es donde Docker escribe los CSV)
│   └── videos/                 (Debe existir, es donde Docker escribe los vídeos)
│
├── videos/                     (¡Carpeta obligatoria!)
│   └── (Debe existir, es donde Docker lee los vídeos a procesar)
│
└── README.md                   

**Nota:** Las carpetas `config/`, `videos/`, `results/stats/` y `results/videos/` **deben existir** (aunque estén vacías) *antes* de ejecutar `app.py` por primera vez para evitar errores `FileNotFoundError`.

## Requisitos y Componentes Externos

Esta aplicación (`app.py`, `analisis.py`) es una plataforma de análisis que gestiona la calibración, el post-procesado y la visualización de datos de trayectoria.
Para reproducir el experimento completo, se necesitan dos tipos de componentes externos:

### 1. Datos de Vídeo (Datos Crudos)

Los 17 vídeos `.mp4` originales utilizados en el experimento están disponibles permanentemente en Zenodo:
* **DOI:** [https://doi.org/10.5281/zenodo.17589599]

### 2. Backend de Tracking (Capa de Acceso a Datos)

La aplicación está diseñada para ser **agnóstica al motor de tracking**. Funciona orquestando un servicio de *backend* externo a través de una "interfaz" simple:
1.  Modifica un archivo `config/run_conf.yaml` para establecer parámetros (ej. renderizado).
2.  Ejecuta un comando de sistema (ej. `docker compose up`) para iniciar el *backend*.
3.  Espera un archivo `.csv` con los datos de trayectoria en la carpeta `results/stats/`.

Cualquier motor de tracking (ej. `idtracker.ai`, `DeepLabCut`) que se empaquete en un contenedor Docker y sea modificado para respetar este "contrato" de archivos podría ser utilizado por la aplicación.

El *backend* específico **utilizado y validado en este TFG** es el desarrollado por Matías Martín González (basado en YOLOv8 + ByteTrack).

Por lo tanto, para que el botón "Procesar Vídeo" de esta aplicación funcione tal y como está en este repositorio, se necesita tener **Docker** instalado y el contenedor de dicho proyecto accesible.

Los ficheros `docker-hub-docker-compose.yaml` y `config/run_conf.yaml` (incluidos en este repositorio) son los archivos de configuración originales de dicho *backend*, y son necesarios para que `app.py` pueda comunicarse con el contenedor de Docker.
* **Repositorio del Backend (TFM de M. Martin Gonzalez):** [https://github.com/mmartiasg/crab-track-app]

## Dependencias de Python

El motor de análisis (`analisis.py`) y la interfaz (`app.py`) requieren las siguientes librerías de Python:

* pandas
* numpy
* opencv-python-headless
* customtkinter
* matplotlib
* reportlab
* PyYAML (para leer/escribir el `run_conf.yaml`)