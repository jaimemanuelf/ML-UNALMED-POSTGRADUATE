# Proyecto final de Aprendizaje de Máquinas (UNAL Medellín)

Proyecto de Machine Learning para **pitch shifting (octava arriba)** aplicado a procesamiento de audio y orientado a integración posterior en un plugin VST3.

## Tabla de contenido
- [Descripción](#descripción)
- [Objetivo](#objetivo)
- [Alcance actual](#alcance-actual)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Dataset esperado](#dataset-esperado)
- [Salida del entrenamiento](#salida-del-entrenamiento)
- [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)
- [Hola]
## Descripción
Este repositorio contiene un notebook de experimentación donde se construye y entrena un modelo para transformar señales de audio originales (`x`) en señales objetivo desplazadas una octava (`y`), usando una arquitectura basada en STFT + Conv1D y una búsqueda de hiperparámetros con enfoque tipo REINFORCE.

## Objetivo
Entrenar un modelo de regresión de audio que:
- reciba buffers de entrada,
- aprenda la transformación de pitch shifting (octave up),
- y pueda ser usado posteriormente en un flujo de inferencia de baja latencia.

## Alcance actual
- Preprocesamiento y validación de pares de audio (`x`, `y`).
- Construcción de dataset por ventanas/buffers.
- Entrenamiento en TensorFlow/Keras.
- Selección de hiperparámetros mediante agente RL.
- Exportación del mejor modelo entrenado.

## Estructura del repositorio
```text
.
├── ML_Project_v8__3__(4).ipynb   # Notebook principal del proyecto
└── README.md                     # Documentación del repositorio
```

## Requisitos
- Python 3.10+ (recomendado)
- Entorno Jupyter o Google Colab
- Dependencias principales:
  - `tensorflow`
  - `numpy`
  - `librosa`
  - `soundfile`
  - `scikit-learn`
  - `matplotlib`

## Instalación
Ejemplo rápido en entorno local:

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install tensorflow numpy librosa soundfile scikit-learn matplotlib notebook
```

Luego abre el notebook:

```bash
jupyter notebook
```

## Uso
1. Abre `ML_Project_v8__3__(4).ipynb`.
2. Ajusta rutas y constantes globales (sample rate, tamaño de buffer, paths del dataset).
3. Verifica la estructura del dataset (`x/` y `y/`).
4. Ejecuta las celdas en orden para:
   - emparejado de archivos,
   - validación técnica de señales,
   - construcción de ventanas,
   - entrenamiento y guardado del mejor modelo.

> El notebook está planteado para ejecución en Colab con Google Drive, pero puede adaptarse a entorno local cambiando las rutas.

## Dataset esperado
Se espera una carpeta con la siguiente estructura:

```text
DataSetParaPitchShift/
├── x/   # audios originales (.wav)
└── y/   # audios objetivo (.wav), con convención de nombre asociada a x
```

Cada archivo en `x` debe tener su correspondiente versión desplazada en `y` para formar pares de entrenamiento válidos.

## Salida del entrenamiento
Durante el entrenamiento se guarda el mejor modelo encontrado en la ruta definida en el notebook (por ejemplo, `best_rl_model.h5` dentro del dataset en Drive/local).

## Limitaciones y trabajo futuro
- Migrar el pipeline a scripts modulares (`src/`) para facilitar mantenibilidad y pruebas.
- Incluir archivo `requirements.txt` y/o `environment.yml`.
- Agregar métricas objetivas de calidad de audio y evaluación perceptual.
- Empaquetar inferencia para integración directa con plugin VST3.
