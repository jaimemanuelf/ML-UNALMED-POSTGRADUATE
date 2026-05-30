## Random Forest para Pitch Shifting 

- [Descripción](#descripción)
- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Estructura del dataset](#estructura)
- [Configuración](#configuración)
- [Flujo del notebook](#flujo)
- [Métricas de evaluación](#metricas)

# Descripción
El modelo recibe buffers de 512 muestras de audio de guitarra eléctrica y predice el buffer equivalente con la frecuencia fundamental desplazada una octava hacia arriba . Este modelo aprende la transformación directamente desde los datos.

# Arquitectura
'''text
Input: (N, 512)  ── buffer de waveform de guitarra
         │
   MultiOutputRegressor
         │
   RandomForestRegressor  ×512  (un RF por cada sample de salida)
   ├── Árbol 1  ──┐
   ├── Árbol 2  ──┤  promedio de predicciones
   ├── ...      ──┤
   └── Árbol 100──┘
         │
Output: (N, 512) ── waveform con pitch shift +1 octava
'''

MultiOutputRegressor entrena un RandomForestRegressor independiente por cada una de las 512 posiciones de salida. Cada árbol considera sqrt(512) ≈ 22 features por split.
Hiperparámetros por defecto:
ParámetroValorDescripciónn_estimators100Número de árboles por RFmax_depth20Profundidad máxima por árbolmin_samples_split5Mínimo de muestras para dividir un nodomin_samples_leaf2Mínimo de muestras en una hojamax_features"sqrt"Features evaluados por splitn_jobs-1Usa todos los núcleos disponibles

#Requisitos

-LibreríaVersiónscikit-learn
-1.6.1librosa0.11.0
-numpy2.4.6
-scipy1.16.3
-joblib1.5.3

El notebook está diseñado para ejecutarse en Kaggle 

#Estructura del dataset
El dataset debe estar organizado así:
DataSetParaPitchShift/
├── x/
│   ├── AcordesYArpegios.wav
│   ├── Prueba - MAIN.wav
│   └── ...                     ← audio original de guitarra
└── y/
    ├── AcordesYArpegiosShifted.wav
    ├── Prueba - MAINShifted.wav
    └── ...                     ← mismo audio con +1 octava
    
El emparejado x → y es automático: busca {nombre}Shifted.wav en la carpeta y. Si no encuentra coincidencia exacta, intenta un fallback por prefijo. Los archivos sin par son omitidos con advertencia.


# Configuración
Los parámetros globales se definen en la celda de configuración:
pythonSAMPLE_RATE    = 96000   # Hz
BUFFER_SIZE    = 512     # muestras por buffer (~5.33 ms)
HOP_SIZE       = 256     # overlap 50% durante entrenamiento
HOP_SIZE_INFER = 512     # sin overlap durante inferencia

MAX_BUFFERS_PER_FILE = 2000  # límite de buffers por archivo (None = todos)

# Flujo del notebook
1. Instalación de dependencias
        ↓
2. Imports
        ↓
3. Configuración global + validación de rutas
        ↓
4. Emparejado automático x → y  +  split train/val (90/10 a nivel de archivo)
        ↓
5. Funciones utilitarias
   ├── normalize_audio_pair()   — normalización compartida x/y
   └── create_buffers()         — segmentación con hop_size
        ↓
6. Carga del dataset → arrays numpy (X_train, Y_train, X_val, Y_val)
        ↓
7. Definición del modelo  MultiOutputRegressor(RandomForestRegressor)
        ↓
8. Entrenamiento  →  model_rf.fit(X_train, Y_train)
        ↓
9. Evaluación  →  MSE · SNR · Correlación de Pearson
        ↓
10. Visualización de importancia de features
        ↓
11. Inferencia sobre audio completo con overlap-add
        ↓
12. Tabla comparativa Random Forest vs WaveNet

# Métricas de evaluación
Tres métricas calculadas sobre el conjunto de validación, idénticas a las del notebook WaveNet para comparación directa:
- MSE (Mean Squared Error) — error cuadrático medio muestra a muestra. Menor es mejor.
- SNR (Signal-to-Noise Ratio) — relación señal/ruido en dB calculada por frame, promediada sobre frames no silenciosos. Mayor es mejor.
- Correlación de Pearson — similitud lineal entre predicción y ground truth, promedio sobre 500 frames. Rango [-1, 1], mayor es mejor.
