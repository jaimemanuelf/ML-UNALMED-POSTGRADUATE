#WaveNet para Pitch Shifting

## Tabla de contenidos

- [Descripción](#descripción)
- [Arquitectura](#arquitectura)
- [Agente REINFORCE](#agente-reinforce)
- [Requisitos](#requisitos)
- [Estructura del dataset](#estructura-del-dataset)
- [Configuración](#configuración)
- [Flujo del notebook](#flujo-del-notebook)
- [Pipeline de datos](#pipeline-de-datos)
- [Inferencia](#inferencia)
- [Outputs generados](#outputs-generados)

## Descripción
El modelo recibe buffers de 512 muestras de audio de guitarra eléctrica (a 96 kHz, ~5.33 ms) y predice el buffer equivalente con la frecuencia fundamental desplazada una octava hacia arriba (+12 semitonos). Aprende la transformación completamente en el **dominio del tiempo**, sin ninguna representación frecuencial intermedia.

El entrenamiento no usa un conjunto de hiperparámetros fijo: un **agente REINFORCE** explora el espacio de configuraciones disponible durante `MAX_TRIALS` trials, guarda únicamente el modelo con menor `val_loss`, y actualiza su política para favorecer las combinaciones que mejor funcionaron.

---

## Arquitectura

```
Input: (batch, 512, 1)  ── buffer de waveform normalizado
         │
    Conv1D(n_filters, 1, causal)   ← proyección inicial
         │
    ┌────┴──────────────────────────────────┐
    │  Bloque dilatado × n_layers           │
    │                                       │
    │  dilation = 2^i  (i = 0 … n_layers-1)│
    │                                       │
    │  h_tanh = Conv1D(causal, tanh)  ─┐   │
    │  h_sig  = Conv1D(causal, sigmoid)─┤   │
    │           Multiply()  (gated)    │   │
    │                │                 │   │
    │  residual ─────┤  skip ──────────┘   │
    └────────────────┴───────────────────── ┘
         │
    sum(all skip outputs)
         │
    ReLU → Conv1D(n_filters, 1) → ReLU → Conv1D(1, 1)
         │
Output: (batch, 512, 1) ── waveform con pitch shift +1 octava
```

## Agente REINFORCE

El agente automatiza la búsqueda de hiperparámetros sin grid search exhaustivo.

**Espacio de búsqueda:**

| Hiperparámetro | Opciones |
|---|---|
| `lr` (learning rate) | `1e-4`, `3e-4` |
| `batch_size` | `64`, `128` |
| `n_layers` | `6`, `8` |
| `n_filters` | `32`, `64` |

**Funcionamiento por trial:**
1. El agente muestrea una combinación de hiperparámetros según su política softmax.
2. Se construye y entrena un WaveNet con esa configuración durante `EPOCHS_PER_TRIAL` épocas.
3. La **recompensa** se calcula como `reward = 1.0 - val_loss`.
4. El agente actualiza sus logits favoreciendo combinaciones con `reward > baseline` (media histórica).
5. Si el trial supera el mejor anterior, el modelo se guarda en disco.

```
reward = 1.0 - val_loss
advantage = reward - baseline
logits[k][idx] += lr_agent × advantage
```

**Parámetros del agente:**

| Parámetro | Valor |
|---|---|
| `MAX_TRIALS` | 5 |
| `EPOCHS_PER_TRIAL` | 30 |
| `lr_agent` | 0.15 |

---

## Requisitos

```bash
pip install "tensorflow>=2.16" librosa soundfile numpy scipy
```

Versiones mínimas verificadas:

| Librería | Versión |
|---|---|
| TensorFlow | ≥ 2.16 |
| librosa | 0.11.0 |
| numpy | compatible con TF 2.16 |
| scipy | cualquier reciente |

El notebook está diseñado para ejecutarse en **Kaggle con GPU Tesla T4** (acelerador recomendado para TensorFlow). La GPU reduce significativamente el tiempo de entrenamiento por trial.

---

## Estructura del dataset

```
DataSetParaPitchShift/
├── x/
│   ├── AcordesYArpegios.wav
│   ├── Prueba - MAIN.wav
│   └── ...                     ← audio original de guitarra (96 kHz, mono)
└── y/
    ├── AcordesYArpegiosShifted.wav
    ├── Prueba - MAINShifted.wav
    └── ...                     ← mismo audio con pitch shift +1 octava
```

El emparejado `x → y` es automático: el notebook busca `{nombre}Shifted.wav` en la carpeta `y`. Si no hay coincidencia exacta, intenta un fallback por prefijo. Archivos sin par son omitidos con advertencia.

En Kaggle, añadir el dataset desde: **Add data → buscar `datasetparapitchshift` (by jhorlandaniel)**.

El split train/val se hace al **90/10 a nivel de archivo**, nunca a nivel de buffer, para evitar data leakage.

---

## Configuración

Todos los parámetros globales se definen una sola vez en la celda de configuración:

```python
# Audio
SAMPLE_RATE    = 96000   # Hz
BUFFER_SIZE    = 512     # muestras por buffer (~5.33 ms) — coincide con tamaño ASIO
HOP_SIZE       = 256     # overlap 50% durante entrenamiento
HOP_SIZE_INFER = 512     # sin overlap durante inferencia

# Entrenamiento RL
MAX_TRIALS       = 5
EPOCHS_PER_TRIAL = 30

# Espacio de hiperparámetros
HPARAM_SPACE = {
    "lr"        : [1e-4, 3e-4],
    "batch_size": [64, 128],
    "n_layers"  : [6, 8],
    "n_filters" : [32, 64],
}
```

---

## Flujo del notebook

```
1. Instalación de dependencias (TensorFlow ≥ 2.16, librosa, soundfile)
        ↓
2. Imports
        ↓
3. Configuración global (rutas, audio, RL, espacio de hiperparámetros)
        ↓
4. Verificación de rutas Kaggle (X_PATH / Y_PATH)
        ↓
5. Emparejado automático x → y  +  split train/val (90/10 por archivo)
        ↓
6. Funciones utilitarias
   ├── normalize_audio_pair()   — normalización compartida por máximo absoluto
   └── create_buffers()         — segmentación con hop_size configurable
        ↓
7. Generador de waveforms  +  pipeline tf.data
   (sin materializar arrays grandes en RAM)
        ↓
8. Arquitectura WaveNet  — build_wavenet(hparams)
        ↓
9. Agente REINFORCE  — ReinforceAgent
        ↓
10. Ciclo de entrenamiento RL
    ├── Muestrear hiperparámetros
    ├── Entrenar WaveNet (EPOCHS_PER_TRIAL épocas)
    ├── Calcular reward = 1 - val_loss
    ├── Actualizar política del agente
    └── Guardar modelo si es el mejor trial
        ↓
11. Inferencia con overlap-add sobre audio completo
    └── Exportar audio .wav transformado
```

---

## Pipeline de datos

El pipeline usa **generadores Python + `tf.data`** para mantener el uso de RAM acotado a un archivo a la vez, sin materializar el dataset completo en memoria.

```python
waveform_generator(pairs_list)
    ↓ produce (x_buf, y_buf) de shape (512, 1) bajo demanda
tf.data.Dataset.from_generator(...)
    ↓ shuffle → batch → prefetch
make_tf_dataset(pairs_list, batch_size, shuffle_buffer=2000)
```

Esto es especialmente importante porque a 96 kHz con hop de 256 muestras, un solo archivo de 30 segundos genera ~11 250 buffers. Materializar todos en RAM antes de entrenar sería innecesario con `tf.data`.

---

## Inferencia

La función `predict_audio` transforma una señal completa usando el modelo guardado:

1. Divide el audio en buffers de 512 muestras con `HOP_SIZE_INFER = 512` (sin solapamiento).
2. Predice en lotes de 256 con `model.predict`.
3. Reconstruye la señal completa con **overlap-add** (acumula predicciones y divide por conteos).
4. Exporta el resultado como `.wav` a 96 kHz.

```python
modelo = tf.keras.models.load_model(
    BEST_MODEL_PATH,
    custom_objects={"wavenet_loss": wavenet_loss},
    compile=False,
)
y_out = predict_audio(modelo, x_signal)
sf.write("wavenet_output.wav", y_out, SAMPLE_RATE)
```

---

## Outputs generados

Todos los archivos se guardan en `/kaggle/working/`:

| Archivo | Descripción |
|---|---|
| `best_wavenet_model.h5` | Modelo Keras del trial con menor `val_loss` |
| `wavenet_output.wav` | Audio de prueba transformado con pitch shift +1 octava |

---

