# Language-based Audio Retrieval in DCASE 2023 Challenge

This repository provides the baseline system for **Language-based Audio Retrieval** (Task 6B) in DCASE 2023 Challenge.

**2023/03/20 Update:**
:fast_forward:
Training checkpoints for the baseline system and its audio encoder are available on Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7752975.svg)](https://doi.org/10.5281/zenodo.7752975).

![Language-based Audio Retrieval](figs/dcase2023_task_6b.png)

# Baseline Retrieval System

![Baseline Retrieval System](figs/baseline_system.png)

```
- Audio Encoder                   # fine-tuned PANNs, i.e., CNN14
- Text Encoder                    # pretrained Sentence-BERT, i.e., all-mpnet-base-v2
- Contrastive Learning Objective  # InfoNCE loss
```

# Quick Start

This codebase is developed with Python 3.9 and [PyTorch 1.13.0](https://pytorch.org/).

## 1. Check out source code and install required python packages:

```
git clone https://github.com/xieh97/dcase2023-audio-retrieval.git
pip install -r requirements.txt
```

## 2. Download the [Clotho](https://zenodo.org/record/4783391) dataset:

```
Clotho
├─ clotho_captions_development.csv
├─ clotho_captions_validation.csv
├─ clotho_captions_evaluation.csv
├─ development
│   └─...(3839 wavs)
├─ validation
│   └─...(1045 wavs)
└─ evaluation
    └─...(1045 wavs)
```

## 3. Pre-process audio and caption data:

```
preprocessing
├─ audio_logmel.py              # extract log-mel energies from audio clips
├─ clotho_dataset.py            # process audio captions, generate fids and cids
├─ sbert_embeddings.py          # generate sentence embeddings using Sentence-BERT (all-mpnet-base-v2)
└─ cnn14_transfer.py            # transfer pretrained CNN14 (Cnn14_mAP=0.431.pth)

```
*  ```audio_logmel.py```  

    - Procesa archivos de audio contenidos en diferentes particiones (development, validation, evaluation) de un directorio y extrae las características del log mel-spectrogram para cada archivo .wav.
    - Almacena los resultados de este procesamiento en archivos HDF5, uno por cada partición.
    - Lee el archivo "audio_info.pkl" para obtener un mapeo entre identificadores y nombres de los archivos de audio, lo que es necesario para asignar correctamente cada archivo procesado en los HDF5.

* ```clotho_dataset.py``` :

    - Procesa datos de audio y texto de un dataset, asignando identificadores únicos a cada archivo de audio y asociándolos con su duración y nombre.
    - Almacena la información procesada en ficheros: genera "audio_info.pkl" con los mapeos de audios y sus duraciones, guarda archivos CSV procesados para los textos (por ejemplo, "development_text.csv", "validation_text.csv", "evaluation_text.csv") y un archivo "vocab_info.pkl" con estadísticas y el vocabulario obtenido.
    - Accede a otros archivos para cumplir su función: lee archivos de audio (.wav) desde los directorios "development", "validation" y "evaluation" para extraer duraciones, y lee archivos CSV de captions (como "development_captions.csv", "validation_captions.csv", "evaluation_captions.csv") para procesar la información textual y enlazarla con los audios.

* ```sbert_embeddings.py``` :

    - El código genera embeddings de texto utilizando un modelo preentrenado SentenceTransformer ('all-mpnet-base-v2') y asigna a cada entrada de texto un vector de 768 dimensiones.
    - Almacena los embeddings resultantes en un archivo llamado "sbert_embeds.pkl".
    - Accede a archivos CSV de textos (por ejemplo, "development_text.csv", "validation_text.csv" y "evaluation_text.csv") para extraer los datos crudos de texto y procesarlos, lo que le permite enlazar cada embedding con su identificador único (tid).


* ```cnn14_transfer.py``` : 

    - El código transfiere parámetros de un modelo CNN14 preentrenado a una nueva instancia de la clase CNN14Encoder con salida de dimensión 300.
    - Almacena el modelo resultante en un archivo llamado "CNN14_300.pth".
    - Accede al archivo "Cnn14_mAP=0.431.pth" para cargar los parámetros preentrenados, y utiliza un mapeo de claves para adaptarlos al nuevo modelo.


## 4. Train the baseline system:

```
models
├─ core.py                      # dual-encoder framework
├─ audio_encoders.py            # audio encoders
└─ text_encoders.py             # text encoders

utils
├─ criterion_utils.py           # loss functions
├─ data_utils.py                # Pytorch dataset classes
└─ model_utils.py               # model.train(), model.eval(), etc.

conf.yaml                       # experimental settings
main.py                         # main()
```

- ***models***
    * ```core.py```
    Funcionalidad general:

        - Define una red neuronal llamada DualEncoderModel que combina dos codificadores: uno para audio y otro para texto.
        Su propósito es generar representaciones (embeddings) de entrada de audio y texto para compararlas o combinarlas.

        - Importa módulos externos, específicamente audio_encoders y text_encoders, para construir los codificadores.

    * ```audio_encoders.py```
        - Implementa un codificador de audio basado en una red neuronal convolucional profunda (CNN).

        - Convierte una entrada de audio en una representación numérica (embeddings)


    * ```text_encoders.py```
        - Define un codificador de texto basado en BERT, llamado SentBERTBaseEncoder.
        
        - Convierte secuencias de texto en representaciones numéricas (embeddings).

- ***utils***

    * ```criterion_utils.py```
        - Implementa una función de pérdida basada en Log Softmax utilizada en aprendizaje profundo para comparar incrustaciones de audio y texto. Evalúa la similitud entre pares de datos y penaliza diferencias mediante una variante de la pérdida InfoNCE.
        
        - Recibe como entrada incrustaciones de audio y texto generadas previamente y los procesa dentro de la misma ejecución.

    * ```data_utils.py```
        - El código implementa un conjunto de clases y funciones para manejar un dataset que combina información de audio y texto. Se enfoca en la representación, carga y procesamiento de estos datos con el fin de facilitar su uso en modelos de aprendizaje automático.
        - Se cargan embeddings de texto desde archivos en formato pickle (`sbert_embeds.pkl`)  
        - Los datos de audio se leen desde archivos HDF5 (`h5py`) para extraer los datos de audio.  
        - Se emplea un diccionario (`Vocabulary`) para almacenar los embeddings de texto y sus identificadores numéricos.
        - Se leen archivos CSV para cargar datos textuales.  
        - Estos archivos se usan para construir un dataset compatible con PyTorch. 

    * ```model_utils.py```
        - El código define funciones para inicializar, entrenar, evaluar y restaurar un modelo de aprendizaje profundo basado en la arquitectura de doble codificador. Este modelo trabaja con representaciones de audio y texto.
        - Durante la inicialización, puede cargar pesos preexistentes para los codificadores de audio y texto.
        - Guarda y restaura estados del modelo desde archivos de punto de control (`checkpoint`).
        - Carga pesos preentrenados de un archivo especificado en `params["audio_enc"]["weight"]`.
        - Restaura un modelo guardado desde un archivo de `checkpoint` ubicado en `ckp_dir`.




- ```conf.yaml```
    - El código configura un entorno de entrenamiento para un modelo de aprendizaje profundo basado en la arquitectura **DualEncoderModel**. Utiliza **Ray Tune** para la optimización de hiperparámetros y define la configuración de datos, modelo, criterio de pérdida y optimización.
    - El lugar donde se va a almacenar los resultados del entrenamiento habrá que establecerlo modificando **YOUR_OUTPUT_DIR** y **YOUR_OUTPUT_PATH**
    - Carga datos de entrenamiento, validación y evaluación desde **YOUR_DATA_PATH/Clotho**, incluyendo:
        - Archivos de audio en formato **.hdf5**.
        - Archivos de texto en formato **.csv**.
        - Embeddings de texto en formato **.pkl**.
    - Carga pesos preentrenados del modelo de audio desde **YOUR_DATA_PATH/pretrained_models/CNN14_300.pth**.
    - Usa **Adam** como optimizador y **ReduceLROnPlateau** como programador de tasa de aprendizaje.
    - Emplea la función de pérdida **LogSoftmaxLoss** con similitud de producto punto o coseno.
    - Define un criterio de detención temprana basado en la métrica **stop_metric**.

- ```main.py```
    - El código implementa un sistema de entrenamiento y evaluación de modelos de aprendizaje profundo utilizando `PyTorch` y `Ray Tune`. Permite realizar experimentos de optimización de hiperparámetros mediante pruebas (trials), gestionando automáticamente la ejecución y el almacenamiento de los mejores modelos.
    - Guarda modelos y estados de optimización en archivos de checkpoint durante el entrenamiento y registra métricas de entrenamiento y validación para su posterior análisis. Además, almacena la mejor configuración encontrada en `Ray Tune`.
    - Carga configuraciones desde un archivo YAML (`conf.yaml`), lo que permite definir parámetros del modelo, datos y optimización sin modificar el código.
    - Accede a datos de entrenamiento y validación a través de funciones en `data_utils`, cargándolos en `DataLoader` para su procesamiento.
    - Recupera checkpoints previos para continuar entrenamientos interrumpidos o evaluar modelos guardados.


## 5. Calculate retrieval metrics:

```
postprocessing
├─ xmodal_scores.py             # calculate audio-text scores
└─ xmodal_retrieval.py          # calculate mAP, R@1, R@5, R@10, etc.
```

* ```xmodal_scores.py``` :


* ```xmodal_retrieval.py``` :

# Examples

## 1. Code example for using the pretrained audio encoder:

```
example
├─ audio_encoder.py             # code example for audio encoder
├─ example.wav                  # audio segment example
└─ audio_encoder.pth            # audio encoder checkpoint (https://doi.org/10.5281/zenodo.7752975)
```