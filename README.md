Implementación y Comparación de Modelos de Traducción Automática (NMT)

Este repositorio contiene la implementación y evaluación comparativa de cuatro arquitecturas de Traducción Automática Neuronal (NMT) para el par de idiomas Español-Inglés. El objetivo es analizar la evolución de las arquitecturas NMT, desde un RNN simple hasta el modelo Transformer, bajo un pipeline de datos homogéneo.

Este proyecto fue desarrollado por estudiantes de la Universidad Andina del Cusco.

🚀 Modelos Implementados

Se implementaron cuatro arquitecturas clave, cada una en su propio script:

simpleRNN.py

Arquitectura: Seq2Seq básico con SimpleRNN.

Framework: TensorFlow (Keras).

Descripción: Un modelo de línea base sin mecanismo de atención.

LSTM_traductor.py

Arquitectura: Seq2Seq con BiLSTM y Atención Bahdanau.

Framework: PyTorch.

Descripción: Implementa la atención aditiva clásica para mejorar la captura de contexto.

traductor2.py

Arquitectura: Seq2Seq con BiGRU y Atención Bahdanau Vectorizada.

Framework: TensorFlow (Keras).

Descripción: Una implementación optimizada de la atención de Bahdanau para un entrenamiento más rápido en TensorFlow.

traductor_transformer.py

Arquitectura: Modelo Transformer completo (Encoder-Decoder).

Framework: TensorFlow (Keras).

Descripción: Basado en el paper "Attention Is All You Need", utiliza únicamente auto-atención y atención cruzada.

📚 Dataset y Preprocesamiento

Todos los modelos fueron entrenados y evaluados utilizando el mismo corpus:

Fuente: Corpus Tatoeba (ES-EN) vía OPUS.

Muestreo: ~50,000 pares de oraciones.

Datos Finales: ~46,104 pares (después de limpieza, normalización y filtrado).

Tokenización: SentencePiece (BPE). Se entrena un tokenizador sobre los datos de entrenamiento para manejar palabras raras o desconocidas (OOV) de forma efectiva.

📊 Resultados Comparativos

La siguiente tabla resume el rendimiento y el costo computacional de cada modelo bajo las condiciones experimentales del informe.

Modelo

Archivo

BLEU Score

Parámetros

Tiempo (6 épocas)

Arq. Clave

RNN Simple

simpleRNN.py

17.40

~12.6 M

~15.0 min

SimpleRNN

LSTM

LSTM_traductor.py

25.66

~51.7 M

~28.5 min

BiLSTM + Atención

GRU

traductor2.py

40.78

~22.0 M

~116.5 min

BiGRU + Atención (Vect.)

Transformer

traductor_transformer.py

53.22

~19.7 M

~501.6 min

Multi-Head Attention

Análisis de Hallazgos

Calidad (BLEU): El Transformer (53.22) es el claro ganador, seguido por el GRU (40.78). Ambos modelos con atención superan significativamente a las arquitecturas más antiguas.

Eficiencia (Parámetros): El modelo LSTM (51.7 M) es el más pesado, mientras que el Transformer (19.7 M) y el GRU (22.0 M) demuestran un mejor balance entre complejidad y rendimiento.

Costo (Tiempo): El Transformer (~8.4h) es, por mucho, el más lento de entrenar debido a su complejidad, mientras que los modelos recurrentes son significativamente más rápidos.

⚙️ Uso y Ejecución

Cada script está diseñado para ser ejecutado de forma independiente.

Requisitos

Necesitarás las siguientes bibliotecas de Python:

# Para todos los modelos
pip install tensorflow sentencepiece sacrebleu

# Adicionalmente para el modelo LSTM
pip install torch


Ejecutar un Modelo

Descarga los datos del corpus Tatoeba (p.ej., Tatoeba.en-es.es y Tatoeba.en-es.en).

Coloca los archivos del corpus en el mismo directorio que los scripts.

Ejecuta el script de Python del modelo que deseas probar:


# Ejemplo para ejecutar el modelo Transformer
python traductor_transformer.py


El script se encargará de todo el pipeline:

Limpiar y preprocesar los datos.

Entrenar el tokenizador SentencePiece.

Construir y entrenar el modelo.

Evaluar el BLEU score final.

Iniciar una interfaz interactiva en la consola para probar traducciones.


