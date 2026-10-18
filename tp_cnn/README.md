# TP CNN — Clasificación de prendas de Fashion-MNIST

Trabajo práctico de Conceptos de Inteligencia Artificial (Universidad de Palermo): construir, entrenar y
validar una red neuronal convolucional (CNN) con `tensorflow.keras` sobre Fashion-MNIST
(Zalando Research; Xiao, Rasul y Vollgraf, 2017, arXiv:1708.07747; licencia MIT).

Los resultados de cada etapa quedan en `outputs/resultados.json`; el informe se genera a partir de ese archivo.

## Instalación

Requiere Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

Los scripts leen los archivos idx de Fashion-MNIST, comprimidos (`.gz`) o no, desde la carpeta
`archive/` ubicada junto a `tp_cnn/`. Para usar otra carpeta, pasá `--origen RUTA` o definí la variable de
entorno `FASHION_MNIST_DIR`. Los archivos oficiales están en
https://github.com/zalandoresearch/fashion-mnist (carpeta `data/fashion/`). El dataset no se incluye en este
paquete.

## Ejecución (desde la carpeta `tp_cnn`)

| Comando | Qué hace |
|---|---|
| `python src/datos.py` | a. Carga y verifica el dataset, compara CSV vs idx, guarda `entorno.txt` y las figuras 01 y 02 |
| `python src/modelo.py` | b. Construye la CNN (sin entrenar), guarda el resumen, la figura 03 y la prueba forward |
| `python src/entrenar.py --prueba` | Mide el tiempo de una época sin guardar nada |
| `python src/entrenar.py` | c. Entrenamiento completo con early stopping; guarda el modelo, el historial y las figuras 04 a 06 |
| `python src/entrenar.py --solo-figuras` | Regenera las figuras 04 a 06 sin reentrenar |
| `python src/mlp_referencia.py` | Entrena la red sin convolución usada como referencia |
| `python src/validar.py` | d. Evalúa el modelo guardado sobre el test oficial; figuras 07 a 09 |
| `python src/predecir.py imagen.png` | Clasifica una imagen suelta y muestra el top-3 |
| `python src/generar_informe.py --modulo N --fecha DD/MM/AAAA` | Genera el informe .docx a partir de `outputs/` |

El orden importa: `validar.py` necesita el modelo entrenado y los resultados de `mlp_referencia.py`.
La semilla es 42 en todas las etapas; sin determinismo forzado de operaciones, en CPU puede haber
variaciones mínimas entre corridas.

## Uso del modelo entrenado

`outputs/modelo_fashion_cnn.keras` es el modelo final, con los pesos de la mejor época:

```bash
python src/predecir.py mi_prenda.png
```

La imagen se pasa a escala de grises, se lleva a 28 × 28 con antialiasing y, si el fondo es claro, se
invierte para que la prenda quede clara sobre fondo negro, como en el dataset. El modelo espera prendas
centradas y recortadas, como las de Fashion-MNIST.

## Estructura

```
src/        datos.py, modelo.py, entrenar.py, mlp_referencia.py, validar.py, predecir.py, generar_informe.py
outputs/    modelo_fashion_cnn.keras
recursos/   logo_up.png (encabezado del informe)
```
