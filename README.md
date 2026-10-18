# TP: Redes Neuronales Convolucionales (CNN) sobre Fashion-MNIST

**Universidad de Palermo · Facultad de Ingeniería · Inteligencia Artificial**
Alumno: Zaragoza Yago (legajo 0129980) · Docente: Ignacio Casais

Consigna: construir un modelo e implementarlo utilizando una CNN. (a) Buscar y seleccionar un conjunto de
datos; (b) programar la red neuronal; (c) realizar el proceso de aprendizaje; (d) validar el modelo con
ejemplos, cotejando el resultado.

## Entregables

| Archivo | Contenido |
|---|---|
| [Informe (.docx)](tp_cnn/entrega/TP_Redes_Neuronales_Convolucionales_-_Zaragoza_Yago.docx) | Informe del trabajo práctico |
| [Informe (.pdf)](tp_cnn/entrega/TP_Redes_Neuronales_Convolucionales_-_Zaragoza_Yago.pdf) | El mismo informe en PDF |
| [Notebook](tp_cnn/entrega/TP_CNN_FashionMNIST.ipynb) | Flujo completo ejecutado, con sus salidas (datos → modelo → entrenamiento → validación → predicción). Compatible con Google Colab |
| [Código](tp_cnn/src/) | Scripts de cada etapa |

## Estructura

```
tp_cnn/
  src/        datos.py · modelo.py · entrenar.py · mlp_referencia.py · validar.py · predecir.py · generar_informe.py
  outputs/    resultados.json (todas las métricas), historial.json, figuras/, modelo entrenado (.keras)
  notebook/   notebook ejecutado
  entrega/    informe (.docx y .pdf), notebook y código comprimido
  recursos/   logo del encabezado del informe
```

## Cómo ejecutar

Requiere Python 3.12. Las instrucciones detalladas están en [tp_cnn/README.md](tp_cnn/README.md).

```bash
cd tp_cnn
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python src/predecir.py outputs/pruebas/test_881_original.png   # usa el modelo ya entrenado
```

El dataset no se incluye en el repositorio. Para reentrenar, descargá los archivos idx de Fashion-MNIST desde el
[repositorio oficial](https://github.com/zalandoresearch/fashion-mnist) (carpeta `data/fashion/`) y ubicalos en
una carpeta `archive/` en la raíz de este repositorio, o pasá su ruta con `--origen`. El notebook los descarga
solo si no los encuentra.

## Fuentes

- Xiao, H., Rasul, K. y Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine
  Learning Algorithms*. arXiv:1708.07747. Licencia MIT.
- Repositorio oficial de Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- Material de cátedra: PPT Redes Neuronales Convolucionales, apunte Aprendizaje en las Redes Neuronales y video M9.
