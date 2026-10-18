"""Predicción sobre imágenes sueltas con el modelo entrenado.

Uso:
    python src/predecir.py ruta_imagen.png [--modelo RUTA]

La imagen se pasa a escala de grises y se redimensiona a 28x28 con antialiasing. Si el fondo es
claro (media de los bordes > 127), se invierte para que la prenda quede clara sobre fondo negro,
como en Fashion-MNIST. Después se normaliza a [0, 1].
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import datos

UMBRAL_FONDO_CLARO = 127
MODELO_POR_DEFECTO = datos.DIR_SALIDAS / "modelo_fashion_cnn.keras"


def preparar_imagen(ruta):
    """Devuelve la imagen lista para el modelo, forma (1, 28, 28, 1), y datos del preprocesamiento."""
    img = Image.open(ruta)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        # Las zonas transparentes se apoyan sobre fondo blanco antes de pasar a grises
        fondo = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(fondo, img.convert("RGBA"))
    img = img.convert("L")
    tamano_original = img.size
    if img.size != (datos.LADO, datos.LADO):
        img = img.resize((datos.LADO, datos.LADO), Image.Resampling.LANCZOS, reducing_gap=3.0)
    pixeles = np.asarray(img, dtype=np.float32)
    bordes = np.concatenate([pixeles[0], pixeles[-1], pixeles[1:-1, 0], pixeles[1:-1, -1]])
    media_bordes = float(bordes.mean())
    invertida = media_bordes > UMBRAL_FONDO_CLARO
    if invertida:
        pixeles = 255.0 - pixeles
    x = (pixeles / 255.0).reshape(1, datos.LADO, datos.LADO, 1).astype("float32")
    return x, {"tamano_original": tamano_original, "media_bordes": media_bordes, "invertida": bool(invertida)}


def top_k(probabilidades, k=3):
    """Las k clases más probables como lista de (índice, nombre, probabilidad)."""
    orden = np.argsort(probabilidades)[::-1][:k]
    return [(int(c), datos.CLASES[int(c)], float(probabilidades[c])) for c in orden]


def coma(valor, decimales=2):
    """Número con coma decimal."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def main():
    parser = argparse.ArgumentParser(description="Clasifica una imagen de prenda con la CNN entrenada")
    parser.add_argument("imagen")
    parser.add_argument("--modelo", default=os.environ.get("FASHION_MODELO", str(MODELO_POR_DEFECTO)))
    args = parser.parse_args()

    from tensorflow import keras

    modelo = keras.models.load_model(args.modelo)
    x, info = preparar_imagen(Path(args.imagen))
    probabilidades = modelo.predict(x, verbose=0)[0]
    ranking = top_k(probabilidades)

    print(f"Imagen: {Path(args.imagen).name} ({info['tamano_original'][0]}×{info['tamano_original'][1]}, "
          f"media de bordes {coma(info['media_bordes'], 1)}, "
          f"{'invertida' if info['invertida'] else 'sin invertir'})")
    print(f"Clase predicha: {ranking[0][1]} (p = {coma(ranking[0][2])})")
    for i, (_, nombre, p) in enumerate(ranking, start=1):
        print(f"  {i}. {nombre}: {coma(p, 4)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
