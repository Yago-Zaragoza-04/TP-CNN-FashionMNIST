"""Etapa 1 — Carga, verificación y descripción del conjunto de datos Fashion-MNIST.

Uso:
    python src/datos.py [--origen RUTA]

La ruta de los datos se toma, en este orden, de --origen, de la variable de
entorno FASHION_MNIST_DIR o, por defecto, de la carpeta archive/ del repositorio.
"""

import argparse
import gzip
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

SEMILLA = 42

# Nombres de clase en español, en el orden de las etiquetas originales del dataset
CLASES = {
    0: "Remera/top",
    1: "Pantalón",
    2: "Buzo",
    3: "Vestido",
    4: "Abrigo",
    5: "Sandalia",
    6: "Camisa",
    7: "Zapatilla",
    8: "Bolso",
    9: "Bota",
}

# Nombres originales, tal como figuran en la Tabla 2 del paper (Xiao, Rasul y Vollgraf, 2017)
CLASES_ORIGINALES = {
    0: "T-Shirt/Top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandals",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boots",
}

LADO = 28  # las imágenes son de 28x28 píxeles
NUM_CLASES = len(CLASES)

# Nombres de archivo según cada fuente
CSV_TRAIN = "fashion-mnist_train.csv"
CSV_TEST = "fashion-mnist_test.csv"
IDX = {
    "x_train": "train-images-idx3-ubyte",
    "y_train": "train-labels-idx1-ubyte",
    "x_test": "t10k-images-idx3-ubyte",
    "y_test": "t10k-labels-idx1-ubyte",
}

# Rutas del proyecto (relativas a este archivo)
RAIZ_TP = Path(__file__).resolve().parent.parent
RAIZ_REPO = RAIZ_TP.parent
DIR_SALIDAS = RAIZ_TP / "outputs"
DIR_FIGURAS = DIR_SALIDAS / "figuras"
RUTA_RESULTADOS = DIR_SALIDAS / "resultados.json"


def fijar_semilla(semilla=SEMILLA):
    """Fija la semilla de random, numpy y tensorflow para que todo sea reproducible."""
    random.seed(semilla)
    np.random.seed(semilla)
    tf.random.set_seed(semilla)


# ---------------------------------------------------------------- lectura

def _buscar_idx(origen, nombre):
    """Devuelve la ruta del archivo idx, comprimido (.gz, GitHub) o no (Kaggle)."""
    for candidato in (origen / nombre, origen / f"{nombre}.gz"):
        if candidato.exists():
            return candidato
    return None


def detectar_formato(origen):
    """Indica qué formato usar: 'idx' (partición oficial) tiene prioridad sobre 'csv'."""
    origen = Path(origen)
    if all(_buscar_idx(origen, n) for n in IDX.values()):
        return "idx"
    if (origen / CSV_TRAIN).exists() and (origen / CSV_TEST).exists():
        return "csv"
    raise FileNotFoundError(f"No se encontraron los archivos de Fashion-MNIST en {origen}")


def _leer_csv(ruta):
    """Lee un CSV de Kaggle: columna 'label' + 784 columnas de píxeles."""
    df = pd.read_csv(ruta)
    assert df.shape[1] == 1 + LADO * LADO, f"{ruta.name}: cantidad de columnas inesperada"
    assert not df.isnull().values.any(), f"{ruta.name}: hay valores nulos"
    y = df["label"].to_numpy(dtype=np.int64)
    x = df.drop(columns="label").to_numpy(dtype=np.uint8)
    return x, y


def _leer_idx(ruta, magico_esperado):
    """Lee un archivo idx (formato original de LeCun) comprimido o no."""
    abrir = gzip.open if ruta.suffix == ".gz" else open
    with abrir(ruta, "rb") as f:
        datos = f.read()
    magico = int.from_bytes(datos[0:4], "big")
    assert magico == magico_esperado, f"{ruta.name}: número mágico {magico}, se esperaba {magico_esperado}"
    n = int.from_bytes(datos[4:8], "big")
    if magico == 2051:  # imágenes: n, filas, columnas
        filas = int.from_bytes(datos[8:12], "big")
        cols = int.from_bytes(datos[12:16], "big")
        return np.frombuffer(datos, dtype=np.uint8, offset=16).reshape(n, filas * cols)
    return np.frombuffer(datos, dtype=np.uint8, offset=8).astype(np.int64)  # etiquetas


def cargar_datos(origen, formato=None):
    """Carga Fashion-MNIST desde una carpeta con los CSV de Kaggle o los idx de GitHub.

    Devuelve un diccionario con:
      x_train, x_test: float32 en [0, 1], forma (n, 28, 28, 1)
      y_train, y_test: etiquetas enteras 0..9
      y_train_cat, y_test_cat: etiquetas one-hot (to_categorical, como en el video de la cátedra)
      formato: 'csv' o 'idx'
    """
    origen = Path(origen)
    formato = formato or detectar_formato(origen)

    if formato == "csv":
        x_train, y_train = _leer_csv(origen / CSV_TRAIN)
        x_test, y_test = _leer_csv(origen / CSV_TEST)
    elif formato == "idx":
        x_train = _leer_idx(_buscar_idx(origen, IDX["x_train"]), 2051)
        y_train = _leer_idx(_buscar_idx(origen, IDX["y_train"]), 2049)
        x_test = _leer_idx(_buscar_idx(origen, IDX["x_test"]), 2051)
        y_test = _leer_idx(_buscar_idx(origen, IDX["y_test"]), 2049)
    else:
        raise ValueError(f"Formato desconocido: {formato}")

    # Normalización a [0, 1] y reshape a un solo canal (escala de grises)
    x_train = (x_train.astype("float32") / 255.0).reshape(-1, LADO, LADO, 1)
    x_test = (x_test.astype("float32") / 255.0).reshape(-1, LADO, LADO, 1)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "y_train_cat": to_categorical(y_train, NUM_CLASES),
        "y_test_cat": to_categorical(y_test, NUM_CLASES),
        "formato": formato,
    }


# ---------------------------------------------------------------- verificación

def verificar_datos(d):
    """Controles de integridad del dataset. Si alguno falla, se corta la ejecución."""
    assert d["x_train"].shape == (60000, LADO, LADO, 1), d["x_train"].shape
    assert d["x_test"].shape == (10000, LADO, LADO, 1), d["x_test"].shape
    assert len(d["y_train"]) == 60000 and len(d["y_test"]) == 10000
    assert d["x_train"].dtype == np.float32 and d["x_test"].dtype == np.float32
    for x in (d["x_train"], d["x_test"]):
        assert not np.isnan(x).any(), "hay píxeles nulos (NaN)"
        assert x.min() >= 0.0 and x.max() <= 1.0, "hay píxeles fuera de [0, 1]"
    assert np.bincount(d["y_train"], minlength=NUM_CLASES).tolist() == [6000] * NUM_CLASES
    assert np.bincount(d["y_test"], minlength=NUM_CLASES).tolist() == [1000] * NUM_CLASES
    assert d["y_train_cat"].shape == (60000, NUM_CLASES)
    assert d["y_test_cat"].shape == (10000, NUM_CLASES)
    assert (d["y_train_cat"].argmax(axis=1) == d["y_train"]).all()
    assert (d["y_test_cat"].argmax(axis=1) == d["y_test"]).all()


def diagnosticar_csv_vs_idx(d_csv, d_idx):
    """Compara imagen por imagen los CSV de Kaggle con los idx (partición oficial de Zalando)."""

    def claves(x):
        x8 = np.round(x * 255).astype(np.uint8).reshape(len(x), -1)
        return [fila.tobytes() for fila in x8]

    csv_tr, csv_te = claves(d_csv["x_train"]), claves(d_csv["x_test"])
    idx_tr, idx_te = set(claves(d_idx["x_train"])), set(claves(d_idx["x_test"]))
    s_tr, s_te = set(csv_tr), set(csv_te)
    return {
        "identicos": all(np.array_equal(d_csv[k], d_idx[k]) for k in ("x_train", "y_train", "x_test", "y_test")),
        "csv_train_imagenes_repetidas": len(csv_tr) - len(s_tr),
        "csv_test_imagenes_repetidas": len(csv_te) - len(s_te),
        "csv_train_en_train_oficial": len(s_tr & idx_tr),
        "csv_train_en_test_oficial": len(s_tr & idx_te),
        "csv_train_ausentes_del_oficial": len(s_tr - idx_tr - idx_te),
        "csv_test_en_test_oficial": len(s_te & idx_te),
        "csv_test_en_train_oficial": len(s_te & idx_tr),
        "csv_test_ausentes_del_oficial": len(s_te - idx_tr - idx_te),
        "imagenes_compartidas_csv_train_y_csv_test": len(s_tr & s_te),
    }


def md5_archivos(origen, nombres):
    """MD5 de los archivos usados, para poder cotejarlos con los del repositorio oficial."""
    import hashlib

    return {n: hashlib.md5((origen / n).read_bytes()).hexdigest() for n in nombres}


# ---------------------------------------------------------------- figuras

# Convención visual común a todas las figuras con imágenes del dataset (01, 04, 08, 09)
EPIGRAFE_INVERTIDA = (
    "Visualización invertida (prenda oscura sobre fondo claro); el modelo recibe la imagen "
    "original, con la prenda clara sobre fondo negro."
)


def mostrar_imagen(eje, imagen, titulo=None):
    """Dibuja una imagen del dataset con la convención del TP: escala de grises invertida, sin ejes.

    Solo cambia la visualización (cmap 'gray_r'); los datos que recibe el modelo no se tocan.
    """
    eje.imshow(np.asarray(imagen).reshape(LADO, LADO), cmap="gray_r", vmin=0, vmax=1)
    eje.set_xticks([])
    eje.set_yticks([])
    for borde in eje.spines.values():
        borde.set_visible(False)
    if titulo:
        eje.set_title(titulo, fontsize=9)


def agregar_epigrafe(fig, texto=EPIGRAFE_INVERTIDA, tam=8.5):
    """Agrega el epígrafe de la visualización invertida al pie de la figura."""
    fig.text(0.5, 0.005, texto, ha="center", va="bottom", fontsize=tam, color="#444444", wrap=True)


def figura_muestras(x, y, ruta, por_clase=6):
    """Grilla de 10 clases x 6 ejemplos tomados al azar (semilla fija) del conjunto de entrenamiento."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEMILLA)
    fig, ejes = plt.subplots(NUM_CLASES, por_clase, figsize=(por_clase * 1.25 + 1.6, NUM_CLASES * 1.25 + 0.4))
    for c in range(NUM_CLASES):
        indices = rng.choice(np.flatnonzero(y == c), size=por_clase, replace=False)
        for j, i in enumerate(indices):
            mostrar_imagen(ejes[c, j], x[i])
        ejes[c, 0].set_ylabel(f"{c} · {CLASES[c]}", rotation=0, ha="right", va="center", fontsize=15)
    fig.suptitle("Fashion-MNIST: seis ejemplos por clase\n(conjunto de entrenamiento)", fontsize=16)
    fig.tight_layout()  # la aclaración de la visualización invertida va en el epígrafe del informe
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_distribucion(y_train, y_test, ruta):
    """Barras agrupadas con la cantidad de imágenes por clase en entrenamiento y prueba."""
    import matplotlib.pyplot as plt

    n_train = np.bincount(y_train, minlength=NUM_CLASES)
    n_test = np.bincount(y_test, minlength=NUM_CLASES)
    pos = np.arange(NUM_CLASES)
    ancho = 0.38

    fig, eje = plt.subplots(figsize=(10, 5))
    b1 = eje.bar(pos - ancho / 2 - 0.01, n_train, ancho, label="Entrenamiento", color="#2a78d6")
    b2 = eje.bar(pos + ancho / 2 + 0.01, n_test, ancho, label="Prueba", color="#eb6834")
    eje.bar_label(b1, labels=[f"{v:,}".replace(",", ".") for v in n_train], fontsize=8, color="#333333", padding=2)
    eje.bar_label(b2, labels=[f"{v:,}".replace(",", ".") for v in n_test], fontsize=8, color="#333333", padding=2)
    eje.set_xticks(pos, [CLASES[c] for c in range(NUM_CLASES)], rotation=30, ha="right")
    eje.set_xlabel("Clase")
    eje.set_ylabel("Cantidad de imágenes")
    eje.set_title("Distribución de imágenes por clase: entrenamiento vs. prueba")
    eje.yaxis.set_major_formatter(lambda v, _: f"{int(v):,}".replace(",", "."))
    eje.set_ylim(0, n_train.max() * 1.25)  # margen superior para la leyenda
    eje.grid(axis="y", color="#e0e0e0", linewidth=0.8)
    eje.set_axisbelow(True)
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    eje.legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- registros

def guardar_entorno(ruta):
    """Guarda las versiones exactas del entorno de ejecución."""
    import matplotlib
    import sklearn

    lineas = [
        f"python: {platform.python_version()}",
        f"tensorflow: {tf.__version__}",
        f"keras: {tf.keras.__version__}",
        f"numpy: {np.__version__}",
        f"pandas: {pd.__version__}",
        f"scikit-learn: {sklearn.__version__}",
        f"matplotlib: {matplotlib.__version__}",
        f"sistema operativo: {platform.system()} {platform.release()} ({platform.machine()})",
    ]
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return lineas


def actualizar_resultados(seccion, contenido, ruta=RUTA_RESULTADOS):
    """Escribe una sección de resultados.json sin pisar las de otras etapas."""
    resultados = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    resultados[seccion] = contenido
    ruta.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")


def resumen_dataset(d, origen, formato, motivo_formato, diagnostico):
    """Arma la sección 'dataset' de resultados.json a partir de los datos cargados."""
    n_train = np.bincount(d["y_train"], minlength=NUM_CLASES)
    n_test = np.bincount(d["y_test"], minlength=NUM_CLASES)
    archivos = [CSV_TRAIN, CSV_TEST] if formato == "csv" else [_buscar_idx(origen, n).name for n in IDX.values()]
    return {
        "nombre": "Fashion-MNIST",
        "autor": "Zalando Research",
        "origen_usado": f"Kaggle (carpeta archive/, archivos {formato})",
        "archivos_usados": archivos,
        "md5_archivos_usados": md5_archivos(origen, archivos),
        "carpeta_origen": os.path.relpath(origen, RAIZ_REPO).replace("\\", "/"),
        "motivo_formato": motivo_formato,
        "diagnostico_csv_vs_idx": diagnostico,
        "url_kaggle": "https://www.kaggle.com/datasets/zalando-research/fashionmnist",
        "url_repositorio_oficial": "https://github.com/zalandoresearch/fashion-mnist",
        "licencia": "MIT",
        "cita": (
            "Xiao, H., Rasul, K. y Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset "
            "for Benchmarking Machine Learning Algorithms. arXiv:1708.07747."
        ),
        "n_entrenamiento": int(len(d["y_train"])),
        "n_prueba": int(len(d["y_test"])),
        "n_total": int(len(d["y_train"]) + len(d["y_test"])),
        "forma_imagen": list(d["x_train"].shape[1:]),
        "canales": int(d["x_train"].shape[-1]),
        "tipo_pixeles": str(d["x_train"].dtype),
        "rango_pixeles": [float(min(d["x_train"].min(), d["x_test"].min())),
                          float(max(d["x_train"].max(), d["x_test"].max()))],
        "num_clases": NUM_CLASES,
        "clases": {str(c): nombre for c, nombre in CLASES.items()},
        "clases_originales": {str(c): nombre for c, nombre in CLASES_ORIGINALES.items()},
        "fuente_clases_originales": "Xiao, Rasul y Vollgraf (2017), arXiv:1708.07747, Tabla 2",
        "distribucion_entrenamiento": {CLASES[c]: int(n_train[c]) for c in range(NUM_CLASES)},
        "distribucion_prueba": {CLASES[c]: int(n_test[c]) for c in range(NUM_CLASES)},
        "preprocesamiento": [
            "normalización de píxeles a [0, 1] (división por 255), tipo float32",
            "reshape a (n, 28, 28, 1): un canal, escala de grises",
            "etiquetas one-hot con to_categorical (10 clases)",
        ],
        "verificaciones": "OK",
        "semilla": SEMILLA,
    }


# ---------------------------------------------------------------- principal

def main():
    parser = argparse.ArgumentParser(description="Etapa 1: carga y descripción de Fashion-MNIST")
    parser.add_argument("--origen", default=os.environ.get("FASHION_MNIST_DIR", str(RAIZ_REPO / "archive")))
    parser.add_argument("--formato", choices=["csv", "idx"], default=None,
                        help="forzar formato; por defecto idx (partición oficial de Zalando)")
    args = parser.parse_args()
    origen = Path(args.origen).resolve()

    fijar_semilla()
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    for linea in guardar_entorno(DIR_SALIDAS / "entorno.txt"):
        print(linea)

    formato = args.formato or detectar_formato(origen)
    if args.formato:
        motivo = "elegido por parámetro"
    elif formato == "idx":
        motivo = "formato por defecto: idx, idénticos a los del repositorio oficial (partición oficial)"
    else:
        motivo = "no hay archivos idx en el origen; se usan los CSV"
    diagnostico = None

    # Si en la carpeta están ambos formatos, se cotejan imagen por imagen y queda registrado
    hay_csv = (origen / CSV_TRAIN).exists() and (origen / CSV_TEST).exists()
    hay_idx = all(_buscar_idx(origen, n) for n in IDX.values())
    if hay_csv and hay_idx:
        d_csv, d_idx = cargar_datos(origen, "csv"), cargar_datos(origen, "idx")
        verificar_datos(d_csv)
        verificar_datos(d_idx)
        diagnostico = diagnosticar_csv_vs_idx(d_csv, d_idx)
        print("Diagnóstico CSV vs idx:", json.dumps(diagnostico, ensure_ascii=False))
        if formato == "idx" and not diagnostico["identicos"]:
            motivo += "; los CSV no reproducen la partición oficial (ver diagnostico_csv_vs_idx)"
        d = d_csv if formato == "csv" else d_idx
    else:
        d = cargar_datos(origen, formato)
        verificar_datos(d)

    print(f"\nOrigen: {origen} (formato {formato}: {motivo})")
    print("Verificaciones: OK")

    figura_muestras(d["x_train"], d["y_train"], DIR_FIGURAS / "01_muestras.png")
    figura_distribucion(d["y_train"], d["y_test"], DIR_FIGURAS / "02_distribucion.png")
    actualizar_resultados("dataset", resumen_dataset(d, origen, formato, motivo, diagnostico))
    print(f"Figuras y {RUTA_RESULTADOS.name} actualizados en {DIR_SALIDAS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
