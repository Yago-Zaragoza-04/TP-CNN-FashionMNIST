"""Etapa 2 — Construcción de la red neuronal convolucional (sin entrenar).

Uso:
    python src/modelo.py [--origen RUTA]

Sigue la lógica de la CNN sobre MNIST del video de la cátedra (Conv2D 3x3, MaxPooling 2x2,
Dropout 25 % / 50 %, Flatten, Dense, salida de 10) con un bloque convolucional más.
"""

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import datos

FORMA_ENTRADA = (datos.LADO, datos.LADO, 1)


def construir_modelo():
    """Construye y compila la CNN.

    Fija la semilla antes, para que los pesos iniciales sean reproducibles.
    """
    datos.fijar_semilla()
    modelo = keras.Sequential(
        [
            # En Keras 3 la forma de entrada se declara con keras.Input; reemplaza al
            # argumento input_shape=(28, 28, 1) que el video pasa a la primera Conv2D.
            keras.Input(shape=FORMA_ENTRADA, name="entrada"),
            # Bloque convolucional 1: 32 filtros de 3x3
            layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1_1"),
            layers.Conv2D(32, (3, 3), activation="relu", name="conv1_2"),
            layers.MaxPooling2D((2, 2), name="pool1"),
            layers.Dropout(0.25, name="drop1"),
            # Bloque convolucional 2: 64 filtros de 3x3
            layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2_1"),
            layers.Conv2D(64, (3, 3), activation="relu", name="conv2_2"),
            layers.MaxPooling2D((2, 2), name="pool2"),
            layers.Dropout(0.25, name="drop2"),
            # Clasificador: vector 1D, capa totalmente conectada y salida de 10 clases
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="densa"),
            layers.Dropout(0.5, name="drop3"),
            layers.Dense(datos.NUM_CLASES, activation="softmax", name="salida"),
        ],
        name="cnn_fashion_mnist",
    )
    modelo.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return modelo


# ---------------------------------------------------------------- descripción

TIPOS = {
    "Conv2D": "convolución",
    "MaxPooling2D": "pooling",
    "Dropout": "dropout",
    "Flatten": "flatten",
    "Dense": "densa",
}


def _forma(layer):
    """Forma de salida de una capa sin la dimensión del lote, como lista de enteros."""
    return [int(v) for v in layer.output.shape[1:]]


def _forma_texto(forma):
    return "×".join(str(v) for v in forma)


def describir_capa(layer):
    """Descripción corta en español del rol de cada capa, armada desde su configuración."""
    cfg = layer.get_config()
    clase = type(layer).__name__
    if clase == "Conv2D":
        kh, kw = cfg["kernel_size"]
        borde = "padding 'same', conserva el tamaño" if cfg["padding"] == "same" else "sin padding, reduce el borde"
        return f"Convolución: {cfg['filters']} filtros (kernel) de {kh}×{kw}, activación ReLU, {borde}"
    if clase == "MaxPooling2D":
        ph, pw = cfg["pool_size"]
        return f"Agrupación máxima {ph}×{pw}: conserva el valor máximo de cada grupo y reduce a la mitad alto y ancho"
    if clase == "Dropout":
        return f"Dropout: descarta al azar el {cfg['rate']:.0%} de las activaciones durante el entrenamiento"
    if clase == "Flatten":
        return "Flatten: convierte los mapas de características en un vector 1D"
    if clase == "Dense":
        if cfg["activation"] == "softmax":
            return f"Capa de salida totalmente conectada: {cfg['units']} neuronas con softmax (probabilidad por clase)"
        return f"Capa totalmente conectada: {cfg['units']} neuronas, activación ReLU"
    return clase


def detalle_capa(layer):
    """Texto breve con el tipo y los hiperparámetros, para las cajas del diagrama."""
    cfg = layer.get_config()
    clase = type(layer).__name__
    if clase == "Conv2D":
        kh, kw = cfg["kernel_size"]
        return f"Conv2D {kh}×{kw}, {cfg['filters']}" + (", same" if cfg["padding"] == "same" else "")
    if clase == "MaxPooling2D":
        return "MaxPooling {}×{}".format(*cfg["pool_size"])
    if clase == "Dropout":
        return f"Dropout {cfg['rate']:.0%}"
    if clase == "Flatten":
        return "Flatten"
    if clase == "Dense":
        return f"Dense {cfg['units']}, {cfg['activation']}"
    return clase


def config_capa(layer):
    """Hiperparámetros relevantes de la capa, para citarlos en el informe sin tipearlos."""
    cfg = layer.get_config()
    clase = type(layer).__name__
    if clase == "Conv2D":
        return {"filtros": cfg["filters"], "kernel": list(cfg["kernel_size"]), "padding": cfg["padding"],
                "activacion": cfg["activation"]}
    if clase == "MaxPooling2D":
        return {"pool": list(cfg["pool_size"])}
    if clase == "Dropout":
        return {"tasa": cfg["rate"]}
    if clase == "Dense":
        return {"unidades": cfg["units"], "activacion": cfg["activation"]}
    return {}


def resumen_texto(modelo):
    """Devuelve el texto de model.summary()."""
    buffer = io.StringIO()
    modelo.summary(print_fn=lambda linea, **_: buffer.write(linea + "\n"))
    return buffer.getvalue()


def verificar_parametros(modelo):
    """Recalcula con la fórmula los parámetros de conv1_1 y de la capa densa y los compara con Keras."""
    conv = modelo.get_layer("conv1_1")
    kh, kw = conv.get_config()["kernel_size"]
    canales = int(conv.input.shape[-1])
    filtros = conv.get_config()["filters"]
    calc_conv = (kh * kw * canales + 1) * filtros

    densa = modelo.get_layer("densa")
    entradas = int(densa.input.shape[-1])
    neuronas = densa.get_config()["units"]
    calc_densa = (entradas + 1) * neuronas

    return {
        "conv1_1": {
            "formula": "(kh·kw·canales_entrada + 1)·filtros",
            "sustitucion": f"({kh}·{kw}·{canales} + 1)·{filtros}",
            "calculado": calc_conv,
            "keras": int(conv.count_params()),
            "coincide": calc_conv == conv.count_params(),
        },
        "densa": {
            "formula": "(entradas + 1)·neuronas",
            "sustitucion": f"({entradas} + 1)·{neuronas}",
            "calculado": calc_densa,
            "keras": int(densa.count_params()),
            "coincide": calc_densa == densa.count_params(),
        },
    }


def prueba_forward(modelo, x, n=8):
    """Pasa un lote de n imágenes por el modelo sin entrenar y controla forma y suma de probabilidades."""
    salida = modelo.predict(x[:n], verbose=0)
    sumas = salida.sum(axis=1)
    ok = salida.shape == (n, datos.NUM_CLASES) and np.allclose(sumas, 1.0, atol=1e-5)
    assert ok, f"salida {salida.shape}, sumas {sumas}"
    return {
        "lote": f"primeras {n} imágenes del conjunto de prueba",
        "forma_entrada": [int(v) for v in x[:n].shape],
        "forma_salida": [int(v) for v in salida.shape],
        "suma_por_fila": [round(float(s), 6) for s in sumas],
        "desvio_maximo_respecto_de_1": float(np.abs(sumas - 1.0).max()),
        "ok": bool(ok),
        "nota": "modelo sin entrenar: las probabilidades todavía no tienen significado",
    }


def describir_arquitectura(modelo):
    """Arma la sección 'arquitectura' de resultados.json."""
    capas = [{
        "nombre": "entrada",
        "tipo": "Input",
        "forma_salida": list(FORMA_ENTRADA),
        "parametros": 0,
        "descripcion": "Imagen de 28×28 píxeles con un canal (escala de grises), normalizada a [0, 1]",
    }]
    for layer in modelo.layers:
        capas.append({
            "nombre": layer.name,
            "tipo": type(layer).__name__,
            "forma_salida": _forma(layer),
            "parametros": int(layer.count_params()),
            "descripcion": describir_capa(layer),
            "config": config_capa(layer),
        })
    entrenables = int(sum(np.prod(w.shape) for w in modelo.trainable_weights))
    no_entrenables = int(sum(np.prod(w.shape) for w in modelo.non_trainable_weights))
    mayor = max(capas, key=lambda c: c["parametros"])
    por_tipo = {}
    for c in capas[1:]:
        tipo = TIPOS[c["tipo"]]
        por_tipo[tipo] = por_tipo.get(tipo, 0) + c["parametros"]
    return {
        "nombre_modelo": modelo.name,
        "capas": capas,
        "total_parametros": entrenables + no_entrenables,
        "parametros_entrenables": entrenables,
        "parametros_no_entrenables": no_entrenables,
        "capa_con_mas_parametros": {
            "nombre": mayor["nombre"],
            "parametros": mayor["parametros"],
            "porcentaje_del_total": round(100 * mayor["parametros"] / entrenables, 2),
        },
        "parametros_por_tipo": {
            t: {"parametros": p, "porcentaje_del_total": round(100 * p / entrenables, 2)} for t, p in por_tipo.items()
        },
        "compilacion": {
            "optimizador": "adam",
            "funcion_perdida": "categorical_crossentropy",
            "metricas": ["accuracy"],
        },
        "semilla": datos.SEMILLA,
    }


# ---------------------------------------------------------------- figura

COLORES = {  # paleta categórica validada (contraste reforzado con el tipo escrito en cada caja)
    "entrada": "#8a8a8a",
    "convolución": "#2a78d6",
    "pooling": "#eb6834",
    "dropout": "#1baf7a",
    "flatten": "#eda100",
    "densa": "#e87ba4",
}


def figura_arquitectura(modelo, ruta):
    """Diagrama de bloques propio: una caja por capa con nombre, tipo y forma de salida, en dos filas."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

    cajas = [("entrada", "Input", "entrada", _forma_texto(FORMA_ENTRADA))]
    for layer in modelo.layers:
        tipo = TIPOS[type(layer).__name__]
        cajas.append((layer.name, detalle_capa(layer), tipo, _forma_texto(_forma(layer))))

    por_fila = 7
    filas = [cajas[:por_fila], cajas[por_fila:]]
    ancho, alto, sep_x, sep_y = 1.9, 1.25, 0.55, 1.15

    fig, eje = plt.subplots(figsize=(15, 5.6))
    centros = []
    for f, fila in enumerate(filas):
        y = -f * (alto + sep_y)
        for c, (nombre, detalle, tipo, forma) in enumerate(fila):
            x = c * (ancho + sep_x)
            color = COLORES[tipo]
            eje.add_patch(FancyBboxPatch((x, y), ancho, alto, boxstyle="round,pad=0.02,rounding_size=0.08",
                                         facecolor=color + "33", edgecolor=color, linewidth=2))
            eje.text(x + ancho / 2, y + alto * 0.74, nombre, ha="center", va="center",
                     fontsize=10.5, fontweight="bold", color="#1a1a1a")
            eje.text(x + ancho / 2, y + alto * 0.47, detalle, ha="center", va="center", fontsize=8.5, color="#333333")
            eje.text(x + ancho / 2, y + alto * 0.2, forma, ha="center", va="center", fontsize=10, color="#1a1a1a",
                     family="monospace")
            centros.append((x, y))

    # Flechas entre capas consecutivas; el salto de fila es un codo por el espacio entre filas
    gris = "#555555"
    for i in range(len(centros) - 1):
        (x0, y0), (x1, y1) = centros[i], centros[i + 1]
        if y0 == y1:
            ini, fin = (x0 + ancho, y0 + alto / 2), (x1, y1 + alto / 2)
        else:
            y_medio = y0 - sep_y / 2
            eje.plot([x0 + ancho / 2, x0 + ancho / 2, x1 + ancho / 2], [y0, y_medio, y_medio],
                     color=gris, linewidth=1.3, solid_joinstyle="round")
            ini, fin = (x1 + ancho / 2, y_medio), (x1 + ancho / 2, y1 + alto)
        eje.add_patch(FancyArrowPatch(ini, fin, arrowstyle="-|>", mutation_scale=13, color=gris,
                                      linewidth=1.3, shrinkA=0, shrinkB=2))

    leyenda = [Patch(facecolor=COLORES[t] + "33", edgecolor=COLORES[t], linewidth=2, label=t.capitalize())
               for t in COLORES]
    eje.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(leyenda),
               frameon=False, fontsize=10, title="Tipo de capa", title_fontsize=10)
    total = int(sum(np.prod(w.shape) for w in modelo.trainable_weights))
    eje.set_title(f"Arquitectura de la CNN (forma de salida de cada capa; "
                  f"{total:,} parámetros entrenables)".replace(",", "."), fontsize=13, pad=12)
    eje.set_xlim(-0.2, por_fila * (ancho + sep_x) - sep_x + 0.2)
    eje.set_ylim(-(alto + sep_y) - 0.2, alto + 0.2)
    eje.set_aspect("equal")
    eje.axis("off")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- principal

def main():
    parser = argparse.ArgumentParser(description="Etapa 2: construcción de la CNN")
    parser.add_argument("--origen", default=os.environ.get("FASHION_MNIST_DIR", str(datos.RAIZ_REPO / "archive")))
    args = parser.parse_args()

    modelo = construir_modelo()
    resumen = resumen_texto(modelo)
    print(resumen)
    (datos.DIR_SALIDAS / "resumen_modelo.txt").write_text(resumen, encoding="utf-8")

    arquitectura = describir_arquitectura(modelo)
    arquitectura["verificacion_parametros"] = verificar_parametros(modelo)
    assert all(v["coincide"] for v in arquitectura["verificacion_parametros"].values()), "parámetros no coinciden"

    d = datos.cargar_datos(Path(args.origen).resolve())
    arquitectura["prueba_forward"] = prueba_forward(modelo, d["x_test"])

    datos.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura_arquitectura(modelo, datos.DIR_FIGURAS / "03_arquitectura.png")
    datos.actualizar_resultados("arquitectura", arquitectura)

    print("Verificación de parámetros:", arquitectura["verificacion_parametros"])
    print("Prueba forward:", arquitectura["prueba_forward"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
