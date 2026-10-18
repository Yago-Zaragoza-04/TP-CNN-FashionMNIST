"""Etapa 4 — Red de referencia sin convolución (MLP), para cotejar qué aporta la convolución.

Uso:
    python src/mlp_referencia.py

Se entrena con la misma partición, semilla, lote, optimizador, pérdida, épocas máximas y paciencia
que la CNN, y se evalúa sobre el test oficial.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import datos
from entrenar import BATCH, EPOCAS_MAX, PACIENCIA, TiempoEpoca, particionar


def construir_mlp():
    """Perceptrón multicapa: aplana la imagen y la pasa por una capa totalmente conectada."""
    datos.fijar_semilla()
    modelo = keras.Sequential(
        [
            keras.Input(shape=(datos.LADO, datos.LADO, 1), name="entrada"),
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="densa"),
            layers.Dropout(0.5, name="drop"),
            layers.Dense(datos.NUM_CLASES, activation="softmax", name="salida"),
        ],
        name="mlp_referencia",
    )
    modelo.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return modelo


def main():
    parser = argparse.ArgumentParser(description="Etapa 4: MLP de referencia")
    parser.add_argument("--origen", default=os.environ.get("FASHION_MNIST_DIR", str(datos.RAIZ_REPO / "archive")))
    args = parser.parse_args()

    keras.utils.set_random_seed(datos.SEMILLA)
    d = datos.cargar_datos(Path(args.origen).resolve())
    idx_entr, idx_val = particionar(d["y_train"])

    modelo = construir_mlp()
    tiempo = TiempoEpoca()
    parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA, restore_best_weights=True)
    inicio = time.perf_counter()
    h = modelo.fit(d["x_train"][idx_entr], d["y_train_cat"][idx_entr],
                   validation_data=(d["x_train"][idx_val], d["y_train_cat"][idx_val]),
                   batch_size=BATCH, epochs=EPOCAS_MAX, shuffle=True, callbacks=[tiempo, parada], verbose=2)
    total = time.perf_counter() - inicio

    loss_test, acc_test = modelo.evaluate(d["x_test"], d["y_test_cat"], batch_size=BATCH, verbose=0)
    val_loss = h.history["val_loss"]
    mejor = int(np.argmin(val_loss)) + 1
    resumen = {
        "arquitectura": "Input(28,28,1) → Flatten → Dense(128, relu) → Dropout(0.5) → Dense(10, softmax)",
        "parametros": int(modelo.count_params()),
        "epocas_ejecutadas": len(val_loss),
        "mejor_epoca": mejor,
        "corte_por_early_stopping": len(val_loss) < EPOCAS_MAX,
        "val_accuracy_mejor_epoca": float(h.history["val_accuracy"][mejor - 1]),
        "val_loss_mejor_epoca": float(val_loss[mejor - 1]),
        "accuracy_test": float(acc_test),
        "loss_test": float(loss_test),
        "tiempo_total_s": round(total, 2),
        "tiempo_promedio_por_epoca_s": round(float(np.mean(tiempo.tiempos)), 2),
        "mismas_condiciones_que_la_cnn": {
            "particion": "indices_validacion.npy", "semilla": datos.SEMILLA, "batch_size": BATCH,
            "optimizador": "adam", "funcion_perdida": "categorical_crossentropy",
            "epocas_maximas": EPOCAS_MAX, "paciencia": PACIENCIA,
        },
    }
    datos.actualizar_resultados("mlp_referencia", resumen)
    print(resumen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
