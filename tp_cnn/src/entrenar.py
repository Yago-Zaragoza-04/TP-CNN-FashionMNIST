"""Etapa 3 — Proceso de aprendizaje de la CNN.

Uso:
    python src/entrenar.py --prueba         # mide 1 época; no guarda nada
    python src/entrenar.py                  # entrenamiento completo
    python src/entrenar.py --solo-figuras   # regenera las figuras 04-06 desde lo guardado

La ruta de los datos se toma de --origen, de FASHION_MNIST_DIR o de archive/ (ver datos.py).
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

import datos
from modelo import construir_modelo

# Hiperparámetros del proceso de aprendizaje
BATCH = 128
# Primera corrida: 20 épocas y paciencia 3. La mejor época coincidió con el tope y val_loss oscilaba
# sin tendencia, así que se amplió a 50 épocas con paciencia 5 (ver "corrida_previa" en resultados.json)
EPOCAS_MAX = 50
PACIENCIA = 5
N_VALIDACION = 6000

SALIDAS = datos.DIR_SALIDAS
RUTA_INDICES_VAL = SALIDAS / "indices_validacion.npy"
RUTA_MEJOR = SALIDAS / "mejor_modelo.keras"
RUTA_MODELO = SALIDAS / "modelo_fashion_cnn.keras"
RUTA_LOG = SALIDAS / "entrenamiento_log.csv"
RUTA_HISTORIAL = SALIDAS / "historial.json"

COLOR_TRAIN, COLOR_VAL = "#2a78d6", "#eb6834"  # mismos colores que la figura 02


class TiempoEpoca(keras.callbacks.Callback):
    """Registra la duración de cada época en segundos (también la agrega a los logs)."""

    def on_train_begin(self, logs=None):
        self.tiempos = []

    def on_epoch_begin(self, epoch, logs=None):
        self._inicio = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        segundos = time.perf_counter() - self._inicio
        self.tiempos.append(segundos)
        if logs is not None:
            logs["segundos"] = segundos


def particionar(y_train):
    """Separa el train oficial en entrenamiento y validación, estratificado por clase."""
    indices = np.arange(len(y_train))
    idx_entr, idx_val = train_test_split(indices, test_size=N_VALIDACION, stratify=y_train,
                                         random_state=datos.SEMILLA)
    por_clase = N_VALIDACION // datos.NUM_CLASES
    assert np.bincount(y_train[idx_val], minlength=datos.NUM_CLASES).tolist() == [por_clase] * datos.NUM_CLASES
    assert len(np.intersect1d(idx_entr, idx_val)) == 0
    # Si ya hay una partición guardada, tiene que ser exactamente la misma
    if RUTA_INDICES_VAL.exists():
        assert np.array_equal(np.load(RUTA_INDICES_VAL), idx_val), "la partición no coincide con la guardada"
    return idx_entr, idx_val


def hardware():
    """Describe el equipo donde se entrenó."""
    procesador = platform.processor()
    if platform.system() == "Windows":
        try:
            import winreg

            clave = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            procesador = winreg.QueryValueEx(clave, "ProcessorNameString")[0].strip()
        except OSError:
            pass

    def _powershell(comando):
        try:
            salida = subprocess.run(["powershell", "-NoProfile", "-Command", comando],
                                    capture_output=True, text=True, timeout=30)
            return int(salida.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    nucleos = ram = None
    if platform.system() == "Windows":
        nucleos = _powershell("(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum")
        ram = _powershell("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
    return {
        "sistema_operativo": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "procesador": procesador,
        "nucleos_fisicos": nucleos,
        "nucleos_logicos": os.cpu_count(),
        "ram_gb": round(ram / 1024 ** 3, 1) if ram else None,
        "gpu": "sin GPU (TensorFlow ≥ 2.11 no soporta GPU en Windows nativo)",
        "gpus_visibles_para_tensorflow": len(tf.config.list_physical_devices("GPU")),
    }


# ---------------------------------------------------------------- figuras

def figura_curva(historial, metrica, mejor_epoca, ruta, titulo, etiqueta_y):
    """Curva de entrenamiento vs. validación por época, con la mejor época y la de detención marcadas."""
    import matplotlib.pyplot as plt

    epocas = np.arange(1, len(historial[metrica]) + 1)
    entr, val = historial[metrica], historial[f"val_{metrica}"]
    detencion = int(epocas[-1])

    fig, eje = plt.subplots(figsize=(9, 4.8))
    eje.plot(epocas, entr, color=COLOR_TRAIN, linewidth=2, marker="o", markersize=4, label="Entrenamiento")
    eje.plot(epocas, val, color=COLOR_VAL, linewidth=2, marker="s", markersize=4, label="Validación")
    eje.axvline(mejor_epoca, color="#555555", linestyle=":", linewidth=1.5)
    eje.annotate(f"mejor época: {mejor_epoca}\n(mínima pérdida de validación;\nse restauran sus pesos)",
                 xy=(mejor_epoca, 0.55), xycoords=("data", "axes fraction"), xytext=(-8, 0),
                 textcoords="offset points", ha="right", va="center", fontsize=9, color="#333333")
    if detencion != mejor_epoca:
        eje.axvline(detencion, color="#555555", linestyle="--", linewidth=1.2)
        eje.annotate(f"detención: época {detencion}\n(early stopping)", xy=(detencion, 0.3),
                     xycoords=("data", "axes fraction"), xytext=(-8, 0), textcoords="offset points",
                     ha="right", va="center", fontsize=9, color="#333333")
    eje.set_xticks(epocas if len(epocas) <= 25 else epocas[::2])
    eje.set_xlabel("Época")
    eje.set_ylabel(etiqueta_y)
    # Coma decimal, con la misma cantidad de decimales en todas las marcas (según el paso entre ellas)
    marcas = eje.get_yticks()
    decimales = len(f"{marcas[1] - marcas[0]:.6f}".rstrip("0").split(".")[1])
    eje.set_yticks(marcas)
    eje.yaxis.set_major_formatter(lambda v, _: f"{v:.{decimales}f}".replace(".", ","))
    eje.set_title(titulo)
    eje.grid(color="#e0e0e0", linewidth=0.8)
    eje.set_axisbelow(True)
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    eje.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def mapas_mas_activos(mapas, n):
    """Índices de los n filtros con mayor activación media para la imagen (evita mostrar mapas vacíos)."""
    medias = mapas.mean(axis=(0, 1))
    return sorted(np.argsort(medias)[::-1][:n].tolist())


def figura_activaciones(modelo, imagen, indice, clase, ruta, n=8):
    """Imagen de test y 8 mapas de activación de conv1_1 y de conv2_2 del modelo entrenado."""
    import matplotlib.pyplot as plt

    capas = ["conv1_1", "conv2_2"]
    extractor = keras.Model(inputs=modelo.inputs, outputs=[modelo.get_layer(c).output for c in capas])
    salidas = [s[0] for s in extractor.predict(imagen[np.newaxis], verbose=0)]

    # Columna 0: rótulo de la fila; columna 1: imagen de entrada; columnas 2 en adelante: mapas
    fig, ejes = plt.subplots(2, n + 2, figsize=(1.75 * (n + 2), 4.8),
                             gridspec_kw={"width_ratios": [1.1, 1] + [1] * n})
    for eje in (ejes[0, 0], ejes[1, 0], ejes[1, 1]):
        eje.axis("off")
    datos.mostrar_imagen(ejes[0, 1], imagen, titulo=f"Entrada\nn.º {indice} ({datos.CLASES[clase]})")
    ejes[0, 1].title.set_fontsize(13)
    filtros_usados = {}
    for fila, (capa, mapas) in enumerate(zip(capas, salidas)):
        filtros = mapas_mas_activos(mapas, n)
        filtros_usados[capa] = filtros
        for j, f in enumerate(filtros):
            eje = ejes[fila, j + 2]
            eje.imshow(mapas[:, :, f], cmap="viridis")
            eje.set_xticks([])
            eje.set_yticks([])
            eje.set_title(f"filtro {f}", fontsize=13)
        alto, ancho, total = mapas.shape
        ejes[fila, 0].text(1.0, 0.5, f"{capa}\n{alto}×{ancho}\n({n} de {total})", ha="right", va="center",
                           fontsize=14, transform=ejes[fila, 0].transAxes)
    fig.suptitle("Mapas de activación del modelo entrenado: primera y última convolución", fontsize=16)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return filtros_usados


def generar_figuras(historial, mejor_epoca, modelo, d):
    """Figuras 04, 05 y 06. Devuelve los datos de la figura 04 para resultados.json."""
    datos.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura_curva(historial, "accuracy", mejor_epoca, datos.DIR_FIGURAS / "05_curva_precision.png",
                 "Exactitud (accuracy) por época: entrenamiento vs. validación", "Exactitud (accuracy)")
    figura_curva(historial, "loss", mejor_epoca, datos.DIR_FIGURAS / "06_curva_perdida.png",
                 "Pérdida (entropía cruzada) por época: entrenamiento vs. validación", "Pérdida")
    # Imagen de la figura 04: la primera bota del conjunto de prueba (criterio fijo, sin mirar predicciones)
    clase = 9
    indice = int(np.flatnonzero(d["y_test"] == clase)[0])
    filtros = figura_activaciones(modelo, d["x_test"][indice], indice, clase,
                                  datos.DIR_FIGURAS / "04_activaciones.png")
    return {
        "indice_test": indice,
        "clase": datos.CLASES[clase],
        "criterio": "primera imagen de la clase Bota en el conjunto de prueba",
        "filtros_mostrados": filtros,
        "criterio_filtros": "los 8 filtros con mayor activación media para esa imagen",
    }


# ---------------------------------------------------------------- resumen

def indicadores_curvas(historial):
    """Indicadores para leer las curvas sin mirarlas a ojo (épocas numeradas desde 1)."""
    acc, val_acc = historial["accuracy"], historial["val_accuracy"]
    loss, val_loss = historial["loss"], historial["val_loss"]
    return {
        "epocas_con_accuracy_train_mayor_que_val": [e + 1 for e in range(len(acc)) if acc[e] > val_acc[e]],
        "epocas_con_loss_train_menor_que_val": [e + 1 for e in range(len(loss)) if loss[e] < val_loss[e]],
        "epocas_con_val_loss_peor_que_su_minimo_previo": [e + 1 for e in range(1, len(val_loss))
                                                          if val_loss[e] > min(val_loss[:e])],
    }


def resumen_entrenamiento(modelo, historial, tiempos, total, idx_entr, idx_val, y_train, verificacion, hw):
    """Arma la sección 'entrenamiento' de resultados.json a partir del historial real."""
    val_loss = historial["val_loss"]
    mejor = int(np.argmin(val_loss)) + 1
    i = mejor - 1
    acc, val_acc, loss = historial["accuracy"], historial["val_accuracy"], historial["loss"]
    pasos = math.ceil(len(idx_entr) / BATCH)
    return {
        "hiperparametros": {
            "optimizador": "adam",
            "learning_rate": float(keras.ops.convert_to_numpy(modelo.optimizer.learning_rate)),
            "learning_rate_nota": "valor por defecto de Adam en Keras",
            "funcion_perdida": "categorical_crossentropy",
            "batch_size": BATCH,
            "epocas_maximas": EPOCAS_MAX,
            "early_stopping": {"monitor": "val_loss", "paciencia": PACIENCIA, "restore_best_weights": True},
            "shuffle": True,
        },
        "particion": {
            "metodo": "train_test_split estratificado (stratify=y_train, random_state=42) sobre el train oficial",
            "n_entrenamiento": int(len(idx_entr)),
            "n_validacion": int(len(idx_val)),
            "validacion_por_clase": {datos.CLASES[c]: int(n) for c, n in
                                     enumerate(np.bincount(y_train[idx_val], minlength=datos.NUM_CLASES))},
            "archivo_indices_validacion": RUTA_INDICES_VAL.name,
            "test_oficial": "no se usa en esta etapa (salvo una imagen para la figura 04)",
        },
        "pasos_por_epoca": pasos,
        "tamano_ultimo_lote": int(len(idx_entr) - (pasos - 1) * BATCH),
        "epocas_ejecutadas": len(val_loss),
        "corte_por_early_stopping": len(val_loss) < EPOCAS_MAX,
        "mejor_epoca": mejor,
        "en_mejor_epoca": {
            "accuracy_entrenamiento": acc[i],
            "accuracy_validacion": val_acc[i],
            "loss_entrenamiento": loss[i],
            "loss_validacion": val_loss[i],
        },
        "nota_metricas_entrenamiento": ("accuracy y loss de entrenamiento son el promedio de la época con "
                                        "dropout activo; las de validación se calculan al final de la época "
                                        "sin dropout"),
        **indicadores_curvas(historial),
        "verificacion_pesos_restaurados": verificacion,
        "tiempo": {
            "segundos_por_epoca": [round(t, 2) for t in tiempos],
            "promedio_s": round(float(np.mean(tiempos)), 2),
            "minimo_s": round(float(np.min(tiempos)), 2),
            "maximo_s": round(float(np.max(tiempos)), 2),
            "total_fit_s": round(total, 2),
            "total_fit_min": round(total / 60, 2),
        },
        "hardware": hw,
        "reproducibilidad": ("semilla 42 con keras.utils.set_random_seed; sin op determinism: en CPU puede "
                             "haber variaciones mínimas entre corridas"),
        "archivos": {
            "mejor_checkpoint": RUTA_MEJOR.name,
            "modelo_final": RUTA_MODELO.name,
            "log_csv": RUTA_LOG.name,
            "historial": RUTA_HISTORIAL.name,
        },
    }


# ---------------------------------------------------------------- principal

def main():
    parser = argparse.ArgumentParser(description="Etapa 3: proceso de aprendizaje")
    parser.add_argument("--origen", default=os.environ.get("FASHION_MNIST_DIR", str(datos.RAIZ_REPO / "archive")))
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--prueba", action="store_true", help="medir una época sin guardar nada")
    modo.add_argument("--solo-figuras", action="store_true", help="regenerar figuras desde lo guardado")
    args = parser.parse_args()

    keras.utils.set_random_seed(datos.SEMILLA)
    d = datos.cargar_datos(Path(args.origen).resolve())

    if args.solo_figuras:
        historial = json.loads(RUTA_HISTORIAL.read_text(encoding="utf-8"))
        mejor = int(np.argmin(historial["val_loss"])) + 1
        figura = generar_figuras(historial, mejor, keras.models.load_model(RUTA_MODELO), d)
        resultados = json.loads(datos.RUTA_RESULTADOS.read_text(encoding="utf-8"))
        resultados["entrenamiento"]["figura_04"] = figura
        resultados["entrenamiento"].update(indicadores_curvas(historial))
        datos.actualizar_resultados("entrenamiento", resultados["entrenamiento"])
        return 0

    idx_entr, idx_val = particionar(d["y_train"])
    x_entr, y_entr = d["x_train"][idx_entr], d["y_train_cat"][idx_entr]
    x_val, y_val = d["x_train"][idx_val], d["y_train_cat"][idx_val]
    print(f"Entrenamiento: {len(idx_entr)} · Validación: {len(idx_val)} (600 por clase)")

    modelo = construir_modelo()
    tiempo = TiempoEpoca()
    if args.prueba:
        callbacks, epocas = [tiempo], 1
    else:
        np.save(RUTA_INDICES_VAL, idx_val)
        callbacks, epocas = [
            tiempo,
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA,
                                          restore_best_weights=True, verbose=1),
            keras.callbacks.ModelCheckpoint(str(RUTA_MEJOR), monitor="val_loss",
                                            save_best_only=True, verbose=1),
            keras.callbacks.CSVLogger(str(RUTA_LOG)),
        ], EPOCAS_MAX

    inicio = time.perf_counter()
    h = modelo.fit(x_entr, y_entr, validation_data=(x_val, y_val), batch_size=BATCH,
                   epochs=epocas, shuffle=True, callbacks=callbacks, verbose=2)
    total = time.perf_counter() - inicio

    if args.prueba:
        print(f"PRUEBA: 1 época en {tiempo.tiempos[0]:.1f} s (no se guardó nada)")
        return 0

    # Grabado del modelo final (con los mejores pesos restaurados por EarlyStopping)
    modelo.save(RUTA_MODELO)
    historial = {k: [float(v) for v in vals] for k, vals in h.history.items() if k != "segundos"}
    historial["segundos"] = [float(t) for t in tiempo.tiempos]
    RUTA_HISTORIAL.write_text(json.dumps(historial, ensure_ascii=False, indent=2), encoding="utf-8")

    # Control: el modelo guardado debe reproducir la pérdida de validación de la mejor época
    loss_val, acc_val = modelo.evaluate(x_val, y_val, batch_size=BATCH, verbose=0)
    mejor_val_loss = min(historial["val_loss"])
    verificacion = {
        "val_loss_mejor_epoca": mejor_val_loss,
        "val_loss_modelo_guardado": float(loss_val),
        "val_accuracy_modelo_guardado": float(acc_val),
        "coincide": abs(float(loss_val) - mejor_val_loss) < 1e-3,
    }
    print("Verificación de pesos restaurados:", verificacion)

    resumen = resumen_entrenamiento(modelo, historial, tiempo.tiempos, total, idx_entr, idx_val,
                                    d["y_train"], verificacion, hardware())
    resumen["figura_04"] = generar_figuras(historial, resumen["mejor_epoca"], modelo, d)
    # Se conserva el resumen de la corrida anterior, si existe
    if datos.RUTA_RESULTADOS.exists():
        previo = json.loads(datos.RUTA_RESULTADOS.read_text(encoding="utf-8")).get("entrenamiento", {})
        if "corrida_previa" in previo:
            resumen["corrida_previa"] = previo["corrida_previa"]
    datos.actualizar_resultados("entrenamiento", resumen)
    print(json.dumps({k: resumen[k] for k in ("epocas_ejecutadas", "mejor_epoca", "en_mejor_epoca", "tiempo")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
