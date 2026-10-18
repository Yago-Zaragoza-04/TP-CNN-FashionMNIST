"""Etapa 4 — Validación del modelo con el test oficial y cotejo de resultados.

Uso:
    python src/validar.py

Carga outputs/modelo_fashion_cnn.keras (no reentrena) y usa por primera vez el test oficial de
10.000 imágenes. Requiere haber corrido antes src/mlp_referencia.py para la comparación CNN vs MLP.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

import datos
from entrenar import BATCH, RUTA_MODELO
from predecir import coma, preparar_imagen, top_k

DIR_PRUEBAS = datos.DIR_SALIDAS / "pruebas"
RUTA_BENCHMARK = datos.DIR_SALIDAS / "benchmark_referencias.json"
NOMBRES = [datos.CLASES[c] for c in range(datos.NUM_CLASES)]
VERDE, ROJO = "#1a7f37", "#c62828"
# Grupo de prendas de torso, entre las que se espera la mayor confusión
TORSO = ["Remera/top", "Buzo", "Abrigo", "Camisa"]


# ---------------------------------------------------------------- métricas

def metricas_por_clase(y, y_pred):
    """classification_report con nombres en español, más la mejor y la peor clase por f1."""
    rep = classification_report(y, y_pred, target_names=NOMBRES, output_dict=True, digits=4)
    por_clase = {n: {k: (int(v) if k == "support" else float(v)) for k, v in rep[n].items()} for n in NOMBRES}
    mejor = max(NOMBRES, key=lambda n: por_clase[n]["f1-score"])
    peor = min(NOMBRES, key=lambda n: por_clase[n]["f1-score"])
    return {
        "por_clase": por_clase,
        "macro_avg": {k: float(v) for k, v in rep["macro avg"].items()},
        "weighted_avg": {k: float(v) for k, v in rep["weighted avg"].items()},
        "accuracy": float(rep["accuracy"]),
        "mejor_clase_f1": {"clase": mejor, "f1": por_clase[mejor]["f1-score"]},
        "peor_clase_f1": {"clase": peor, "f1": por_clase[peor]["f1-score"]},
    }


def analizar_confusion(matriz):
    """Pares (real → predicha) más confundidos y confusiones dentro del grupo de prendas de torso."""
    pares = [(matriz[i, j], i, j) for i in range(len(matriz)) for j in range(len(matriz)) if i != j]
    pares.sort(key=lambda t: (-t[0], t[1], t[2]))
    top3 = [{"real": NOMBRES[i], "predicha": NOMBRES[j], "cantidad": int(n)} for n, i, j in pares[:3]]
    idx_torso = [NOMBRES.index(n) for n in TORSO]
    dentro = sum(int(matriz[i, j]) for i in idx_torso for j in idx_torso if i != j)
    total_errores = int(matriz.sum() - np.trace(matriz))
    return top3, {
        "grupo": TORSO,
        "top3_dentro_del_grupo": [p["real"] in TORSO and p["predicha"] in TORSO for p in top3],
        "errores_dentro_del_grupo": dentro,
        "errores_totales": total_errores,
        "porcentaje_de_los_errores": round(100 * dentro / total_errores, 2) if total_errores else 0.0,
    }


# ---------------------------------------------------------------- figuras

def figura_confusion(matriz, ruta):
    """Matriz de confusión 10x10 con valores absolutos, un solo tono y texto según el fondo."""
    import matplotlib.pyplot as plt

    fig, eje = plt.subplots(figsize=(9, 7.8))
    im = eje.imshow(matriz, cmap="Blues")
    umbral = matriz.max() / 2
    for i in range(len(matriz)):
        for j in range(len(matriz)):
            eje.text(j, i, f"{matriz[i, j]}", ha="center", va="center", fontsize=9,
                     color="white" if matriz[i, j] > umbral else "#1a1a1a")
    eje.set_xticks(range(len(NOMBRES)), NOMBRES, rotation=40, ha="right")
    eje.set_yticks(range(len(NOMBRES)), NOMBRES)
    eje.set_xlabel("Clase predicha")
    eje.set_ylabel("Clase real")
    eje.set_title("Matriz de confusión sobre el conjunto de prueba (10.000 imágenes)")
    barra = fig.colorbar(im, ax=eje, fraction=0.046, pad=0.04)
    barra.set_label("Cantidad de imágenes")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _marco(eje, color):
    for borde in eje.spines.values():
        borde.set_visible(True)
        borde.set_color(color)
        borde.set_linewidth(3)


def figura_ejemplos(x, ejemplos, ruta):
    """Grilla 4x5 de ejemplos cotejados: real, predicho y probabilidad; verde si acierta, rojo si falla."""
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(4, 5, figsize=(11, 13.6))
    for eje, e in zip(ejes.flat, ejemplos):
        color = VERDE if e["acierta"] else ROJO
        datos.mostrar_imagen(eje, x[e["indice"]])
        _marco(eje, color)
        marca = "✓" if e["acierta"] else "✗"
        eje.set_xlabel(f"Real: {e['clase_real']}\nPredicho: {e['clase_predicha']} {marca}\n"
                       f"p = {coma(e['probabilidad_predicha'])}", fontsize=12, color=color)
        eje.set_title(f"n.º {e['indice']}", fontsize=11, color="#555555")
    aciertos = sum(e["acierta"] for e in ejemplos)
    fig.suptitle(f"Ejemplos cotejados: {aciertos} de {len(ejemplos)} aciertos "
                 "(2 por clase, al azar con semilla 42)", fontsize=16)
    # Espaciado manual: los rótulos de cada fila no deben tocar el número de la fila siguiente.
    # La aclaración de la visualización invertida va en el epígrafe del informe.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.07, wspace=0.18, hspace=0.62)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_errores(x, errores, ruta):
    """Grilla 3x4 con los errores de mayor confianza y su top-3."""
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(3, 4, figsize=(11, 13.8))
    for eje, e in zip(ejes.flat, errores):
        datos.mostrar_imagen(eje, x[e["indice"]])
        _marco(eje, ROJO)
        eje.set_title(f"Real: {e['clase_real']}\nPredicho: {e['clase_predicha']}\n"
                      f"p = {coma(e['probabilidad_predicha'])} · n.º {e['indice']}",
                      fontsize=11.5, color=ROJO)
        top3 = "\n".join(f"{i}. {t['clase']}: {coma(t['probabilidad'], 3)}" for i, t in enumerate(e["top3"], 1))
        eje.set_xlabel(top3, fontsize=10.5, color="#333333")
    fig.suptitle("Los 12 errores con mayor confianza del modelo en el conjunto de prueba", fontsize=16, y=0.985)
    # Espaciado manual: el top-3 bajo cada imagen necesita lugar antes de los títulos de la fila siguiente.
    # La aclaración de la visualización invertida va en el epígrafe del informe.
    fig.subplots_adjust(left=0.03, right=0.97, top=0.87, bottom=0.07, wspace=0.25, hspace=1.05)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- ejemplos

def describir(indice, y, probs):
    """Datos de una predicción individual."""
    ranking = top_k(probs[indice], k=3)
    return {
        "indice": int(indice),
        "clase_real": datos.CLASES[int(y[indice])],
        "clase_predicha": ranking[0][1],
        "probabilidad_predicha": ranking[0][2],
        "top2": [{"clase": n, "probabilidad": p} for _, n, p in ranking[:2]],
        "top3": [{"clase": n, "probabilidad": p} for _, n, p in ranking],
        "acierta": bool(ranking[0][0] == int(y[indice])),
    }


def elegir_ejemplos(y, por_clase=2):
    """2 índices por clase al azar con semilla fija, sin mirar las predicciones."""
    rng = np.random.default_rng(datos.SEMILLA)
    indices = []
    for c in range(datos.NUM_CLASES):
        indices.extend(sorted(rng.choice(np.flatnonzero(y == c), size=por_clase, replace=False).tolist()))
    return indices


def exportar_y_verificar_pruebas(modelo, x, y, probs, ejemplos):
    """Exporta 2 imágenes de test a PNG (una original y otra invertida) y verifica que el
    preprocesamiento de predecir.py reproduce la predicción de esta validación."""
    DIR_PRUEBAS.mkdir(parents=True, exist_ok=True)
    original = ejemplos[0]["indice"]                                         # primer ejemplo (Remera/top)
    invertida = next(e["indice"] for e in ejemplos if e["clase_real"] == "Bota")  # primera bota de la muestra
    pruebas = []
    for indice, invertir in ((original, False), (invertida, True)):
        pixeles = np.round(x[indice, :, :, 0] * 255).astype(np.uint8)
        if invertir:
            pixeles = 255 - pixeles
        ruta = DIR_PRUEBAS / f"test_{indice}_{'invertida' if invertir else 'original'}.png"
        Image.fromarray(pixeles).save(ruta)  # uint8 2D → modo "L" (escala de grises)
        x_png, info = preparar_imagen(ruta)
        p_png = modelo.predict(x_png, verbose=0)[0]
        esperado = describir(indice, y, probs)
        coincide = int(np.argmax(p_png)) == int(np.argmax(probs[indice])) and np.allclose(p_png, probs[indice], atol=1e-5)
        assert coincide, f"predecir.py no reproduce la predicción del índice {indice}"
        pruebas.append({
            "archivo": f"pruebas/{ruta.name}",
            "indice_test": int(indice),
            "formato": "invertida (fondo blanco)" if invertir else "original (fondo negro)",
            "invertida_por_predecir": info["invertida"],
            "clase_real": esperado["clase_real"],
            "prediccion_validar": esperado["clase_predicha"],
            "probabilidad_validar": esperado["probabilidad_predicha"],
            "prediccion_predecir": datos.CLASES[int(np.argmax(p_png))],
            "probabilidad_predecir": float(p_png.max()),
            "coincide": bool(coincide),
        })
    return pruebas


# ---------------------------------------------------------------- comparaciones

def comparar_con_mlp(resultados, acc_test):
    """Tabla CNN vs MLP a partir de resultados.json."""
    mlp = resultados.get("mlp_referencia")
    if not mlp:
        return None
    ent = resultados["entrenamiento"]
    return {"ventaja_cnn_pp": round(100 * (acc_test - mlp["accuracy_test"]), 2), "tabla": [
        {"modelo": "CNN (este trabajo)", "parametros": resultados["arquitectura"]["parametros_entrenables"],
         "accuracy_test": acc_test, "epocas": ent["epocas_ejecutadas"], "mejor_epoca": ent["mejor_epoca"],
         "tiempo_entrenamiento_s": ent["tiempo"]["total_fit_s"]},
        {"modelo": "MLP de referencia", "parametros": mlp["parametros"], "accuracy_test": mlp["accuracy_test"],
         "epocas": mlp["epocas_ejecutadas"], "mejor_epoca": mlp["mejor_epoca"],
         "tiempo_entrenamiento_s": mlp["tiempo_total_s"]},
    ]}


def ubicar_en_benchmark(acc_cnn, acc_mlp):
    """Ordena nuestras redes junto a las referencias publicadas."""
    if not RUTA_BENCHMARK.exists():
        return None
    refs = json.loads(RUTA_BENCHMARK.read_text(encoding="utf-8"))["referencias"]
    filas = [{"modelo": r["modelo"], "accuracy_test": r["accuracy_test"], "fuente": r["fuente"], "url": r["url"]}
             for r in refs]
    filas.append({"modelo": "CNN (este trabajo)", "accuracy_test": acc_cnn, "fuente": "este trabajo"})
    if acc_mlp is not None:
        filas.append({"modelo": "MLP de referencia (este trabajo)", "accuracy_test": acc_mlp, "fuente": "este trabajo"})
    filas.sort(key=lambda f: -f["accuracy_test"])
    return {
        "tabla_ordenada": filas,
        "posicion_cnn": 1 + [f["modelo"] for f in filas].index("CNN (este trabajo)"),
        "referencias_por_debajo_de_la_cnn": [r["modelo"] for r in refs if r["accuracy_test"] < acc_cnn],
        "referencias_por_encima_de_la_cnn": [r["modelo"] for r in refs if r["accuracy_test"] > acc_cnn],
    }


# ---------------------------------------------------------------- principal

def main():
    parser = argparse.ArgumentParser(description="Etapa 4: validación con el test oficial")
    parser.add_argument("--origen", default=os.environ.get("FASHION_MNIST_DIR", str(datos.RAIZ_REPO / "archive")))
    args = parser.parse_args()

    d = datos.cargar_datos(Path(args.origen).resolve())
    x, y = d["x_test"], d["y_test"]
    modelo = keras.models.load_model(RUTA_MODELO)
    resultados = json.loads(datos.RUTA_RESULTADOS.read_text(encoding="utf-8"))

    # 1. Evaluación global y comparación con la validación de la mejor época
    loss_test, acc_test = modelo.evaluate(x, d["y_test_cat"], batch_size=BATCH, verbose=0)
    val = resultados["entrenamiento"]["en_mejor_epoca"]
    probs = modelo.predict(x, batch_size=BATCH, verbose=0)
    y_pred = probs.argmax(axis=1)
    assert abs(float((y_pred == y).mean()) - acc_test) < 1e-6
    global_ = {
        "accuracy_test": float(acc_test),
        "loss_test": float(loss_test),
        "accuracy_validacion_mejor_epoca": val["accuracy_validacion"],
        "loss_validacion_mejor_epoca": val["loss_validacion"],
        "diferencia_accuracy_pp": round(100 * (float(acc_test) - val["accuracy_validacion"]), 2),
        "diferencia_loss": round(float(loss_test) - val["loss_validacion"], 4),
        "aciertos": int((y_pred == y).sum()),
        "errores": int((y_pred != y).sum()),
        "n_test": int(len(y)),
    }

    # 2. Métricas por clase y 3. matriz de confusión
    por_clase = metricas_por_clase(y, y_pred)
    matriz = confusion_matrix(y, y_pred, labels=list(range(datos.NUM_CLASES)))
    top3, torso = analizar_confusion(matriz)
    figura_confusion(matriz, datos.DIR_FIGURAS / "07_matriz_confusion.png")

    # 4. Ejemplos cotejados (muestra fija, sin mirar predicciones)
    indices = elegir_ejemplos(y)
    ejemplos = [describir(i, y, probs) for i in indices]
    aciertos = sum(e["acierta"] for e in ejemplos)
    figura_ejemplos(x, ejemplos, datos.DIR_FIGURAS / "08_ejemplos_cotejados.png")

    # 5. Errores con mayor confianza
    idx_err = np.flatnonzero(y_pred != y)
    orden = idx_err[np.argsort(-probs[idx_err, y_pred[idx_err]], kind="stable")][:12]
    errores = [describir(i, y, probs) for i in orden]
    figura_errores(x, errores, datos.DIR_FIGURAS / "09_errores.png")

    # 8. Prueba de predecir.py sobre imágenes exportadas
    pruebas = exportar_y_verificar_pruebas(modelo, x, y, probs, ejemplos)

    # 6 y 7. Comparaciones
    tabla_mlp = comparar_con_mlp(resultados, float(acc_test))
    acc_mlp = resultados.get("mlp_referencia", {}).get("accuracy_test")

    datos.actualizar_resultados("validacion", {
        "modelo_evaluado": RUTA_MODELO.name,
        "global": global_,
        **por_clase,
        "matriz_confusion": {"etiquetas": NOMBRES, "filas_real_columnas_predicha": matriz.tolist()},
        "pares_mas_confundidos": top3,
        "confusion_prendas_de_torso": torso,
        "ejemplos": {
            "criterio": "2 imágenes por clase con np.random.default_rng(42) sobre los índices de cada clase",
            "indices": indices,
            "aciertos": aciertos,
            "total": len(ejemplos),
            "aciertos_esperados_segun_accuracy_global": round(len(ejemplos) * float(acc_test), 2),
            "detalle": ejemplos,
        },
        "errores_alta_confianza": {
            "criterio": "los 12 errores del test con mayor probabilidad asignada a la clase equivocada",
            "errores_totales": int(len(idx_err)),
            "indices": [int(i) for i in orden],
            "detalle": errores,
        },
        "comparacion_cnn_mlp": tabla_mlp,
        "benchmark": ubicar_en_benchmark(float(acc_test), acc_mlp),
        "prueba_predecir": pruebas,
    })

    print(json.dumps({"global": global_, "mejor": por_clase["mejor_clase_f1"], "peor": por_clase["peor_clase_f1"],
                      "top3": top3, "torso": torso, "ejemplos": f"{aciertos}/{len(ejemplos)}",
                      "cnn_vs_mlp": tabla_mlp, "pruebas": pruebas}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
