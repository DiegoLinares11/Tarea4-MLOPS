"""Ejecución de extremo a extremo de la calibración.

Este script sirve como *prueba* de que el paquete se instaló y corre en una
computadora: imprime información del equipo (nombre, sistema, versión de
Python) junto con los mejores hiperparámetros encontrados y las métricas del
modelo. Basta con tomarle una captura de pantalla en cada computadora.

Como todas las semillas están fijas, **los mejores hiperparámetros y las
métricas deben salir idénticos en cualquier máquina**.

Uso (después de instalar el paquete):

    act3-demo                 # las tres búsquedas y elige la mejor por CV
    act3-demo --busqueda grid # solo GridSearchCV (96 combinaciones)
    act3-demo --rapido        # rejilla reducida (8 combinaciones)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import platform
import socket
import time
from pathlib import Path

from sklearn.utils import estimator_html_repr

from . import __version__
from .data import cargar_datos, extraer_datos, filtrar_datos
from .evaluation import baseline, evaluar_en_prueba, matriz_confusion
from .tuning import (
    REJILLA_RAPIDA,
    SCORING,
    busqueda_aleatoria,
    busqueda_grid,
    busqueda_por_mitades,
    resumen_mejor,
)


def _banner(texto: str) -> None:
    print("\n" + "=" * 68)
    print(texto)
    print("=" * 68)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibración de hiperparámetros del pipeline (Actividad 3)."
    )
    parser.add_argument(
        "--busqueda",
        choices=["grid", "aleatoria", "mitades", "todas"],
        default="todas",
        help="Objeto de búsqueda a usar (por defecto: todas).",
    )
    parser.add_argument(
        "--rapido",
        action="store_true",
        help="Usa la rejilla reducida de 8 combinaciones (solo con --busqueda grid).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=60,
        help="Combinaciones a muestrear con --busqueda aleatoria.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Procesos en paralelo (-1 = todos los núcleos).",
    )
    parser.add_argument(
        "--diagrama",
        metavar="RUTA",
        default="pipeline_diagram.html",
        help="Archivo HTML donde guardar el diagrama del pipeline.",
    )
    args = parser.parse_args(argv)

    _banner("ACTIVIDAD 3 - Calibracion de hiperparametros (UEFA Champions League)")
    print(f"Paquete act3_pipeline v{__version__}")
    print(f"Equipo (hostname): {socket.gethostname()}")
    print(f"Sistema operativo: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"Fecha/hora: {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")

    # --- Datos (Data Preparation, ya resuelta en la Actividad 1) ---
    crudo = extraer_datos()
    limpio = filtrar_datos(crudo)
    X_train, X_test, y_train, y_test = cargar_datos()
    print(f"\n[datos] CSV crudo   -> {crudo.shape[0]} filas, {crudo.shape[1]} columnas")
    print(f"[datos] Filtrado    -> {limpio.shape[0]} partidos")
    print(f"[datos] Separacion  -> train={len(X_train)}  test={len(X_test)}")

    # --- MODELING: calibracion de hiperparametros ---
    rejilla = REJILLA_RAPIDA if args.rapido else None
    constructores = {
        "GridSearchCV": lambda: busqueda_grid(rejilla, n_jobs=args.n_jobs),
        "RandomizedSearchCV": lambda: busqueda_aleatoria(
            n_iter=args.n_iter, n_jobs=args.n_jobs
        ),
        "HalvingGridSearchCV": lambda: busqueda_por_mitades(rejilla, n_jobs=args.n_jobs),
    }
    elegidas = {
        "grid": ["GridSearchCV"],
        "aleatoria": ["RandomizedSearchCV"],
        "mitades": ["HalvingGridSearchCV"],
        "todas": list(constructores),
    }[args.busqueda]

    print(f"\n[modeling] Validacion cruzada: StratifiedKFold(5), metrica {SCORING}")
    busquedas = {}
    for nombre in elegidas:
        search = constructores[nombre]()
        inicio = time.perf_counter()
        search.fit(X_train, y_train)
        segundos = time.perf_counter() - inicio
        busquedas[nombre] = search
        resumen = resumen_mejor(search)
        print(
            f"[modeling] {nombre:<20} {resumen['combinaciones_evaluadas']:>4} combinaciones"
            f"  {segundos:>6.1f} s   mejor {SCORING} = {resumen['mejor_score_cv']:.3f}"
        )

    # El finalista se elige por el puntaje de VALIDACION CRUZADA, nunca
    # mirando el conjunto de prueba: si no, el test dejaria de ser una
    # medicion independiente.
    ganadora = max(busquedas, key=lambda n: busquedas[n].best_score_)
    search = busquedas[ganadora]
    mejor = search.best_estimator_
    print(f"\n[modeling] Busqueda ganadora: {ganadora}")
    print(f"[modeling] Modelo final: {type(mejor.named_steps['modelo']).__name__}")
    print("[modeling] Mejores hiperparametros:")
    for k, v in sorted(search.best_params_.items()):
        print(f"           {k} = {v}")

    # --- EVALUATION: conjunto reservado y baseline ---
    metricas = evaluar_en_prueba(mejor, X_test, y_test)
    dummy = baseline(X_train, y_train, X_test, y_test)

    _banner(
        f"RESULTADO - accuracy={metricas['accuracy']:.3f}  "
        f"f1_macro={metricas['f1_macro']:.3f}"
    )
    print(
        f"Baseline (clase mayoritaria): accuracy={dummy['accuracy']:.3f}  "
        f"f1_macro={dummy['f1_macro']:.3f}"
    )
    print()
    print(metricas["reporte"])
    print("Matriz de confusion:")
    print(matriz_confusion(mejor, X_test, y_test).to_string())

    # --- Diagrama del pipeline ---
    ruta = Path(args.diagrama)
    ruta.write_text(estimator_html_repr(search), encoding="utf-8")
    print(f"\nDiagrama del pipeline guardado en: {ruta.resolve()}")

    print("\n[OK] Pipeline calibrado y evaluado correctamente en este equipo.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
