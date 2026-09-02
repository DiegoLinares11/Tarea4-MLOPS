"""Calibración de hiperparámetros — etapa **Modeling** de CRISP-DM.

Objetos de scikit-learn que se investigan en esta actividad
-----------------------------------------------------------
``StratifiedKFold``
    Esquema de validación cruzada que conserva la proporción de las tres
    clases en cada partición. Con 144 partidos y solo 25 empates es
    indispensable: un ``KFold`` normal podría dejar *folds* casi sin empates.

``GridSearchCV``
    Prueba **todas** las combinaciones de una rejilla. Exhaustivo y
    reproducible, pero el costo crece multiplicativamente.

``RandomizedSearchCV``
    Muestrea ``n_iter`` combinaciones de distribuciones continuas o discretas.
    Permite comparar **familias de modelos distintas** en una sola búsqueda,
    pasando una lista de diccionarios.

``HalvingGridSearchCV``
    Búsqueda por mitades sucesivas: evalúa muchas combinaciones con pocos
    datos y va descartando las peores. Sigue siendo experimental, por eso hay
    que importar ``sklearn.experimental.enable_halving_search_cv``.

La llave de todo esto es la sintaxis ``paso__hiperparametro`` del ``Pipeline``:
gracias a ella la búsqueda calibra en el mismo barrido el imputador, la
selección de variables y el clasificador, reajustando el preprocesamiento
dentro de cada *fold* (sin fuga de información).

Qué NO se calibra, y por qué
----------------------------
- ``handle_unknown="ignore"`` del One-Hot: no es una perilla de desempeño sino
  un requisito — si aparece un equipo no visto, el pipeline no puede reventar.
- ``StandardScaler``: al bosque le da igual, pero la regresión logística y el
  SVC lo necesitan. Ponerlo siempre no cuesta nada.
- La métrica (``f1_macro``) y el esquema de CV: son decisiones de diseño
  derivadas del problema (clases desbalanceadas, dataset chico). Calibrarlas
  sería elegir la vara con la que uno mismo se mide.
- ``random_state=42``: reproducibilidad entre computadoras.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC

from .pipeline import SEED, construir_pipeline

#: Métrica que se optimiza. Con clases desbalanceadas (71/48/25), ``f1_macro``
#: pesa igual a las tres clases y evita que el modelo gane puntos ignorando
#: los empates, que es justo lo que pasaría optimizando ``accuracy``.
SCORING = "f1_macro"

#: Validación cruzada estratificada, con semilla fija para que la calibración
#: dé el mismo resultado en cualquier computadora.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


# =====================================================================
#  Rejillas para GridSearchCV
# =====================================================================

#: Rejilla completa sobre Random Forest: 2 x 3 x 2 x 2 x 2 x 2 = 96
#: combinaciones x 5 folds = 480 ajustes.
#:
#: Cada hiperparámetro está aquí por una razón concreta. El criterio para
#: incluirlo: (1) controla la capacidad del modelo, que es el problema real de
#: este dataset (115 partidos de entrenamiento contra 86 columnas de entrada),
#: (2) su valor correcto no se puede deducir de antemano, y (3) interactúa con
#: los demás, así que hay que buscarlos juntos.
REJILLA_COMPLETA: dict = {
    # --- hiperparámetros del PREPROCESAMIENTO ---
    # Los nulos vienen de partidos con "0 of 0" atajadas: distribución sesgada
    # y acotada (0-100). La media se corre con los extremos, la mediana no.
    # Son pocos nulos, así que lo que se quiere averiguar es si la decisión
    # importa o da igual.
    "preprocesamiento__numericas__imputar__strategy": ["median", "mean"],
    # 72 de las 86 columnas son el One-Hot de los equipos, y cada equipo
    # aparece en ~6-8 partidos: casi ruido. Cuántas conservar no se sabe de
    # antemano.
    "seleccion__k": [15, 30, "all"],
    # --- hiperparámetros del MODELO ---
    # Más árboles reducen la varianza del promedio y nunca sobreajustan más,
    # pero cuestan tiempo: se busca dónde se aplana la curva.
    "modelo__n_estimators": [200, 400],
    # LA perilla de sobreajuste: sin límite, con 115 filas el árbol llega a
    # hojas puras y memoriza.
    "modelo__max_depth": [None, 6],
    # Con leaf=1 una hoja puede sostenerse en un solo partido raro, y con 20
    # empates en entrenamiento eso es justo lo que hace ruido.
    "modelo__min_samples_leaf": [1, 3],
    # Respuesta directa al desbalance 57/38/20: 'balanced' hace que fallar un
    # empate cueste ~3x más. Es la hipótesis más fuerte de la rejilla.
    "modelo__class_weight": [None, "balanced"],
}

#: Versión reducida (8 combinaciones) para pruebas rápidas y validación
#: cruzada anidada, donde el costo se multiplica.
REJILLA_RAPIDA: dict = {
    "seleccion__k": [15, "all"],
    "modelo__max_depth": [None, 6],
    "modelo__class_weight": [None, "balanced"],
}


# =====================================================================
#  Espacio para RandomizedSearchCV (comparación de familias de modelos)
# =====================================================================

def espacio_aleatorio() -> list[dict]:
    """Espacio de búsqueda con cuatro familias de modelos.

    ``RandomizedSearchCV`` acepta una **lista** de diccionarios: primero
    elige uno al azar y luego muestrea dentro de él. Como el clasificador es
    un paso más del pipeline (``modelo``), se puede sustituir por completo
    igual que cualquier otro hiperparámetro.

    Por qué estos hiperparámetros por familia:

    - **LogisticRegression**: ``C`` es el inverso de la regularización L2. Con
      86 columnas correlacionadas y 115 filas hay que encoger los
      coeficientes, pero *cuánto* se desconoce en órdenes de magnitud — por eso
      ``loguniform`` y no una rejilla lineal.
    - **SVC**: ``C`` (dureza del margen) y ``gamma`` (ancho del kernel) se
      compensan entre sí; es el par clásico que hay que buscar junto.
    - **RandomForest**: se agrega ``max_features``, que descorrelaciona los
      árboles y pesa mucho con 72 columnas casi vacías.
    - **HistGradientBoosting**: ``learning_rate`` y ``max_iter`` son
      inseparables (pasos chicos necesitan más iteraciones); el
      ``min_samples_leaf`` por defecto (20) es enorme para 115 filas.

    Una rejilla exhaustiva sobre estos cuatro espacios tendría miles de
    combinaciones; con 60 muestras se cubre un espacio mucho más ancho.
    """
    comun = {
        "preprocesamiento__numericas__imputar__strategy": ["median", "mean"],
        "seleccion__k": [10, 20, 40, "all"],
    }
    return [
        {
            **comun,
            "modelo": [RandomForestClassifier(random_state=SEED)],
            "modelo__n_estimators": randint(100, 600),
            "modelo__max_depth": [None, 4, 6, 10, 16],
            "modelo__min_samples_leaf": randint(1, 6),
            "modelo__max_features": ["sqrt", "log2", None],
            "modelo__class_weight": [None, "balanced"],
        },
        {
            **comun,
            "modelo": [LogisticRegression(max_iter=5000, random_state=SEED)],
            "modelo__C": loguniform(1e-3, 1e2),
            "modelo__class_weight": [None, "balanced"],
        },
        {
            **comun,
            "modelo": [SVC(random_state=SEED)],
            "modelo__C": loguniform(1e-2, 1e2),
            "modelo__gamma": loguniform(1e-4, 1e0),
            "modelo__kernel": ["rbf", "linear"],
            "modelo__class_weight": [None, "balanced"],
        },
        {
            **comun,
            "modelo": [HistGradientBoostingClassifier(random_state=SEED)],
            "modelo__learning_rate": uniform(0.02, 0.28),
            "modelo__max_iter": randint(80, 400),
            "modelo__max_leaf_nodes": [7, 15, 31],
            "modelo__min_samples_leaf": randint(5, 25),
        },
    ]


# =====================================================================
#  Constructores de las búsquedas
# =====================================================================

def busqueda_grid(rejilla: dict | None = None, n_jobs: int = -1) -> GridSearchCV:
    """``GridSearchCV`` sobre el pipeline completo.

    ``refit=True`` hace que, al terminar, se reentrene el mejor pipeline con
    **todos** los datos de entrenamiento: ``search.best_estimator_`` queda
    listo para predecir.
    """
    return GridSearchCV(
        estimator=construir_pipeline(),
        param_grid=REJILLA_COMPLETA if rejilla is None else rejilla,
        scoring=SCORING,
        cv=CV,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    )


def busqueda_aleatoria(n_iter: int = 60, n_jobs: int = -1) -> RandomizedSearchCV:
    """``RandomizedSearchCV`` sobre cuatro familias de modelos."""
    return RandomizedSearchCV(
        estimator=construir_pipeline(),
        param_distributions=espacio_aleatorio(),
        n_iter=n_iter,
        scoring=SCORING,
        cv=CV,
        n_jobs=n_jobs,
        random_state=SEED,
        refit=True,
        return_train_score=True,
    )


def busqueda_por_mitades(rejilla: dict | None = None, n_jobs: int = -1):
    """``HalvingGridSearchCV``: misma rejilla, menos cómputo.

    Empieza con muchas combinaciones y pocos datos por combinación
    (``resource="n_samples"``) y en cada iteración conserva solo la fracción
    ``1/factor`` de las mejores, dándoles más datos.
    """
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingGridSearchCV

    return HalvingGridSearchCV(
        estimator=construir_pipeline(),
        param_grid=REJILLA_COMPLETA if rejilla is None else rejilla,
        factor=3,
        scoring=SCORING,
        cv=CV,
        n_jobs=n_jobs,
        random_state=SEED,
        refit=True,
    )


# =====================================================================
#  Lectura de resultados
# =====================================================================

def tabla_resultados(search, top: int = 10) -> pd.DataFrame:
    """Convierte ``cv_results_`` en una tabla legible con las mejores filas.

    Se incluye ``mean_train_score`` cuando está disponible porque la brecha
    entre entrenamiento y validación es la señal de sobreajuste que se revisa
    en la etapa de Evaluation.
    """
    res = pd.DataFrame(search.cv_results_)
    columnas = ["rank_test_score", "mean_test_score", "std_test_score"]
    if "mean_train_score" in res.columns:
        columnas.append("mean_train_score")
    columnas.append("params")
    return (
        res[columnas]
        .sort_values("rank_test_score")
        .head(top)
        .reset_index(drop=True)
    )


def resumen_mejor(search) -> dict:
    """Diccionario compacto con el mejor resultado de la búsqueda."""
    idx = int(np.argmin(search.cv_results_["rank_test_score"]))
    return {
        "mejor_score_cv": float(search.best_score_),
        "desviacion_cv": float(search.cv_results_["std_test_score"][idx]),
        "combinaciones_evaluadas": int(len(search.cv_results_["params"])),
        "mejores_parametros": search.best_params_,
    }
