"""Construcción del pipeline de scikit-learn.

Objetos de scikit-learn que se usan aquí y por qué
--------------------------------------------------
``Pipeline``
    Encadena pasos con nombre. Cada paso expone sus hiperparámetros con la
    sintaxis ``paso__hiperparametro``, y esa es justamente la llave que
    ``GridSearchCV`` usa para calibrarlos.

``ColumnTransformer``
    Aplica un sub-pipeline distinto a cada grupo de columnas. Con
    ``remainder="drop"`` descarta automáticamente lo que no se declaró.
    Sus hiperparámetros se anidan un nivel más:
    ``preprocesamiento__numericas__imputar__strategy``.

``SelectKBest``
    Paso intermedio de selección de variables. Su ``k`` es un hiperparámetro
    del *preprocesamiento*, no del modelo, y aun así se calibra en la misma
    búsqueda. Es el ejemplo más claro de que un pipeline permite optimizar
    preparación y modelo **de forma conjunta**.

El pipeline queda así::

    preprocesamiento (ColumnTransformer)
        ├── numericas    : imputar -> escalar
        ├── porcentajes  : '63%' -> número -> imputar -> escalar
        ├── razones      : '3 of 10' -> 2 números -> imputar -> escalar
        └── categoricas  : imputar -> One-Hot
        -> seleccion (SelectKBest)
        -> modelo (clasificador)
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PERCENT_FEATURES,
    RATIO_FEATURES,
)
from .transformers import PorcentajeATexto, RatioATexto

SEED = 42


def construir_preprocesamiento() -> ColumnTransformer:
    """Arma el preprocesamiento según el tipo de cada variable."""

    # 1) Numéricas que ya vienen como float
    pipeline_numerico = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )

    # 2) Porcentajes guardados como texto ('63%')
    pipeline_porcentaje = Pipeline(
        steps=[
            ("parsear", PorcentajeATexto()),
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )

    # 3) Razones guardadas como texto ('3 of 10')
    pipeline_ratio = Pipeline(
        steps=[
            ("parsear", RatioATexto()),
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )

    # 4) Categóricas (equipos). sparse_output=False para que la salida del
    #    ColumnTransformer sea densa y SelectKBest trabaje sin sorpresas.
    pipeline_categorico = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numericas", pipeline_numerico, NUMERIC_FEATURES),
            ("porcentajes", pipeline_porcentaje, PERCENT_FEATURES),
            ("razones", pipeline_ratio, RATIO_FEATURES),
            ("categoricas", pipeline_categorico, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def construir_pipeline(modelo=None) -> Pipeline:
    """Devuelve el pipeline completo: preprocesamiento + selección + modelo.

    El modelo por defecto es un ``RandomForestClassifier`` con parámetros de
    arranque; los valores buenos salen de la calibración (ver ``tuning.py``).
    """
    if modelo is None:
        modelo = RandomForestClassifier(n_estimators=300, random_state=SEED)
    return Pipeline(
        steps=[
            ("preprocesamiento", construir_preprocesamiento()),
            ("seleccion", SelectKBest(score_func=f_classif, k="all")),
            ("modelo", modelo),
        ]
    )


def nombres_de_hiperparametros(pipe: Pipeline | None = None) -> list[str]:
    """Lista las llaves calibrables del pipeline (``get_params``).

    Útil para el notebook: muestra de dónde salen nombres como
    ``preprocesamiento__numericas__imputar__strategy``.
    """
    pipe = pipe or construir_pipeline()
    return sorted(pipe.get_params(deep=True).keys())
