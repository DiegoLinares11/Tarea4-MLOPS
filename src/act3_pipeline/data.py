"""Carga y separación de los datos.

Dataset: ``champions_league_matches.csv`` — el mismo que se analizó en el
Ejercicio 1 y que se usó en la Actividad 1 (UEFA Champions League 2025/26,
151 filas en crudo, 144 partidos reales).

CRISP-DM
--------
**Business Understanding.** Queremos explicar/anticipar el resultado de un
partido de Champions League (gana el local, gana el visitante o empate) a
partir de las estadísticas de juego. Sirve para análisis deportivo y para
alimentar previas de partidos.

**Data Understanding.** Del Ejercicio 1 sabemos que:

- El CSV trae **7 filas totalmente vacías** que separan jornadas y no son
  partidos; hay que eliminarlas (quedan 144 partidos).
- Varias columnas son **numéricas disfrazadas de texto**: la posesión viene
  como ``'63%'`` y los tiros/atajadas como ``'3 of 10'``.
- Las columnas ``score`` y ``winner`` **revelan el resultado** (fuga de
  información o *data leakage*), así que no pueden usarse como entrada.
- ``home_saves_pct`` tiene algunos nulos (casos triviales de "0 of 0").
- La variable objetivo es ``result``, con tres clases desbalanceadas:
  Home Win (71), Away Win (48), Draw (25).

El desbalance de clases es la razón por la que en esta actividad la métrica
que se optimiza durante la calibración es ``f1_macro`` y no ``accuracy``.
"""

from __future__ import annotations

from importlib import resources

import pandas as pd

# --- Variables de entrada, agrupadas por el tratamiento que necesitan ---

# Ya vienen como número (float) en el CSV.
NUMERIC_FEATURES = [
    "home_shots_on_target_pct",
    "away_shots_on_target_pct",
    "home_saves_pct",
    "away_saves_pct",
]

# Vienen como texto tipo '63%'.
PERCENT_FEATURES = ["home_possession", "away_possession"]

# Vienen como texto tipo '3 of 10'.
RATIO_FEATURES = [
    "home_shots_on_target",
    "away_shots_on_target",
    "home_saves",
    "away_saves",
]

# Variables categóricas.
CATEGORICAL_FEATURES = ["home_team", "away_team"]

# Columnas que se descartan.
LEAKAGE_COLUMNS = ["score", "winner"]        # revelan el resultado
DROP_COLUMNS = ["date", "venue", "referee"]  # no aportan al modelo

TARGET = "result"

FEATURES = NUMERIC_FEATURES + PERCENT_FEATURES + RATIO_FEATURES + CATEGORICAL_FEATURES

_CSV_NAME = "champions_league_matches.csv"


def extraer_datos() -> pd.DataFrame:
    """Carga el CSV de partidos empaquetado dentro de la librería."""
    with resources.files("act3_pipeline.datasets").joinpath(_CSV_NAME).open(
        "r", encoding="utf-8"
    ) as fh:
        return pd.read_csv(fh)


def filtrar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Quita filas vacías y columnas que no deben entrar al modelo.

    - Elimina las filas totalmente vacías que separan jornadas.
    - Elimina las filas sin la variable objetivo.
    - Descarta ``score`` y ``winner`` porque revelan el resultado (leakage).
    - Descarta ``date``, ``venue`` y ``referee`` porque no aportan al modelo.
    """
    df = df.dropna(how="all")        # filas separadoras de jornada
    df = df.dropna(subset=[TARGET])  # sin etiqueta no sirve
    a_quitar = [c for c in LEAKAGE_COLUMNS + DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=a_quitar)
    return df.reset_index(drop=True)


def separar_datos(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Divide en X/y y en entrenamiento y prueba.

    Se estratifica por ``result`` para que las tres clases queden
    representadas en ambos conjuntos. El conjunto de prueba **no se toca**
    durante la calibración de hiperparámetros: se reserva para la etapa de
    Evaluation de CRISP-DM.
    """
    from sklearn.model_selection import train_test_split

    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def cargar_datos(test_size: float = 0.2, seed: int = 42):
    """Atajo: extrae, filtra y separa en un solo paso."""
    return separar_datos(filtrar_datos(extraer_datos()), test_size=test_size, seed=seed)
