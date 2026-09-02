"""Transformadores personalizados para el dataset de Champions League.

En el Ejercicio 1 se identificó que varias columnas son **numéricas disfrazadas
de texto**. Estos transformadores hacen esa conversión *dentro* del pipeline,
de modo que la limpieza queda amarrada al modelo y se aplica igual durante la
validación cruzada, en el conjunto de prueba y en cualquier dato nuevo.

- ``PorcentajeATexto``:  '63%'    -> 63.0
- ``RatioATexto``:       '3 of 10' -> 3.0 (efectivos) y 10.0 (intentos)

Que estos pasos vivan dentro del pipeline es lo que permite que
``GridSearchCV`` reajuste **todo** el preprocesamiento en cada partición de la
validación cruzada, sin filtrar información del *fold* de validación.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PorcentajeATexto(BaseEstimator, TransformerMixin):
    """Convierte columnas de porcentaje en texto ('63%') a número (63.0)."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = (
                X[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
                .replace({"nan": np.nan, "": np.nan})
                .astype(float)
            )
        return X.to_numpy()

    def get_feature_names_out(self, input_features=None):
        nombres = input_features if input_features is not None else self.feature_names_in_
        return np.asarray([f"{c}_pct" for c in nombres])


class RatioATexto(BaseEstimator, TransformerMixin):
    """Convierte '3 of 10' en dos columnas numéricas: efectivos e intentos.

    Por cada columna de entrada se generan dos de salida, por ejemplo
    ``home_saves`` -> ``home_saves_hechos`` y ``home_saves_intentos``.
    """

    _PATRON = r"(\d+)\s*of\s*(\d+)"

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        salida = []
        for col in X.columns:
            partes = X[col].astype(str).str.extract(self._PATRON)
            salida.append(partes[0].astype(float))  # efectivos
            salida.append(partes[1].astype(float))  # intentos
        return pd.concat(salida, axis=1).to_numpy()

    def get_feature_names_out(self, input_features=None):
        nombres = input_features if input_features is not None else self.feature_names_in_
        salida = []
        for c in nombres:
            salida.extend([f"{c}_hechos", f"{c}_intentos"])
        return np.asarray(salida)
