"""Evaluación del modelo — etapa **Evaluation** de CRISP-DM.

Calibrar hiperparámetros deja un número optimista: ``best_score_`` es el mejor
de decenas de combinaciones probadas sobre las mismas particiones, así que ya
está "contaminado" por la propia búsqueda. Por eso esta actividad evalúa en
tres niveles:

1. **Conjunto de prueba reservado** — nunca se usó durante la búsqueda.
2. **Comparación contra un baseline** (``DummyClassifier``) — un modelo sin
   mérito que siempre predice la clase mayoritaria. Si no se le gana, el
   modelo no aporta nada.
3. **Validación cruzada anidada** — vuelve a calibrar dentro de cada partición
   externa y da una estimación honesta de qué tan bien generaliza *el
   procedimiento completo*, no solo el mejor pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve

from .pipeline import SEED
from .tuning import CV, REJILLA_RAPIDA, SCORING, busqueda_grid


def evaluar_en_prueba(pipe, X_test, y_test) -> dict:
    """Métricas del pipeline ya entrenado sobre el conjunto reservado."""
    y_pred = pipe.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "reporte": classification_report(y_test, y_pred, digits=3, zero_division=0),
        "reporte_dict": classification_report(
            y_test, y_pred, digits=3, zero_division=0, output_dict=True
        ),
    }


def baseline(X_train, y_train, X_test, y_test) -> dict:
    """``DummyClassifier`` que siempre predice la clase mayoritaria."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def matriz_confusion(pipe, X_test, y_test) -> pd.DataFrame:
    """Matriz de confusión con etiquetas, como DataFrame."""
    etiquetas = sorted(pd.unique(y_test))
    m = confusion_matrix(y_test, pipe.predict(X_test), labels=etiquetas)
    return pd.DataFrame(
        m,
        index=[f"real: {e}" for e in etiquetas],
        columns=[f"pred: {e}" for e in etiquetas],
    )


def curva_de_validacion(pipe, X, y, nombre_param: str, valores, n_jobs: int = -1):
    """Curva de validación de **un** hiperparámetro del pipeline.

    Sirve para ver de un vistazo dónde empieza el sobreajuste: la brecha
    entre la curva de entrenamiento y la de validación.
    """
    train_scores, test_scores = validation_curve(
        pipe, X, y, param_name=nombre_param, param_range=valores,
        cv=CV, scoring=SCORING, n_jobs=n_jobs,
    )
    return {
        "valores": list(valores),
        "train_media": train_scores.mean(axis=1),
        "train_std": train_scores.std(axis=1),
        "test_media": test_scores.mean(axis=1),
        "test_std": test_scores.std(axis=1),
    }


def curva_de_aprendizaje(pipe, X, y, n_jobs: int = -1):
    """Curva de aprendizaje: ¿el modelo mejoraría con más partidos?"""
    tamanos, train_scores, test_scores = learning_curve(
        pipe, X, y, train_sizes=np.linspace(0.3, 1.0, 6),
        cv=CV, scoring=SCORING, n_jobs=n_jobs, shuffle=True, random_state=SEED,
    )
    return {
        "tamanos": tamanos,
        "train_media": train_scores.mean(axis=1),
        "test_media": test_scores.mean(axis=1),
        "test_std": test_scores.std(axis=1),
    }


def evaluacion_anidada(X, y, rejilla: dict | None = None, n_jobs: int = -1) -> dict:
    """Validación cruzada anidada (calibración dentro de cada *fold* externo).

    Usa por defecto la rejilla reducida: el costo es
    ``folds_externos x combinaciones x folds_internos``.
    """
    interna = busqueda_grid(REJILLA_RAPIDA if rejilla is None else rejilla, n_jobs=1)
    scores = cross_val_score(interna, X, y, cv=CV, scoring=SCORING, n_jobs=n_jobs)
    return {
        "scores": scores,
        "media": float(scores.mean()),
        "std": float(scores.std()),
    }


def importancias(pipe, X_test, y_test, top: int = 12, n_repeats: int = 20) -> pd.DataFrame:
    """Importancia por permutación, medida sobre el conjunto de prueba.

    Se calcula sobre el pipeline completo, así que las columnas son las
    variables **originales** del CSV y no las 86 columnas transformadas.
    """
    r = permutation_importance(
        pipe, X_test, y_test, n_repeats=n_repeats,
        random_state=SEED, scoring=SCORING,
    )
    return (
        pd.DataFrame(
            {
                "variable": X_test.columns,
                "importancia": r.importances_mean,
                "std": r.importances_std,
            }
        )
        .sort_values("importancia", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )
