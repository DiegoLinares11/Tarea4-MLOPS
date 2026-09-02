"""act3_pipeline

Paquete de la Actividad 3: calibración de hiperparámetros de un pipeline de
scikit-learn sobre el dataset de la UEFA Champions League (el mismo del
Ejercicio 1 y la Actividad 1), enmarcada en las etapas **Modeling** y
**Evaluation** de CRISP-DM.
"""

from .data import (
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    FEATURES,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    PERCENT_FEATURES,
    RATIO_FEATURES,
    TARGET,
    cargar_datos,
    extraer_datos,
    filtrar_datos,
    separar_datos,
)
from .evaluation import (
    baseline,
    curva_de_aprendizaje,
    curva_de_validacion,
    evaluacion_anidada,
    evaluar_en_prueba,
    importancias,
    matriz_confusion,
)
from .pipeline import (
    SEED,
    construir_pipeline,
    construir_preprocesamiento,
    nombres_de_hiperparametros,
)
from .transformers import PorcentajeATexto, RatioATexto
from .tuning import (
    CV,
    REJILLA_COMPLETA,
    REJILLA_RAPIDA,
    SCORING,
    busqueda_aleatoria,
    busqueda_grid,
    busqueda_por_mitades,
    espacio_aleatorio,
    resumen_mejor,
    tabla_resultados,
)

__all__ = [
    # datos
    "NUMERIC_FEATURES",
    "PERCENT_FEATURES",
    "RATIO_FEATURES",
    "CATEGORICAL_FEATURES",
    "LEAKAGE_COLUMNS",
    "DROP_COLUMNS",
    "FEATURES",
    "TARGET",
    "extraer_datos",
    "filtrar_datos",
    "separar_datos",
    "cargar_datos",
    # pipeline
    "SEED",
    "construir_pipeline",
    "construir_preprocesamiento",
    "nombres_de_hiperparametros",
    "PorcentajeATexto",
    "RatioATexto",
    # calibración
    "SCORING",
    "CV",
    "REJILLA_COMPLETA",
    "REJILLA_RAPIDA",
    "espacio_aleatorio",
    "busqueda_grid",
    "busqueda_aleatoria",
    "busqueda_por_mitades",
    "tabla_resultados",
    "resumen_mejor",
    # evaluación
    "evaluar_en_prueba",
    "baseline",
    "matriz_confusion",
    "curva_de_validacion",
    "curva_de_aprendizaje",
    "evaluacion_anidada",
    "importancias",
]

__version__ = "0.1.0"
