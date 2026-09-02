# act3-pipeline-mlops

Pipeline de scikit-learn para clasificar el resultado de partidos de la UEFA Champions League,
con calibracion de hiperparametros. Curso de Machine Learning Engineering (MLE/MLOps),
Universidad del Valle de Guatemala.

Diego Linares - Andy Fuentes - Christian Echeverria - Diederich Solis

## Instalacion desde TestPyPI

TestPyPI no tiene las dependencias cientificas (scikit-learn, pandas), asi que hay que
apuntar tambien a PyPI para que las resuelva:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            act3-pipeline-mlops
```

## Uso

El nombre de la distribucion es `act3-pipeline-mlops`, pero el paquete se importa como `act3_pipeline`:

```python
from act3_pipeline import cargar_datos, busqueda_aleatoria, evaluar_en_prueba

X_train, X_test, y_train, y_test = cargar_datos()
busqueda = busqueda_aleatoria(n_iter=60).fit(X_train, y_train)
print(evaluar_en_prueba(busqueda.best_estimator_, X_test, y_test))
```

Tambien trae un comando de linea:

```bash
act3-demo --busqueda aleatoria
```

## Que incluye

- Transformadores propios para las columnas numericas guardadas como texto (`'63%'`, `'3 of 10'`).
- El pipeline completo: preprocesamiento por tipo de variable, seleccion de variables y modelo.
- Tres objetos de busqueda de hiperparametros: rejilla, aleatoria y por mitades sucesivas.
- Funciones de evaluacion: baseline, matriz de confusion, curvas y validacion cruzada anidada.
- El dataset de 144 partidos viaja dentro del paquete.
