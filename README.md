# ANN Portfolio in MATLAB

![MATLAB](https://img.shields.io/badge/MATLAB-R2018b%2B-blue?style=for-the-badge)
![Deep Learning Toolbox](https://img.shields.io/badge/Deep_Learning_Toolbox-Required-orange?style=for-the-badge)
![Project Type](https://img.shields.io/badge/Portfolio-AI%20%7C%20ML-success?style=for-the-badge)

End-to-end MATLAB portfolio project with three Artificial Neural Network workflows: classification, regression, and unsupervised clustering.

## Index / Indice

- [Espanol](#espanol)
- [English](#english)

---

## Espanol

### Resumen

Este repositorio muestra una implementacion completa de redes neuronales artificiales en MATLAB para tres tareas clasicas de Machine Learning:

1. Clasificacion binaria de cancer de mama con patternnet.
2. Regresion de porcentaje de grasa corporal con fitnet.
3. Clustering no supervisado de Iris con selforgmap (SOM).

El proyecto fue desarrollado con enfoque de portafolio tecnico y evaluacion academica, priorizando comparacion experimental, interpretacion de metricas y reproducibilidad.

### Objetivos

1. Implementar y entrenar modelos ANN en problemas supervisados y no supervisados.
2. Comparar configuraciones de entrenamiento y arquitectura.
3. Analizar el impacto de regularizacion y topologia en el rendimiento.
4. Presentar resultados de forma reproducible y visual.

### Estructura del repositorio

```text
.
|- caso1_cancer_patternnet.m   # Clasificacion
|- caso2_bodyfat_fitnet.m      # Regresion
|- caso3_iris_som.m            # Clustering SOM
`- README.md
```

### Casos y hallazgos clave

| Caso | Modelo | Objetivo | Metrica destacada (reportada) |
|---|---|---|---|
| 1 | patternnet | Clasificacion binaria | Precision de prueba cercana a 98% |
| 2 | fitnet | Regresion | R2 mejora de 0.5915 a 0.8038 |
| 3 | selforgmap | Clustering no supervisado | Pureza hasta 97.33% |

### Requisitos

- MATLAB R2018b o superior.
- Deep Learning Toolbox.

### Ejecucion

```matlab
caso1_cancer_patternnet
caso2_bodyfat_fitnet
caso3_iris_som
```

Todos los scripts usan rng(42) para reproducibilidad.

### Salidas esperadas

- Matrices de confusion y curvas de rendimiento.
- Graficas de regresion y analisis de residuos.
- Visualizaciones SOM (U-Matrix) y metricas de pureza.

### Documento tecnico del proyecto

La informacion pertinente (marco teorico, metodologia completa, resultados y discusion) se encuentra en el PDF del repositorio: [Implementar ANN con Matlab - Juan Paredes.pdf](Implementar%20ANN%20con%20Matlab%20-%20Juan%20Paredes.pdf).

---

## English

### Overview

This repository presents a complete MATLAB ANN implementation for three core Machine Learning tasks:

1. Breast cancer binary classification with patternnet.
2. Body fat percentage regression with fitnet.
3. Iris unsupervised clustering with selforgmap (SOM).

The project was built as a technical portfolio and academic evaluation, focusing on experimental comparison, metric interpretation, and reproducibility.

### Objectives

1. Implement and train ANN models for supervised and unsupervised tasks.
2. Compare training configurations and network architectures.
3. Analyze the effect of regularization and map topology on performance.
4. Provide reproducible, visualization-driven results.

### Repository structure

```text
.
|- caso1_cancer_patternnet.m   # Classification
|- caso2_bodyfat_fitnet.m      # Regression
|- caso3_iris_som.m            # SOM clustering
`- README.md
```

### Cases and key findings

| Case | Model | Goal | Highlight metric (reported) |
|---|---|---|---|
| 1 | patternnet | Binary classification | Test accuracy near 98% |
| 2 | fitnet | Regression | R2 improved from 0.5915 to 0.8038 |
| 3 | selforgmap | Unsupervised clustering | Purity up to 97.33% |

### Requirements

- MATLAB R2018b or newer.
- Deep Learning Toolbox.

### Run

```matlab
caso1_cancer_patternnet
caso2_bodyfat_fitnet
caso3_iris_som
```

All scripts use rng(42) for reproducibility.

### Expected outputs

- Confusion matrices and performance curves.
- Regression plots and residual analysis.
- SOM visualizations (U-Matrix) and purity metrics.

### Project technical document

The pertinent information (theoretical background, full methodology, results, and discussion) is available in the repository PDF: [Implementar ANN con Matlab - Juan Paredes.pdf](Implementar%20ANN%20con%20Matlab%20-%20Juan%20Paredes.pdf).

---

## Author

Juan Diego Paredes Gamez  
Electronic Engineering Student - UNET

If this project helped you, consider leaving a star.
