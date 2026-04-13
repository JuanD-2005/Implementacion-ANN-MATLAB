# ANN Portfolio in MATLAB

![MATLAB](https://img.shields.io/badge/MATLAB-R2018b%2B-blue?style=for-the-badge)
![Deep Learning Toolbox](https://img.shields.io/badge/Deep_Learning_Toolbox-Required-orange?style=for-the-badge)
![Project Type](https://img.shields.io/badge/Portfolio-AI%20%7C%20ML-success?style=for-the-badge)

End-to-end MATLAB portfolio project with three Artificial Neural Network workflows:

- Binary classification (Pattern Recognition Network).
- Continuous regression (Fitting Network).
- Unsupervised clustering (Self-Organizing Map).

Built as an academic AI evaluation with reproducible experiments and comparative analysis.

## ES | Resumen rapido

Este portafolio demuestra el uso practico de redes neuronales en tres tareas clasicas de Machine Learning:

1. Clasificacion binaria de cancer de mama.
2. Regresion de porcentaje de grasa corporal.
3. Clustering topologico de Iris con SOM.

Informacion pertinente (fundamentacion teorica, metodologia completa, analisis y discusion de resultados): ver el PDF del repositorio, Implementar ANN con MATLAB - Juan Paredes.pdf

## EN | Quick overview

This portfolio demonstrates practical neural-network workflows across three classic ML tasks:

1. Breast cancer binary classification.
2. Body fat percentage regression.
3. Iris topological clustering with SOM.

Pertinent technical information (theory, full methodology, detailed analysis, and discussion): see the repository PDF, Implementar ANN con MATLAB - Juan Paredes.pdf

## Repository map

```text
.
|- caso1_cancer_patternnet.m   # Classification
|- caso2_bodyfat_fitnet.m      # Regression
|- caso3_iris_som.m            # Clustering (SOM)
`- README.md
```

## Project highlights

| Case | Model | Goal | Key metric (reported) |
|---|---|---|---|
| 1 | patternnet | Binary classification | Test accuracy near 98% |
| 2 | fitnet | Regression | R2 improved from 0.5915 to 0.8038 |
| 3 | selforgmap | Unsupervised clustering | Purity up to 97.33% |

## Quick start

### Requirements

- MATLAB R2018b or newer.
- Deep Learning Toolbox.

### Run all experiments

```matlab
caso1_cancer_patternnet
caso2_bodyfat_fitnet
caso3_iris_som
```

All scripts set rng(42) for reproducibility.

## Outputs you will get

- Confusion matrices and training-performance curves.
- Regression plots and residual analysis.
- SOM topology visualization (U-Matrix style plots) and cluster purity metrics.

## Why this portfolio is useful

- Shows supervised and unsupervised ANN workflows in one place.
- Uses canonical MATLAB datasets for transparent benchmarking.
- Includes practical concerns: compatibility handling, metric comparison, and experiment reproducibility.

## Author

Juan Diego Paredes Gamez  
Electronic Engineering Student - UNET

If this project helped you, consider leaving a star.
