% =========================================================================
% Script  : caso2_bodyfat_fitnet.m
% Título  : Regresión del Porcentaje de Grasa Corporal con Red de Ajuste (fitnet)
% Descripción: Implementa una red neuronal de regresión (fitnet) sobre el dataset
%   'bodyfat_dataset' del Neural Network Toolbox. Se comparan dos configuraciones
%   diferenciadas por función de entrenamiento y profundidad de arquitectura.
% Objetivos:
%   1. Cargar y validar bodyfat_dataset.
%   2. Entrenar una fitnet con dos configuraciones distintas.
%   3. Comparar métricas de regresión: MSE y coeficiente de determinación R².
% Requisitos: MATLAB R2018b+ con Neural Network Toolbox.
% Autor   : [Juan Diego Paredes Gámez.]
% Fecha   : Abril 2026
% =========================================================================

clc; clear; close all;
rng(42); % CAMBIO: Semilla fija para reproducibilidad de resultados

%% -------------------------------------------------------------------------
%  SECCIÓN 1: CARGA Y VALIDACIÓN DE DATOS
% -------------------------------------------------------------------------
fprintf('=== CASO 2: Regresión con bodyfat_dataset (fitnet) ===\n\n');

try
    [X, T] = bodyfat_dataset();
    fprintf('[OK] Dataset cargado: %d muestras | %d variables predictoras.\n', ...
            size(X,2), size(X,1));
catch ME
    error('[ERROR] Fallo al cargar bodyfat_dataset: %s', ME.message);
end

if any(isnan(X(:))) || any(isnan(T(:)))
    error('[ERROR] El dataset contiene valores NaN. Revise la instalación del Toolbox.');
end
fprintf('[OK] Validación de integridad completada (sin NaN).\n\n');

%% -------------------------------------------------------------------------
%  SECCIÓN 2: DEFINICIÓN DE CONFIGURACIONES A COMPARAR
%  Config A: Levenberg-Marquardt, capa simple  → rápido, puede sobreajustar.
%  Config B: Bayesian Regularization, dos capas → más robusto ante overfitting.
% -------------------------------------------------------------------------
configs(1) = struct( ...
    'nombre',     'Config A: trainlm | [10]',   ...
    'trainFcn',   'trainlm',                    ...
    'hiddenSize', 10);

configs(2) = struct( ...
    'nombre',     'Config B: trainbr | [10 5]', ...
    'trainFcn',   'trainbr',                    ...
    'hiddenSize', [10 5]);

nConfigs   = numel(configs);
resultados = cell(nConfigs, 1);

%% -------------------------------------------------------------------------
%  SECCIÓN 3: BUCLE DE ENTRENAMIENTO, EVALUACIÓN Y ALMACENAMIENTO
% -------------------------------------------------------------------------
for k = 1:nConfigs

    fprintf('--- %s ---\n', configs(k).nombre);

    % -- 3.1  Creación de la red de ajuste (función de activación tansig + purelin)
    net = fitnet(configs(k).hiddenSize, configs(k).trainFcn);

    % CAMBIO: trainbr usa todos los datos internamente; divideFcn se desactiva.
    if strcmp(configs(k).trainFcn, 'trainbr')
        net.divideFcn = 'dividenone';
    else
        % -- 3.2  División de datos
        net.divideParam.trainRatio = 0.70;
        net.divideParam.valRatio   = 0.15;
        net.divideParam.testRatio  = 0.15;
    end

    % -- 3.3  Hiperparámetros de entrenamiento
    net.trainParam.epochs     = 1000;
    net.trainParam.goal       = 1e-6;
    net.trainParam.max_fail   = 15;
    net.trainParam.showWindow = false;

    % -- 3.4  Entrenamiento
    [net, tr] = train(net, X, T);

    % -- 3.5  Evaluación sobre el conjunto de prueba
    Xtest = X(:, tr.testInd);
    Ttest = T(:, tr.testInd);
    Ytest = net(Xtest);

    % Métricas de regresión
    mse_val = mean((Ttest - Ytest).^2);
    rmse    = sqrt(mse_val);
    ss_res  = sum((Ttest - Ytest).^2);
    ss_tot  = sum((Ttest - mean(Ttest)).^2);
    r2      = 1 - ss_res / ss_tot;

    fprintf('  Épocas entrenadas  : %d\n',        tr.num_epochs);
    fprintf('  MSE  (test)        : %.6f\n',       mse_val);
    fprintf('  RMSE (test)        : %.6f\n',       rmse);
    fprintf('  R²   (test)        : %.4f\n\n',     r2);

    resultados{k} = struct('net', net, 'tr', tr, ...
                           'Ytest', Ytest, 'Ttest', Ttest, ...
                           'mse', mse_val, 'rmse', rmse, 'r2', r2);
end

%% -------------------------------------------------------------------------
%  SECCIÓN 4: VISUALIZACIÓN COMPARATIVA
% -------------------------------------------------------------------------

% 4.1 Gráficas de regresión (predicho vs real) para cada configuración
figure('Name', 'Caso 2 – Regresión Comparativa', 'NumberTitle', 'off', ...
       'Position', [100 100 900 420]);
for k = 1:nConfigs
    subplot(1, nConfigs, k);
    plotregression(resultados{k}.Ttest, resultados{k}.Ytest);
    title(sprintf('%s\nR²=%.4f | RMSE=%.4f', ...
          configs(k).nombre, resultados{k}.r2, resultados{k}.rmse), 'FontSize', 8);
end

% 4.2 Gráfica de error residual del mejor modelo
[~, iBest] = min(cellfun(@(r) r.mse, resultados));
figure('Name', 'Caso 2 – Curva de Entrenamiento (Mejor Configuración)', ...
       'NumberTitle', 'off');
plotperform(resultados{iBest}.tr);
title(sprintf('Rendimiento – %s', configs(iBest).nombre));

% 4.3 Residuos del mejor modelo
figure('Name', 'Caso 2 – Análisis de Residuos', 'NumberTitle', 'off');
residuos = resultados{iBest}.Ttest - resultados{iBest}.Ytest;
histogram(residuos, 20, 'FaceColor', [0.2 0.5 0.8], 'EdgeColor', 'white');
xlabel('Error Residual (%)'); ylabel('Frecuencia');
title(sprintf('Distribución de Residuos – %s', configs(iBest).nombre));
xline(0, '--r', 'LineWidth', 1.5);

% 4.4 Resumen en consola
fprintf('╔══════════════════════════════════════════════════════════╗\n');
fprintf('║            RESUMEN COMPARATIVO – CASO 2                 ║\n');
fprintf('╠══════════════════════════════════════════════════════════╣\n');
for k = 1:nConfigs
    fprintf('║  %-35s  MSE: %.5f  R²: %.4f ║\n', ...
            configs(k).nombre, resultados{k}.mse, resultados{k}.r2);
end
fprintf('╚══════════════════════════════════════════════════════════╝\n');
fprintf('\n[✓] Mejor configuración: %s  (R²=%.4f)\n\n', ...
        configs(iBest).nombre, resultados{iBest}.r2);
