% =========================================================================
% Script  : caso1_cancer_patternnet.m
% Título  : Clasificación de Cáncer de Mama con Red de Reconocimiento de Patrones
% Descripción: Implementa una red neuronal feedforward (patternnet) sobre el
%   dataset 'cancer_dataset' incluido en el Neural Network Toolbox de MATLAB.
%   Se comparan dos configuraciones de entrenamiento para evaluar el impacto
%   de la función de entrenamiento y el tamaño de la capa oculta.
% Objetivos:
%   1. Cargar y validar la integridad del dataset cancer_dataset.
%   2. Entrenar una patternnet con dos configuraciones distintas.
%   3. Comparar métricas de clasificación (precisión, matriz de confusión).
% Requisitos: MATLAB R2018b+ con Neural Network Toolbox.
% Autor   : [Juan Diego Paredes Gámez.]
% Fecha   : Abril 2026
% =========================================================================

clc; clear; close all;
rng(42); % CAMBIO: Semilla fija para reproducibilidad de resultados

%% -------------------------------------------------------------------------
%  SECCIÓN 1: CARGA Y VALIDACIÓN DE DATOS
% -------------------------------------------------------------------------
fprintf('=== CASO 1: Clasificación con cancer_dataset (patternnet) ===\n\n');

try
    [X, T] = cancer_dataset();
    fprintf('[OK] Dataset cargado: %d muestras | %d entradas | %d clases.\n', ...
            size(X,2), size(X,1), size(T,1));
catch ME
    error('[ERROR] Fallo al cargar cancer_dataset: %s', ME.message);
end

% Verificar ausencia de valores no válidos
if any(isnan(X(:))) || any(isnan(T(:)))
    error('[ERROR] El dataset contiene valores NaN.');
end

warning('off', 'nnet:trainlm:ChangedPerformanceFcn');
fprintf('[OK] Validación completada. (Nota: trainlm usará MSE por compatibilidad).\n\n');

%% -------------------------------------------------------------------------
%  SECCIÓN 2: DEFINICIÓN DE CONFIGURACIONES A COMPARAR
%  Se varían simultáneamente: función de entrenamiento y neuronas en capa oculta.
% -------------------------------------------------------------------------
configs(1) = struct( ...
    'nombre',     'Config A: trainlm  | 10 neuronas', ...
    'trainFcn',   'trainlm', ...
    'hiddenSize', 10);

configs(2) = struct( ...
    'nombre',     'Config B: trainscg | 20 neuronas', ...
    'trainFcn',   'trainscg', ...
    'hiddenSize', 20);

nConfigs   = numel(configs);
resultados = cell(nConfigs, 1);

%% -------------------------------------------------------------------------
%  SECCIÓN 3: BUCLE DE ENTRENAMIENTO, EVALUACIÓN Y ALMACENAMIENTO
% -------------------------------------------------------------------------
for k = 1:nConfigs

    fprintf('--- %s ---\n', configs(k).nombre);

    % -- 3.1  Creación de la red feedforward de reconocimiento de patrones
    net = patternnet(configs(k).hiddenSize, configs(k).trainFcn);

    % -- 3.2  División de datos: 70% entrenamiento / 15% validación / 15% prueba
    net.divideParam.trainRatio = 0.70;
    net.divideParam.valRatio   = 0.15;
    net.divideParam.testRatio  = 0.15;

    % -- 3.3  Hiperparámetros de entrenamiento
    net.trainParam.epochs     = 500;    % Máximo de épocas
    net.trainParam.goal       = 1e-5;   % Objetivo de error mínimo
    net.trainParam.max_fail   = 10;     % Paciencia para early-stopping
    net.trainParam.showWindow = false;  % Suprimir GUI para salida limpia

    % -- 3.4  Entrenamiento de la red
    [net, tr] = train(net, X, T);

    % -- 3.5  Evaluación global
    Y      = net(X);
    acc    = sum(vec2ind(Y) == vec2ind(T)) / size(T,2) * 100;

    % -- 3.6  Evaluación exclusiva sobre el conjunto de prueba
    Xtest   = X(:, tr.testInd);
    Ttest   = T(:, tr.testInd);
    Ytest   = net(Xtest);
    accTest = sum(vec2ind(Ytest) == vec2ind(Ttest)) / numel(tr.testInd) * 100;

    fprintf('  Épocas entrenadas  : %d\n',      tr.num_epochs);
    fprintf('  Precisión total    : %.2f%%\n',   acc);
    fprintf('  Precisión (test)   : %.2f%%\n\n', accTest);

    % -- 3.7  Guardar resultados para visualización posterior
    resultados{k} = struct('net', net, 'tr', tr, 'Y', Y, 'T', T, ...
                           'acc', acc, 'accTest', accTest);
end

%% -------------------------------------------------------------------------
%  SECCIÓN 4: VISUALIZACIÓN COMPARATIVA
% -------------------------------------------------------------------------

% 4.1 Matrices de confusión (una por configuración)
figure('Name', 'Caso 1 – Matrices de Confusión', 'NumberTitle', 'off', ...
       'Position', [100 100 900 400]);
for k = 1:nConfigs
    subplot(1, nConfigs, k);
    plotconfusion(resultados{k}.T, resultados{k}.Y);
    title(sprintf('%s\nPrecisión test: %.2f%%', configs(k).nombre, ...
          resultados{k}.accTest), 'FontSize', 8);
end

% 4.2 Curva de rendimiento del mejor modelo
[~, iBest] = max(cellfun(@(r) r.accTest, resultados));
figure('Name', 'Caso 1 – Curva de Entrenamiento (Mejor Configuración)', ...
       'NumberTitle', 'off');
plotperform(resultados{iBest}.tr);
title(sprintf('Curva de Rendimiento – %s', configs(iBest).nombre));

% 4.3 Resumen comparativo en consola
fprintf('╔══════════════════════════════════════════════════════╗\n');
fprintf('║          RESUMEN COMPARATIVO – CASO 1               ║\n');
fprintf('╠══════════════════════════════════════════════════════╣\n');
for k = 1:nConfigs
    fprintf('║  %-40s : %.2f%% ║\n', configs(k).nombre, resultados{k}.accTest);
end
fprintf('╚══════════════════════════════════════════════════════╝\n');
fprintf('\n[✓] Mejor configuración: %s  (%.2f%% en test)\n\n', ...
        configs(iBest).nombre, resultados{iBest}.accTest);
