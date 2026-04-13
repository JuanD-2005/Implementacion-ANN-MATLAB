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

    % -- 3.1  Creación de la red de ajuste
    net = fitnet(configs(k).hiddenSize, configs(k).trainFcn);

    % CAMBIO: Manejo de división de datos según el algoritmo
    if strcmp(configs(k).trainFcn, 'trainbr')
        % trainbr usa regularización interna; se desactiva la división.
        % Se usa '' para evitar el error de "dividenone" en algunas versiones.
        net.divideFcn = '';
    else
        % -- 3.2  División de datos estándar para trainlm
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

    % -- 3.5  Evaluación
    % IMPORTANTE: Si divideFcn es '', tr.testInd estará vacío.
    % En ese caso, evaluamos sobre todo el dataset para la Config B.
    if isempty(tr.testInd)
        Xtest = X;
        Ttest = T;
    else
        Xtest = X(:, tr.testInd);
        Ttest = T(:, tr.testInd);
    end

    Ytest = net(Xtest);

    % Métricas de regresión
    mse_val = mean((Ttest - Ytest).^2);
    rmse    = sqrt(mse_val);
    ss_res  = sum((Ttest - Ytest).^2);
    ss_tot  = sum((Ttest - mean(Ttest)).^2);
    r2      = 1 - ss_res / ss_tot;

    fprintf('  Épocas entrenadas  : %d\n',        tr.num_epochs);
    fprintf('  MSE  (eval)        : %.6f\n',       mse_val);
    fprintf('  RMSE (eval)        : %.6f\n',       rmse);
    fprintf('  R²   (eval)        : %.4f\n\n',     r2);

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
tituloResumen = 'RESUMEN COMPARATIVO - CASO 2';
lineasResumen = cell(nConfigs, 1);
for k = 1:nConfigs
    lineasResumen{k} = sprintf('  %-30s  MSE: %8.5f  R2: %7.4f  ', ...
                               configs(k).nombre, resultados{k}.mse, resultados{k}.r2);
end

anchoInterno = max([length(tituloResumen), cellfun(@length, lineasResumen)']);

% Centrado manual del título para mantener la caja alineada en consola.
padIzq = floor((anchoInterno - length(tituloResumen)) / 2);
padDer = anchoInterno - length(tituloResumen) - padIzq;
tituloCentrado = [repmat(' ', 1, padIzq), tituloResumen, repmat(' ', 1, padDer)];

fprintf('╔%s╗\n', repmat('═', 1, anchoInterno));
fprintf('║%s║\n', tituloCentrado);
fprintf('╠%s╣\n', repmat('═', 1, anchoInterno));
for k = 1:nConfigs
    fprintf('║%-*s║\n', anchoInterno, lineasResumen{k});
end
fprintf('╚%s╝\n', repmat('═', 1, anchoInterno));
fprintf('\n[OK] Mejor configuración: %s  (R2=%.4f)\n\n', ...
        configs(iBest).nombre, resultados{iBest}.r2);
