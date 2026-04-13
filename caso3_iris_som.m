% =========================================================================
% Script  : caso3_iris_som.m
% Título  : Agrupamiento No Supervisado del Dataset Iris con SOM (selforgmap)
% Descripción: Implementa un Mapa Autoorganizado (Self-Organizing Map, SOM) sobre
%   el dataset 'iris_dataset' para descubrir la estructura natural del espacio de
%   características sin utilizar las etiquetas de clase durante el entrenamiento.
%   Se comparan dos topologías de mapa (2×3 vs 4×3) para analizar su resolución.
% Objetivos:
%   1. Cargar y validar iris_dataset.
%   2. Entrenar un SOM con dos topologías distintas (aprendizaje no supervisado).
%   3. Evaluar la calidad del agrupamiento mediante la métrica de Pureza.
%   4. Visualizar U-Matrix y la proyección de clases reales sobre el mapa.
% Requisitos: MATLAB R2018b+ con Neural Network Toolbox.
% Autor   : [Juan Diego Paredes Gámez.]
% Fecha   : Abril 2026
% =========================================================================

clc; clear; close all;
rng(42); % CAMBIO: Semilla fija para reproducibilidad de resultados

%% -------------------------------------------------------------------------
%  SECCIÓN 1: CARGA Y VALIDACIÓN DE DATOS
% -------------------------------------------------------------------------
fprintf('=== CASO 3: Clustering con iris_dataset (SOM/selforgmap) ===\n\n');

try
    [X, T] = iris_dataset();
    fprintf('[OK] Dataset cargado: %d muestras | %d características | %d clases.\n', ...
            size(X,2), size(X,1), size(T,1));
catch ME
    error('[ERROR] Fallo al cargar iris_dataset: %s', ME.message);
end

if any(isnan(X(:)))
    error('[ERROR] El dataset contiene valores NaN.');
end
fprintf('[OK] Validación de integridad completada (sin NaN).\n\n');

% Índices de clase reales (solo para evaluación posterior, NO usados en entrenamiento)
Tind         = vec2ind(T);
claseNombres = {'Setosa', 'Versicolor', 'Virginica'};
colores      = [0.20 0.55 0.90; 1.00 0.45 0.20; 0.25 0.75 0.40];

%% -------------------------------------------------------------------------
%  SECCIÓN 2: DEFINICIÓN DE CONFIGURACIONES (TOPOLOGÍAS DEL MAPA)
%  Una topología pequeña (2×3) agrupa conservadoramente.
%  Una topología mayor (4×3) provee mayor resolución espacial.
% -------------------------------------------------------------------------
configs(1) = struct('nombre', 'SOM 2×3 (6  nodos)',  'dims', [2 3]);
configs(2) = struct('nombre', 'SOM 4×3 (12 nodos)',  'dims', [4 3]);

nConfigs   = numel(configs);
resultados = cell(nConfigs, 1);

%% -------------------------------------------------------------------------
%  SECCIÓN 3: BUCLE DE ENTRENAMIENTO, EVALUACIÓN Y ALMACENAMIENTO
% -------------------------------------------------------------------------
for k = 1:nConfigs

    fprintf('--- %s ---\n', configs(k).nombre);

    % -- 3.1  Crear SOM hexagonal con la topología indicada
    net = selforgmap(configs(k).dims);

    % -- 3.2  Hiperparámetros de entrenamiento
    net.trainParam.epochs     = 300;
    net.trainParam.showWindow = false;

    % -- 3.3  Entrenamiento NO SUPERVISADO (solo X, sin etiquetas T)
    [net, tr] = train(net, X);

    % -- 3.4  Asignación de nodo ganador (Best Matching Unit) por muestra
    Y    = net(X);
    Yind = vec2ind(Y);   % Índice del BMU para cada muestra

    % -- 3.5  Calcular pureza del clustering respecto a clases reales
    nNodes = prod(configs(k).dims);
    pureza = calcularPureza(Yind, Tind, nNodes);

    % -- 3.6  Cuantization error y topographic error como métricas internas
    qError = mean(min(dist(net.IW{1,1}, X), [], 1));

    fprintf('  Épocas entrenadas  : %d\n',    tr.num_epochs);
    fprintf('  Pureza             : %.4f\n',   pureza);
    fprintf('  Error cuantización : %.6f\n\n', qError);

    resultados{k} = struct('net', net, 'tr', tr, 'Y', Y, 'Yind', Yind, ...
                           'pureza', pureza, 'qError', qError);
end

%% -------------------------------------------------------------------------
%  SECCIÓN 4: VISUALIZACIÓN
% -------------------------------------------------------------------------
for k = 1:nConfigs

    fig = figure('Name', sprintf('Caso 3 – %s', configs(k).nombre), ...
                 'NumberTitle', 'off', 'Position', [50+k*30, 80, 1000, 420]);

    % -- 4.1  U-Matrix: distancias entre pesos de neuronas vecinas
    subplot(1, 3, 1);
    plotsom(resultados{k}.net.IW{1,1}, resultados{k}.net.layers{1}.distances);
    title(sprintf('U-Matrix\n%s', configs(k).nombre), 'FontSize', 9);

    % -- 4.2  Scatter: longitud vs ancho de sépalo coloreado por clase real
    subplot(1, 3, 2);
    hold on; grid on;
    for c = 1:3
        idx = Tind == c;
        scatter(X(1,idx), X(2,idx), 45, colores(c,:), 'filled', ...
                'DisplayName', claseNombres{c});
    end
    legend('Location', 'best', 'FontSize', 8);
    xlabel('Long. Sépalo (cm)'); ylabel('Ancho Sépalo (cm)');
    title('Clases Reales (Dims 1-2)', 'FontSize', 9);
    hold off;

    % -- 4.3  Scatter: coloreado por cluster asignado por el SOM
    subplot(1, 3, 3);
    hold on; grid on;
    nNodes   = prod(configs(k).dims);
    mapColor = lines(nNodes);
    for n = 1:nNodes
        idx = resultados{k}.Yind == n;
        if any(idx)
            scatter(X(1,idx), X(2,idx), 45, mapColor(n,:), 'filled', ...
                    'DisplayName', sprintf('Nodo %d', n));
        end
    end
    legend('Location', 'best', 'FontSize', 7);
    xlabel('Long. Sépalo (cm)'); ylabel('Ancho Sépalo (cm)');
    title(sprintf('Clusters SOM\nPureza: %.4f', resultados{k}.pureza), 'FontSize', 9);
    hold off;
end

% 4.4 Resumen en consola
tituloResumen = 'RESUMEN COMPARATIVO - CASO 3';
lineasResumen = cell(nConfigs, 1);
for k = 1:nConfigs
    lineasResumen{k} = sprintf('  %-24s  Pureza: %7.4f  QErr: %8.5f  ', ...
                               configs(k).nombre, resultados{k}.pureza, resultados{k}.qError);
end

anchoInterno = max([length(tituloResumen), cellfun(@length, lineasResumen)']);

% Centrado manual del titulo para mantener la caja alineada en consola.
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

[~, iBest] = max(cellfun(@(r) r.pureza, resultados));
fprintf('\n[OK] Mejor topologia: %s  (Pureza=%.4f)\n\n', ...
        configs(iBest).nombre, resultados{iBest}.pureza);

%% =========================================================================
%  FUNCIÓN LOCAL: Cálculo de Pureza del Clustering
%  Definición: fracción de muestras asignadas al cluster cuya clase dominante
%  coincide con la clase real de la muestra.
% =========================================================================
function pureza = calcularPureza(Yind, Tind, nNodes)
    totalCorrectos = 0;
    for n = 1:nNodes
        mask = (Yind == n);
        if any(mask)
            clasesEnNodo   = Tind(mask);
            claseMayoritaria = mode(clasesEnNodo);
            totalCorrectos = totalCorrectos + sum(clasesEnNodo == claseMayoritaria);
        end
    end
    pureza = totalCorrectos / numel(Tind);
end
