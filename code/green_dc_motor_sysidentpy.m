%% ========================================================================
%  GREEN MODEL IDENTIFICATION - DC MOTOR
%  Reference: SysIdentPy book case study (page 72, Table 1)
%
%  This script does NOT identify a Standard model.
%  It:
%     1. loads the published SysIdentPy model as the fixed reference;
%     2. identifies only the Green models;
%     3. compares prediction error and arithmetic complexity against the
%        fixed published model.
%
%  This reproduces the data treatment used in the book example:
%     - no signal normalisation;
%     - decimation by 500;
%     - first half for identification and second half for validation;
%     - polynomial NARX with degree 2, ylag = 2 and xlag = 2;
%     - no innovation regressors (LeastSquares with unbiased = false).
%
%  Required toolbox functions:
%     genterms.m
%     orthreg.m
%     get_info.m
%     build_pr.m
%     house.m
%     rowhouse.m
%
%  Data:
%     x_cc.csv -> input x(k); u_cc.csv is accepted as a fallback name
%     y_cc.csv -> output y(k)
%
%  Thalita Nazare
% ========================================================================

clear;
clc;
close all;

format long g


%% ========================================================================
%  1. CONFIGURATION
% =========================================================================

% Green candidate structure.
% The operational comparison is free-run simulation, so lage = 0.
cfg.degree = 2;
cfg.lagy   = 2;
cfg.lagu   = 2;
cfg.lage   = 0;

cfg.minTerms = 2;
cfg.maxTerms = 15;  % complete degree-2 NARX library used in the book

% Figure 6: independent rotations, in degrees (positive = anticlockwise).
cfg.weightTickRotation = 0;
cfg.sysidentpyLabelRotation = 0;  % edit this to tilt only SysIdentPy
cfg.sysidentpyLabelOffset = 0.04;  % distance below axis, in axes-height units

% Data configuration used by the SysIdentPy book example.
cfg.downsample_step = 500;
cfg.nTrain = [];  % empty means an equal 50/50 split after decimation

% Green weights: [error weight, complexity weight]
cfg.errorWeights = (0:2:100)'/100; % 51 weights (49 interior and both endpoints), includes 0.30, 0.50, 0.70
cfg.weightCases = [cfg.errorWeights,1-cfg.errorWeights];
% Keep only these three individual time-series comparisons.
cfg.plotWeights = [0.30 0.50 0.70];

% Arithmetic-complexity proxy.
cfg.word_length_bits = 64;
cfg.mult_exponent = 1.585;

cfg.plotResiduals = true;
cfg.save_results = true;
cfg.results_file = 'green_dc_motor_sysidentpy_page72_results.mat';


%% ========================================================================
%  2. FIGURE CONFIGURATION
% =========================================================================

fig.fontName = 'Times New Roman';
fig.fontSize = 20;
fig.lineWidth = 2.5;
fig.axesWidth = 2;

fig.colorData      = [0.0509803921568627 0.0941176470588235 0.129411764705882];
fig.colorReference = [0.752941176470588 0.341176470588235 0.274509803921569];
fig.colorGreen     = [0.227450980392157 0.431372549019608 0.647058823529412];

set(groot, ...
    'defaultFigureColor','w', ...
    'defaultAxesFontName',fig.fontName, ...
    'defaultAxesFontSize',fig.fontSize, ...
    'defaultAxesLineWidth',fig.axesWidth, ...
    'defaultAxesTickLabelInterpreter','latex', ...
    'defaultTextFontName',fig.fontName, ...
    'defaultTextFontSize',fig.fontSize, ...
    'defaultTextInterpreter','latex', ...
    'defaultLegendFontName',fig.fontName, ...
    'defaultLegendFontSize',fig.fontSize, ...
    'defaultLegendInterpreter','latex', ...
    'defaultLineLineWidth',fig.lineWidth);


%% ========================================================================
%  3. LOAD AND PREPARE DATA
% =========================================================================

Yraw = readmatrix('y_cc.csv');

if isfile('x_cc.csv')
    Uraw = readmatrix('x_cc.csv');
    inputFile = 'x_cc.csv';
elseif isfile('u_cc.csv')
    Uraw = readmatrix('u_cc.csv');
    inputFile = 'u_cc.csv';
else
    error('Input file not found. Expected x_cc.csv or u_cc.csv.');
end

Yraw = Yraw(:);
Uraw = Uraw(:);

lastSample = min(length(Yraw),length(Uraw));
idx = 1:cfg.downsample_step:lastSample;

y = Yraw(idx);
u = Uraw(idx);

N = length(y);

if isempty(cfg.nTrain)
    if mod(N,2) ~= 0
        error(['The number of decimated samples is odd (%d). ' ...
            'The SysIdentPy np.split(...,2) example requires an even number.'],N);
    end
    cfg.nTrain = N/2;
end

if cfg.nTrain >= N
    error('cfg.nTrain must be smaller than the total number of samples.');
end

ut = u(1:cfg.nTrain);
yt = y(1:cfg.nTrain);
validationStart = cfg.nTrain+1;

fprintf('\n============================================================\n');
fprintf('GREEN IDENTIFICATION - DC MOTOR\n');
fprintf('FIXED REFERENCE: SYSIDENTPY BOOK PAGE 72\n');
fprintf('============================================================\n');
fprintf('Samples after downsampling : %d\n',N);
fprintf('Input file                  : %s\n',inputFile);
fprintf('Training samples           : %d\n',cfg.nTrain);
fprintf('Validation samples         : %d\n',N-cfg.nTrain);
fprintf('Polynomial degree          : %d\n',cfg.degree);
fprintf('Output lag                 : %d\n',cfg.lagy);
fprintf('Input lag                  : %d\n',cfg.lagu);
fprintf('Error lag for Green search : %d (NARX, as in the book)\n',cfg.lage);


%% ========================================================================
%  4. INPUT AND OUTPUT
% =========================================================================

sample = (1:N)';

figure(1);
subplot(2,1,1)
plot(sample,u,'Color','k','LineWidth',fig.lineWidth);
hold on
xline(cfg.nTrain,'--','LineWidth',fig.axesWidth);
hold off
xlabel('Sample');
ylabel('Input');
style_axes(gca,fig);

subplot(2,1,2)
plot(sample,y,'Color','k','LineWidth',fig.lineWidth);
hold on
xline(cfg.nTrain,'--','LineWidth',fig.axesWidth);
hold off
xlabel('Sample');
ylabel('Angular velocity');
style_axes(gca,fig);


%% ========================================================================
%  5. FIXED SYSIDENTPY REFERENCE MODEL - PAGE 72, TABLE 1
% =========================================================================
%
% Code convention used by the NARMAX toolbox:
%     1000 + lag -> y(k-lag)
%     2000 + lag -> u(k-lag)
%     3000 + lag -> e(k-lag)
%
% The ten rows below follow the order shown in Table 1 of the book.
% Coefficients are recorded with the precision printed in that table.

reference.model = [
    1001     0       % y(k-1)
    2001  2001       % u(k-1)^2
    1002  1002       % y(k-2)^2
    2001  1001       % u(k-1)y(k-1)
    1002     0       % y(k-2)
    2001  1002       % u(k-1)y(k-2)
    2002     0       % u(k-2)
    2002  1001       % u(k-2)y(k-1)
    2002  2001       % u(k-2)u(k-1)
    1002  1001       % y(k-2)y(k-1)
];

reference.theta = [
     1.0998
     1.0165e+02
    -1.9786e-05
    -1.2138e-01
    -3.2621e-01
     5.3596e-02
     3.4655e+02
    -5.1647e-02
    -8.2162e+00
     4.0961e-05
];

reference.source = ['Wilson R. L. Junior, Nonlinear System Identification ' ...
    'and Forecasting: Theory and Practice With SysIdentPy, page 72, Table 1.'];

reference.nTerms = size(reference.model,1);

[reference.nAdditions,reference.nMultiplications] = ...
    model_operation_count(reference.model);

reference.complexity = model_complexity( ...
    reference.model,cfg.word_length_bits,cfg.mult_exponent);

fprintf('\n============================================================\n');
fprintf('FIXED SYSIDENTPY PAGE 72 REFERENCE MODEL\n');
fprintf('============================================================\n');
display_model(reference.model,reference.theta);
fprintf('Reference model terms       : %d\n',reference.nTerms);


%% ========================================================================
%  6. GREEN CANDIDATE LIBRARY
% =========================================================================

terms = genterms(cfg.degree,cfg.lagy,cfg.lagu,cfg.lage);
nCandidates = size(terms,1);
cfg.maxTerms = min(cfg.maxTerms,nCandidates);

fprintf('Candidate terms            : %d\n',nCandidates);
fprintf('Allowed Green model size   : %d to %d terms\n', ...
    cfg.minTerms,cfg.maxTerms);


%% ========================================================================
%  7. IDENTIFY ONLY THE GREEN MODELS
% =========================================================================

fprintf('\n============================================================\n');
fprintf('GREEN MODEL IDENTIFICATION\n');
fprintf('============================================================\n');

nGreenCases = size(cfg.weightCases,1);
green = struct([]);

for c = 1:nGreenCases
    wE = cfg.weightCases(c,1);
    wC = cfg.weightCases(c,2);

    fprintf('\nw_E = %.2f | w_C = %.2f\n',wE,wC);

    [greenOrder,greenERR] = green_term_order( ...
        terms,ut,yt,cfg.maxTerms,wE,wC, ...
        cfg.word_length_bits,cfg.mult_exponent);

    [kGreen,selectionDetails] = choose_green_size( ...
        terms,greenOrder,greenERR,cfg.minTerms,cfg.maxTerms, ...
        wE,wC,cfg.word_length_bits,cfg.mult_exponent);

    selectedTerms = terms(greenOrder(1:kGreen),:);

    [mGreen,xGreen,eGreen] = orthreg( ...
        selectedTerms,ut,yt,[kGreen 0],0);

    green(c).w_E = wE;
    green(c).w_C = wC;
    green(c).model = mGreen;
    green(c).theta = xGreen(:,1);
    green(c).trainingResiduals = eGreen;
    green(c).nTerms = kGreen;
    green(c).cumulativeERR = sum(greenERR(1:kGreen));
    green(c).selection = selectionDetails;

    [green(c).nAdditions,green(c).nMultiplications] = ...
        model_operation_count(green(c).model);

    green(c).complexity = model_complexity( ...
        green(c).model,cfg.word_length_bits,cfg.mult_exponent);
end


%% ========================================================================
%  8. FREE-RUN VALIDATION
% =========================================================================

[reference.yHat,yValidation] = simulate_continuation_free_run( ...
    reference.model,reference.theta,u,y,validationStart);

reference.metrics = model_metrics(yValidation,reference.yHat);

for c = 1:nGreenCases
    [green(c).yHat,yTrue] = simulate_continuation_free_run( ...
        green(c).model,green(c).theta,u,y,validationStart);

    green(c).metrics = model_metrics(yTrue,green(c).yHat);

    green(c).complexityReduction = 100* ...
        (reference.complexity-green(c).complexity)/reference.complexity;

    green(c).nrmseChange = 100* ...
        (green(c).metrics.NRMSE-reference.metrics.NRMSE)/ ...
        reference.metrics.NRMSE;

    green(c).rmseChange = 100* ...
        (green(c).metrics.RMSE-reference.metrics.RMSE)/ ...
        reference.metrics.RMSE;
end


%% ========================================================================
%  9. RESULTS TABLE
% =========================================================================

nModels = 1+nGreenCases;

Model = strings(nModels,1);
Terms = zeros(nModels,1);
NRMSE = zeros(nModels,1);
RMSE = zeros(nModels,1);
R2 = zeros(nModels,1);
Additions = zeros(nModels,1);
Multiplications = zeros(nModels,1);
Complexity = zeros(nModels,1);
ComplexityReduction = zeros(nModels,1);
NRMSEChange = zeros(nModels,1);

Model(1) = "SysIdentPy reference";
Terms(1) = reference.nTerms;
NRMSE(1) = reference.metrics.NRMSE;
RMSE(1) = reference.metrics.RMSE;
R2(1) = reference.metrics.R2;
Additions(1) = reference.nAdditions;
Multiplications(1) = reference.nMultiplications;
Complexity(1) = reference.complexity;

for c = 1:nGreenCases
    row = c+1;
    Model(row) = string(sprintf('Green: w_E=%.2f, w_C=%.2f', ...
        green(c).w_E,green(c).w_C));
    Terms(row) = green(c).nTerms;
    NRMSE(row) = green(c).metrics.NRMSE;
    RMSE(row) = green(c).metrics.RMSE;
    R2(row) = green(c).metrics.R2;
    Additions(row) = green(c).nAdditions;
    Multiplications(row) = green(c).nMultiplications;
    Complexity(row) = green(c).complexity;
    ComplexityReduction(row) = green(c).complexityReduction;
    NRMSEChange(row) = green(c).nrmseChange;
end

resultsTable = table( ...
    Model,Terms,NRMSE,RMSE,R2,Additions,Multiplications,Complexity, ...
    ComplexityReduction,NRMSEChange);

fprintf('\n============================================================\n');
fprintf('FINAL COMPARISON AGAINST SYSIDENTPY PAGE 72 MODEL\n');
fprintf('============================================================\n');
fprintf(['ComplexityReduction > 0 means that the Green model is less ' ...
    'complex.\n']);
fprintf(['NRMSEChange > 0 means that the Green model has a larger ' ...
    'validation error.\n\n']);
disp(resultsTable)


%% ========================================================================
%  10. DISPLAY GREEN MODELS
% =========================================================================

for c = 1:nGreenCases
    fprintf('\n============================================================\n');
    fprintf('GREEN MODEL - w_E = %.2f | w_C = %.2f\n', ...
        green(c).w_E,green(c).w_C);
    fprintf('============================================================\n');
    display_model(green(c).model,green(c).theta);
end


%% ========================================================================
%  11. VALIDATION FIGURES
% =========================================================================

sampleValidation = validationStart:N;

plotCases = zeros(size(cfg.plotWeights));
for j = 1:numel(cfg.plotWeights)
    [distance,plotCases(j)] = min(abs(cfg.weightCases(:,1)-cfg.plotWeights(j)));
    assert(distance < 1e-10,'Requested plot weight is missing from sweep.');
end
for j = 1:numel(plotCases)
    c = plotCases(j);
    plot_validation_case( ...
        j+1,sampleValidation,yValidation,reference.yHat,green(c).yHat, ...
        green(c).w_E,green(c).w_C,fig);
end


%% ========================================================================
%  12. OPTIONAL RESIDUAL ANALYSIS
% =========================================================================

if cfg.plotResiduals
    figure(5);

    subplot(numel(plotCases)+1,1,1)
    plot(sampleValidation,yValidation-reference.yHat, ...
        'Color',fig.colorReference);
    ylabel('$e(k)$');
    legend('SysIdentPy reference','Location','best');
    style_axes(gca,fig);

    for j = 1:numel(plotCases)
        c = plotCases(j);
        subplot(numel(plotCases)+1,1,j+1)
        plot(sampleValidation,yValidation-green(c).yHat, ...
            'Color',fig.colorGreen);
        ylabel('$e(k)$');
        legend(sprintf('$w_E=%.2f,\;w_C=%.2f$', ...
            green(c).w_E,green(c).w_C),'Location','best');
        style_axes(gca,fig);
    end

    xlabel('Sample');
end


% Figure 6: weight vs complexity, coloured by validation NRMSE.
% Validation is diagnostic: do not tune weights on these results and then
% describe the same validation segment as an independent final test.
tradeoff = plot_tradeoff(green,reference,cfg,fig);

%% ========================================================================
%  13. SAVE RESULTS
% =========================================================================

results.cfg = cfg;
results.data.u = u;
results.data.y = y;
results.data.nTotal = N;
results.data.nTraining = cfg.nTrain;
results.data.validationStart = validationStart;
results.reference = reference;
results.green = green;
results.validation.y = yValidation;
results.validation.referencePrediction = reference.yHat;

for c = 1:nGreenCases
    results.validation.greenPrediction{c} = green(c).yHat;
end

results.summary = resultsTable;
results.tradeoff = tradeoff;

if cfg.save_results
    save(cfg.results_file,'results');
    fprintf('\nResults saved to: %s\n',cfg.results_file);
end

fprintf('\n============================================================\n');
fprintf('DC MOTOR GREEN IDENTIFICATION FINISHED\n');
fprintf('============================================================\n');


%% ========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function [order,errSelected] = green_term_order( ...
    model,u,y,maxTerms,wE,wC,bits,multExponent)
% Forward Orthogonal Regression with the Green ERR-complexity criterion.

    [npr,~,lag] = get_info(model);
    maxTerms = min([maxTerms,npr,size(model,1)]);

    P = build_pr(model,u,y);
    A = P(lag+1:end,:);
    Y = y(lag+1:end);

    [m,n] = size(A);
    sqy = Y'*Y;

    if sqy <= eps
        error('Output energy is zero.');
    end

    piv = 1:n;
    errSelected = zeros(maxTerms,1);

    for r = 1:maxTerms
        available = r:n;
        nAvailable = numel(available);
        errCandidate = zeros(nAvailable,1);
        compCandidate = zeros(nAvailable,1);

        for q = 1:nAvailable
            j = available(q);
            aj = A(r:m,j)'*A(r:m,j);
            bj = A(r:m,j)'*Y(r:m);

            if aj <= eps
                errCandidate(q) = 0;
            else
                errCandidate(q) = (bj^2)/(aj*sqy);
            end

            trialPiv = [piv(1:r-1),piv(j)];
            trialModel = model(trialPiv,:);
            compCandidate(q) = model_complexity( ...
                trialModel,bits,multExponent);
        end

        errNorm = normalise01(errCandidate);
        compNorm = normalise01(compCandidate);
        J = wE*(1-errNorm)+wC*compNorm;

        Jmin = min(J);
        tie = find(abs(J-Jmin) <= 1e-12);

        if numel(tie) > 1
            % Revised tie rule: equivalent ERR, lower cost, original index.
            eT = errCandidate(tie);
            tie = tie(eT >= max(eT)-1e-10*max(max(eT),eps));
            candidateIndex = piv(available(tie));
            ranking = [compCandidate(tie), candidateIndex(:)];
            [~,bestTie] = sortrows(ranking,[1 2]);
            localBest = tie(bestTie(1));
        else
            localBest = tie(1);
        end

        k = available(localBest);

        if k ~= r
            tmp = A(:,r);
            A(:,r) = A(:,k);
            A(:,k) = tmp;

            tmp = piv(r);
            piv(r) = piv(k);
            piv(k) = tmp;
        end

        errSelected(r) = errCandidate(localBest);

        v = house(A(r:m,r));
        A(r:m,r:n) = rowhouse(A(r:m,r:n),v);
        Y(r:m) = rowhouse(Y(r:m),v);
    end

    order = piv(1:maxTerms);
end


function [kBest,details] = choose_green_size( ...
    terms,order,errOrder,minTerms,maxTerms,wE,wC,bits,multExponent)
% Select the final number of terms using error loss and complexity.

    kValues = (minTerms:maxTerms)';
    nK = numel(kValues);
    cumulativeERR = zeros(nK,1);
    errLoss = zeros(nK,1);
    complexity = zeros(nK,1);

    for q = 1:nK
        k = kValues(q);
        cumulativeERR(q) = sum(errOrder(1:k));
        errLoss(q) = max(1-cumulativeERR(q),0);
        complexity(q) = model_complexity( ...
            terms(order(1:k),:),bits,multExponent);
    end

    errNorm = normalise01(errLoss);
    compNorm = normalise01(complexity);
    J = wE*errNorm+wC*compNorm;

    [~,idx] = min(J);
    kBest = kValues(idx);

    details.k = kValues;
    details.cumulativeERR = cumulativeERR;
    details.ERRloss = errLoss;
    details.complexity = complexity;
    details.modelCost = J;
    details.selectedIndex = idx;
end


function C = model_complexity(model,bits,multExponent)
% Computational-complexity proxy per model evaluation.

    [nAdd,nMult] = model_operation_count(model);
    C = nAdd*bits+nMult*(bits^multExponent);
end


function [nAdd,nMult] = model_operation_count(model)
% Arithmetic operations per evaluation, including coefficient products.

    nTerms = size(model,1);
    nAdd = max(nTerms-1,0);
    nMult = 0;

    for i = 1:nTerms
        nMult = nMult+nnz(model(i,:) > 0);
    end
end


function z = normalise01(x)
% Min-max normalisation.

    x = x(:);
    xmin = min(x);
    xmax = max(x);

    if abs(xmax-xmin) <= eps(max(abs(x))+1)
        z = zeros(size(x));
    else
        z = (x-xmin)/(xmax-xmin);
    end
end


function [yhat,ytrue] = simulate_continuation_free_run( ...
    model,theta,u,y,startIndex)
% Free-run simulation with measured outputs only for initial conditions.
% Innovation regressors, if present, are zero in free-run simulation.

    u = u(:);
    y = y(:);

    [~,~,lag] = get_info(model);
    firstIdx = startIndex-lag;
    uSeg = u(firstIdx:end);
    y0 = y(firstIdx:startIndex-1);

    n = length(uSeg);
    degree = size(model,2);
    ySim = zeros(n,1);

    if lag > 0
        ySim(1:lag) = y0;
    end

    for nn = lag+1:n
        total = 0;

        for i = 1:size(model,1)
            value = theta(i);

            for j = 1:degree
                code = model(i,j);

                if code == 0 || code == -1
                    break
                end

                signalType = floor(code/1000);
                lagValue = mod(code,1000);

                switch signalType
                    case 1
                        value = value*ySim(nn-lagValue);
                    case 2
                        value = value*uSeg(nn-lagValue);
                    case 3
                        value = 0;
                    otherwise
                        error('Unknown model code: %d.',code);
                end
            end

            total = total+value;
        end

        ySim(nn) = total;
    end

    yhat = ySim(lag+1:end);
    ytrue = y(startIndex:end);
end


function metrics = model_metrics(y,yhat)
% RMSE, NRMSE and coefficient of determination.

    y = y(:);
    yhat = yhat(:);
    residual = y-yhat;

    metrics.RMSE = sqrt(mean(residual.^2));

    denom = norm(y-mean(y));
    if denom <= eps
        metrics.NRMSE = NaN;
    else
        metrics.NRMSE = norm(residual)/denom;
    end

    SST = sum((y-mean(y)).^2);
    SSE = sum(residual.^2);

    if SST <= eps
        metrics.R2 = NaN;
    else
        metrics.R2 = 1-SSE/SST;
    end
end


function display_model(model,theta)
% Display a polynomial model in readable notation.

    termText = cell(size(model,1),1);

    for i = 1:size(model,1)
        termText{i} = term_to_string(model(i,:));
    end

    T = table((1:size(model,1))',theta(:),termText, ...
        'VariableNames',{'Term','Coefficient','Regressor'});
    disp(T)
end


function txt = term_to_string(termCode)
% Convert toolbox term coding into readable notation.

    factors = {};

    for j = 1:length(termCode)
        code = termCode(j);

        if code == 0 || code == -1
            continue
        end

        signalType = floor(code/1000);
        lagValue = mod(code,1000);

        switch signalType
            case 1
                factors{end+1} = sprintf('y(k-%d)',lagValue); %#ok<AGROW>
            case 2
                factors{end+1} = sprintf('u(k-%d)',lagValue); %#ok<AGROW>
            case 3
                factors{end+1} = sprintf('e(k-%d)',lagValue); %#ok<AGROW>
        end
    end

    if isempty(factors)
        txt = '1';
    else
        txt = strjoin(factors,' * ');
    end
end


function style_axes(ax,fig)
% Apply the same graphical configuration to every axis.

    set(ax, ...
        'FontName',fig.fontName, ...
        'FontSize',fig.fontSize, ...
        'LineWidth',fig.axesWidth, ...
        'TickLabelInterpreter','latex');
    grid(ax,'on');
    box(ax,'on');
end


function plot_validation_case( ...
    figureNumber,x,yData,yReference,yGreen,wE,wC,fig)
% Validation plot for one Green weight case.

    figure(figureNumber);

    plot(x,yData,'-','Color',fig.colorData, ...
        'LineWidth',3.5);
    hold on
    plot(x,yReference,'-o','Color',fig.colorReference, ...
        'LineWidth',2,'MarkerIndices',1:5:numel(x),'MarkerSize',8);
    plot(x,yGreen,'-s','Color',fig.colorGreen, ...
        'LineWidth',2.5,'MarkerIndices',1:5:numel(x),'MarkerSize',8);
    hold off

    xlabel('Sample');
    ylabel('Angular velocity');
    legend('Validation data','SysIdentPy reference', ...
        sprintf('$w_E=%.2f,\;w_C=%.2f$',wE,wC),'Location','best');
    style_axes(gca,fig);
end


function out = plot_tradeoff(green,reference,cfg,fig)
% NRMSE here is ||y-yhat||/||y-mean(y)|| (also called RRSE), lower is better.
% Complexity is an arithmetic proxy, not measured energy consumption.
    w = cfg.weightCases(:,1);
    C = [green.complexity]';
    E = arrayfun(@(g) g.metrics.NRMSE,green)';
    valid = isfinite(C) & isfinite(E);
    fprintf('\nTrade-off: %d/%d finite Green validation results.\n',sum(valid),numel(w));
    if any(~valid)
        warning('Non-finite validation results: weights %s. See results table.', ...
            mat2str(w(~valid)'));
    end

    % Non-dominated set among evaluated Green cases AND fixed reference.
    allC = [C;reference.complexity];
    allE = [E;reference.metrics.NRMSE];
    pareto = isfinite(allC) & isfinite(allE);
    for i = 1:numel(allC)
        if pareto(i)
            dominates = allC <= allC(i) & allE <= allE(i) & ...
                (allC < allC(i) | allE < allE(i));
            pareto(i) = ~any(dominates);
        end
    end

    % Shared colour limits include the fixed reference NRMSE.
    colourValues = [E(valid);reference.metrics.NRMSE];
    colourValues = colourValues(isfinite(colourValues));
    limits = [0 1];
    if ~isempty(colourValues)
        limits = [min(colourValues),max(colourValues)];
        if limits(1) == limits(2)
            padding = max(1e-6,abs(limits(1))*0.01);
            limits = [max(0,limits(1)-padding),limits(2)+padding];
        end
    end
    cmap = parula(256);
    figure(6); clf;
    ax = gca;
    colormap(ax,cmap); caxis(ax,limits);
    hold(ax,'on');
    % The reference has no Green weight; place it in a separate labelled slot.
    refX = 1.12;
    if isfinite(reference.complexity)
        refColour = [0.5 0.5 0.5];
        refLabel = 'SysIdentPy (NRMSE unavailable)';
        if isfinite(reference.metrics.NRMSE)
            fraction = (reference.metrics.NRMSE-limits(1))/diff(limits);
            colourIndex = max(1,min(size(cmap,1), ...
                1+floor(fraction*size(cmap,1))));
            refColour = cmap(colourIndex,:);
            refLabel = sprintf('SysIdentPy (NRMSE = %.4g)',reference.metrics.NRMSE);
        end
        plot(ax,refX,reference.complexity,'p','LineStyle','none', ...
            'MarkerSize',18,'MarkerFaceColor',refColour, ...
            'MarkerEdgeColor','k','LineWidth',1.5,'DisplayName',refLabel);
    end
    scatter(ax,w(valid),C(valid),80,E(valid),'filled', ...
        'MarkerEdgeColor','k','DisplayName','Green models');
    if any(~valid & isfinite(C))
        unavailable = ~valid & isfinite(C);
        plot(ax,w(unavailable),C(unavailable),'x','Color',[0.5 0.5 0.5], ...
            'LineStyle','none','MarkerSize',8, ...
            'DisplayName','Green (NRMSE unavailable)');
    end
    hold(ax,'off');
    xlim(ax,[0 1.22]);
    xticks(ax,[0:0.2:1 refX]);
    xticklabels(ax,{'0','0.2','0.4','0.6','0.8','1',''});
    cb = colorbar(ax);
    cb.Label.String = 'Validation NRMSE (lower is better)';
    xlabel('Error weight $w_E$ ($w_C = 1-w_E$)');
    ylabel('Arithmetic complexity proxy');
    title('Model complexity and prediction error across Green weights');
    legend(ax,'show','Location','best');
    style_axes(gca,fig);

    % MATLAB rotates all native tick labels together. Draw the reference
    % label separately so its rotation can be adjusted independently.
    ax.XTickLabelRotation = cfg.weightTickRotation;
    refXNormalized = (refX-ax.XLim(1))/diff(ax.XLim);
    text(ax,refXNormalized+0.06,-cfg.sysidentpyLabelOffset,'SIPy', ...
        'Units','normalized','Rotation',cfg.sysidentpyLabelRotation, ...
        'HorizontalAlignment','right','VerticalAlignment','middle', ...
        'FontName',ax.FontName,'FontSize',ax.FontSize, ...
        'Interpreter','none','Clipping','off','HandleVisibility','off');

    % Same reference-relative scale for both rows: 1 = reference,
    % <1 = better. Retained in saved results only; no heatmap figure.
    relative = nan(2,numel(w));
    if isfinite(reference.metrics.NRMSE) && reference.metrics.NRMSE > 0
        relative(1,:) = E'/reference.metrics.NRMSE;
    end
    if isfinite(reference.complexity) && reference.complexity > 0
        relative(2,:) = C'/reference.complexity;
    end
    relative(~isfinite(relative)) = NaN;
    out.weights = w;
    out.complexity = C;
    out.NRMSE = E;
    out.finite = valid;
    out.colourLimits = limits;
    out.relativeToReference = relative;
    out.paretoGreen = pareto(1:end-1);
    out.paretoReference = pareto(end);
end


%% NEW FIGURE — DATA WITH SHADED REGIONS

colorEst = [0.87 0.94 0.87];   % Soft green
colorVal = [1.00 0.94 0.84];   % Light orange

sample = (1:N)';
splitPoint = cfg.nTrain + 0.5;

figure('Color','w','Position',[100 100 1400 650]);
tl = tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

signals = {u, y};
axisLabels = {'Input', 'Angular velocity'};
axData = gobjects(2,1);

for j = 1:2
    axData(j) = nexttile(tl);
    ax = axData(j);
    hold(ax,'on');

    signal = signals{j}(:);

    % Vertical limits with a small margin
    low = min(signal);
    high = max(signal);
    margin = 0.08*(high-low);
    if margin == 0
        margin = 0.08*max(1,abs(high));
    end
    limits = [low-margin, high+margin];

    % Background regions
    hEst = patch(ax, ...
        [0.5 splitPoint splitPoint 0.5], ...
        [limits(1) limits(1) limits(2) limits(2)], ...
        colorEst,'EdgeColor','none');

    hVal = patch(ax, ...
        [splitPoint N+0.5 N+0.5 splitPoint], ...
        [limits(1) limits(1) limits(2) limits(2)], ...
        colorVal,'EdgeColor','none');

    % Measured data in black
    plot(ax,sample,signal,'k','LineWidth',1.2, ...
        'HandleVisibility','off');

    xline(ax,splitPoint,'--','Color',[0.5 0.5 0.5], ...
        'LineWidth',1,'HandleVisibility','off');

    xlim(ax,[0.5 N+0.5]);
    ylim(ax,limits);
    ylabel(ax,axisLabels{j});

    set(ax, ...
        'FontName',fig.fontName, ...
        'FontSize',fig.fontSize, ...
        'LineWidth',1, ...
        'Layer','top', ...
        'Box','off', ...
        'TickDir','out', ...
        'XGrid','on', ...
        'YGrid','on', ...
        'GridAlpha',0.12, ...
        'GridColor',[0.5 0.5 0.5]);

    legend(ax,[hEst hVal],{'Estimation','Validation'}, ...
        'Location','northeast','Orientation','horizontal', ...
        'FontSize',14,'Box','on');

    hold(ax,'off');
end

linkaxes(axData,'x');
xlabel(axData(2),'Sample');