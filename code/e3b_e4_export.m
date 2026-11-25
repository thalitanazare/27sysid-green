%% ========================================================================
%  FASE 2 - E3b (exportar modelos para medir tempo) e E4 (tempo de identificacao)
%  Roda os tres casos.
%
%  E3b: para cada caso exporta em JSON o modelo SysIdentPy de referencia,
%       Green com w_E = 0.3, 0.5, 0.7 e os extremos w_E = 0 e 1 (termos,
%       coeficientes estimados por orthreg nos dados de identificacao),
%       a entrada de validacao, a condicao inicial e o free run do MATLAB
%       (para conferir o codigo C).
%  E4:  tempo da ordenacao + selecao final na MESMA implementacao, com
%       (i) so ERR (FROLS) e (ii) criterio Green (w_E = 0.5). A diferenca
%       isola o custo extra do Green. Mediana de nRep repeticoes.
%       Uma thread (maxNumCompThreads(1)) para nao misturar com o BLAS.
%
%  Saidas em fase2_outputs/e3b/<caso>_models.json e fase2_outputs/e4/e4_ident_time.csv
% ========================================================================

clear; clc; close all; format long g
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
addpath(fullfile(scriptDir,'fase2_lib'));
maxNumCompThreads(1);

outE3 = fullfile(scriptDir,'fase2_outputs','e3b');  if ~exist(outE3,'dir'), mkdir(outE3); end
outE4 = fullfile(scriptDir,'fase2_outputs','e4');   if ~exist(outE4,'dir'), mkdir(outE4); end

cases = {'dc','ced','silverbox'};
e4rows = {};
for ic = 1:numel(cases)
    caseName = cases{ic};
    cfg = case_cfg(caseName);
    [ut,yt,uv,yv,startIdx] = load_case(caseName,scriptDir);
    terms = genterms(cfg.degree,cfg.lagy,cfg.lagu,cfg.lage);
    dTerm = sum(terms > 0,2);
    [~,~,lag] = get_info(terms);
    tBuild = tic;
    P = build_pr(terms,ut,yt);  A0 = P(lag+1:end,:);  clear P
    Y0 = yt(lag+1:end);
    tBuild = toc(tBuild);
    fprintf('\n== %s | %d candidatos | %d amostras | matriz em %.2f s\n', ...
        caseName,size(terms,1),numel(yt),tBuild);

    %% E4 - tempo de identificacao (mesma implementacao)
    nRep = 5;  if strcmp(caseName,'silverbox'), nRep = 3; end
    tF = zeros(nRep,1);  tG = zeros(nRep,1);
    for r = 1:nRep
        t0 = tic; mF = run_green(A0,Y0,terms,dTerm,cfg,1,0,64,64^0.585,'none','C',[]); tF(r) = toc(t0);
        t0 = tic; mG = run_green(A0,Y0,terms,dTerm,cfg,0.5,0.5,64,64^0.585,'C','C',[]); tG(r) = toc(t0);
    end
    e4rows(end+1,:) = {caseName,size(terms,1),numel(yt),cfg.maxTerms,tBuild, ...
        median(tF),median(tG),median(tG)/median(tF)}; %#ok<AGROW>
    fprintf('E4 %s: FROLS %.3f s | Green %.3f s | razao %.3f\n', ...
        caseName,median(tF),median(tG),median(tG)/median(tF));

    %% E3b - modelos para timing
    models = struct('label',{},'codes',{},'theta',{},'nAdd',{},'nMult',{},'C',{},'yhat',{});
    % Referencia SysIdentPy
    [refCodes,refTheta] = reference_model(caseName,scriptDir);
    models(end+1) = pack('SysIdentPy',refCodes,refTheta,uv,yv,startIdx);
    % Green e extremos
    for wE = [0.3 0.5 0.7 0 1]
        mdl = run_green(A0,Y0,terms,dTerm,cfg,wE,1-wE,64,64^0.585,'C','C',[]);
        k = size(mdl.sel,1);
        ws = warning('off','all');
        [mG,xG] = orthreg(mdl.sel,ut,yt,[k 0],0);
        warning(ws);
        models(end+1) = pack(sprintf('Green (%.2f)',wE),mG,xG(:,1),uv,yv,startIdx); %#ok<SAGROW>
    end
    for m = 1:numel(models)
        yref = yv(startIdx:end);  yref = yref(:);  yh = models(m).yhat(:);
        nr = norm(yref-yh)/norm(yref-mean(yref));
        fprintf('  %-16s p=%2d A/M=%3d/%3d C=%10.2f NRMSE=%.3f\n',models(m).label, ...
            size(models(m).codes,1),models(m).nAdd,models(m).nMult,models(m).C,nr);
    end
    S.case = caseName;  S.u = uv(:)';  S.y = yv(:)';  S.start0 = startIdx-1;  % 0-based
    S.models = models;
    fid = fopen(fullfile(outE3,[caseName '_models.json']),'w');
    fprintf(fid,'%s',jsonencode(S));  fclose(fid);
    clear A0
end
T4 = cell2table(e4rows,'VariableNames',{'case','nCandidates','nTrain','maxTerms', ...
    'tBuildMatrix_s','tFROLS_s','tGreen_s','ratio'});
writetable(T4,fullfile(outE4,'e4_ident_time.csv'));
disp(T4)


%% ========================================================================
function cfg = case_cfg(caseName)
    cfg.lage = 0; cfg.minTerms = 2; cfg.bits = 64;
    switch caseName
        case 'dc',        cfg.degree = 2; cfg.lagy = 2;  cfg.lagu = 2;  cfg.maxTerms = 15;
        case 'ced',       cfg.degree = 3; cfg.lagy = 10; cfg.lagu = 10; cfg.maxTerms = 40;
        case 'silverbox', cfg.degree = 3; cfg.lagy = 10; cfg.lagu = 10; cfg.maxTerms = 20;
    end
end


function [codes,theta] = reference_model(caseName,scriptDir)
    switch caseName
        case 'dc'   % SysIdentPy book, p. 72, Table 1 (igual ao script do artigo)
            codes = [1001 0; 2001 2001; 1002 1002; 2001 1001; 1002 0; ...
                     2001 1002; 2002 0; 2002 1001; 2002 2001; 1002 1001];
            theta = [1.0998; 1.0165e+02; -1.9786e-05; -1.2138e-01; -3.2621e-01; ...
                     5.3596e-02; 3.4655e+02; -5.1647e-02; -8.2162e+00; 4.0961e-05];
        case 'ced'
            M = load(fullfile(scriptDir,'CED_outputs','CED_exp2_degree3_lag10_model.mat'));
            codes = double(M.model_code); theta = double(M.theta(:));
        case 'silverbox'
            M = load(fullfile(scriptDir,'Silverbox_arrow_validation_outputs', ...
                'Silverbox_arrow_full_degree3_lag14_model.mat'));
            codes = double(M.model_code); theta = double(M.theta(:));
    end
end


function m = pack(label,codes,theta,uv,yv,startIdx)
    codes = double(codes);
    nT = size(codes,1);
    m.label = label;
    m.codes = codes;
    m.theta = theta(:);
    m.nAdd = max(nT-1,0);
    m.nMult = nnz(codes > 0);
    m.C = m.nAdd*64+m.nMult*64^1.585;
    [yh,~] = simulate_free_run(codes,theta,uv,yv,startIdx);
    m.yhat = yh(:)';
end
