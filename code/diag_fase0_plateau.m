%% ========================================================================
%  FASE 0 - DIAGNOSTICO DA SELECAO FINAL
%  Versao generica: caso escolhido pela variavel de ambiente FASE0_CASE
%  ('dc', 'ced' ou 'silverbox'). Mesmo algoritmo dos scripts do artigo:
%     dc        -> green_dc_motor_sysidentpy.m
%     ced       -> green_ced_exp2_sysidentpy_reference.m
%     silverbox -> green_silverbox_arrow_sysidentpy_reference.m
%
%  Validacao usada APENAS como diagnostico (NRMSE free-run de cada prefixo).
%  Saidas em diag_fase0_outputs/<caso>/
% ========================================================================

clear; clc; close all; format long g

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
caseName = lower(getenv('FASE0_CASE'));
if isempty(caseName), caseName = 'silverbox'; end

%% 1. CONFIGURACAO (igual aos scripts do artigo)
cfg.case  = caseName;
cfg.lage  = 0;
cfg.minTerms = 2;
cfg.bits  = 64;
cfg.alpha = 1.585;
switch caseName
    case 'dc'
        cfg.degree = 2; cfg.lagy = 2;  cfg.lagu = 2;  cfg.maxTerms = 15;
        cfg.paper = [0.30 4 0.515; 0.50 6 0.299; 0.70 7 0.162];
    case 'ced'
        cfg.degree = 3; cfg.lagy = 10; cfg.lagu = 10; cfg.maxTerms = 40;
        cfg.paper = [0.30 4 1.420; 0.50 6 0.236; 0.70 10 0.189];
    case 'silverbox'
        cfg.degree = 3; cfg.lagy = 10; cfg.lagu = 10; cfg.maxTerms = 20;
        cfg.paper = [0.30 5 0.285; 0.50 5 0.285; 0.70 5 0.285];
    otherwise
        error('FASE0_CASE desconhecido: %s',caseName);
end

cfg.sweep      = round((0:0.02:1)',2);          % 51 pesos, inclui extremos
cfg.logWeights = [0.30 0.50 0.70 0.90 0.98 1.00];
cfg.fineGrid   = (0:0.001:1)';                  % analise so do estagio final
if strcmp(getenv('FASE0_QUICK'),'1')            % teste rapido: so 6 pesos
    cfg.sweep = cfg.logWeights(:);
end

cfg.outDir = fullfile(scriptDir,'diag_fase0_outputs',caseName);
if ~exist(cfg.outDir,'dir'), mkdir(cfg.outDir); end

%% 2. DADOS (mesma preparacao dos scripts do artigo)
switch caseName
    case 'dc'
        Yraw = readmatrix(fullfile(scriptDir,'y_cc.csv'));
        Uraw = readmatrix(fullfile(scriptDir,'u_cc.csv'));
        Yraw = Yraw(:); Uraw = Uraw(:);
        idxD = 1:500:min(numel(Yraw),numel(Uraw));
        y = Yraw(idxD); u = Uraw(idxD);
        nTrain = numel(y)/2;
        assert(nTrain == fix(nTrain),'Numero impar de amostras apos decimacao.');
        ut = u(1:nTrain); yt = y(1:nTrain);
        uv = u; yv = y;                         % continuacao a partir do treino
        startIdx = nTrain+1;
    case 'ced'
        D = load(fullfile(scriptDir,'CED_outputs','CED_experiment_2_data.mat'));
        ut = double(D.u_train(:)); yt = double(D.y_train(:));
        uv = double(D.u_test(:));  yv = double(D.y_test(:));
        startIdx = double(D.n_initial)+1;
    case 'silverbox'
        D = load(fullfile(scriptDir,'Silverbox_arrow_validation_outputs', ...
            'Silverbox_arrow_full_data.mat'));
        ut = double(D.u_train(:)); yt = double(D.y_train(:));
        uv = double(D.u_test(:));  yv = double(D.y_test(:));
        startIdx = double(D.n_initial)+1;
end
fprintf('Caso %s | treino %d | validacao avaliada %d\n', ...
    caseName,numel(yt),numel(yv)-startIdx+1);

%% 3. CANDIDATOS E MATRIZ DE REGRESSORES (montada uma vez)
terms = genterms(cfg.degree,cfg.lagy,cfg.lagu,cfg.lage);
nCand = size(terms,1);
dTerm = sum(terms > 0,2);                       % multiplicacoes por termo
[~,~,lag] = get_info(terms);
P  = build_pr(terms,ut,yt);
A0 = P(lag+1:end,:);  clear P
Y0 = yt(lag+1:end);
fprintf('Candidatos %d | matriz %d x %d\n',nCand,size(A0,1),size(A0,2));

%% 4. VARREDURA DE PESOS
nW = numel(cfg.sweep);
ckpt = fullfile(cfg.outDir,sprintf('fase0_checkpoint_%dw.mat',nW));
if strcmp(getenv('FASE0_RESUME'),'1') && exist(ckpt,'file')
    load(ckpt,'res');  fprintf('Varredura carregada de %s\n',ckpt);
    startC = nW+1;
else
    res = struct([]);  startC = 1;
end
tAll = tic;
for c = startC:nW
    wE = cfg.sweep(c);  wC = 1-wE;
    tc = tic;
    [order,errSel,stepLog] = green_term_order_log( ...
        A0,Y0,terms,dTerm,cfg.maxTerms,wE,wC,cfg.bits,cfg.alpha);
    [kG,fin] = choose_green_size_log( ...
        terms,order,errSel,cfg.minTerms,cfg.maxTerms,wE,wC,cfg.bits,cfg.alpha);
    res(c).wE = wE;
    res(c).order = order;
    res(c).errSel = errSel;
    res(c).steps = stepLog;
    res(c).final = fin;
    res(c).k = kG;
    res(c).C = fin.complexity(fin.selectedIndex);
    fprintf('w_E = %.2f -> p = %2d | C = %9.2f | %.1f s\n', ...
        wE,kG,res(c).C,toc(tc));
end
fprintf('Varredura completa em %.1f min\n',toc(tAll)/60);
if startC == 1, save(ckpt,'res','-v7.3'); end

%% 5. DIAGNOSTICO A - A ORDEM DOS REGRESSORES MUDA COM O PESO?
orders = vertcat(res.order);                    % nW x maxTerms
[~,~,orderGroup] = unique(orders,'rows','stable');
fprintf('\nOrdens distintas na varredura: %d\n',max(orderGroup));
for g = 1:max(orderGroup)
    idx = find(orderGroup == g);
    fprintf('  grupo %d: w_E = %s\n',g,mat2str(cfg.sweep(idx)',3));
end
first5 = orders(:,1:5);
[~,~,g5] = unique(sort(first5,2),'rows','stable');
fprintf('Conjuntos distintos nos 5 primeiros termos: %d\n',max(g5));

%% 6. DIAGNOSTICO B - SO O ESTAGIO FINAL (ordem fixa, w_E fino)
% Isola o estagio final: para cada ordem distinta, varre w_E em passo 0.001
% e encontra onde p muda.
fprintf('\nEstagio final isolado (ordem fixa):\n');
stage2 = struct([]);
for g = 1:max(orderGroup)
    ref = res(find(orderGroup == g,1));
    Ln = normalise01(ref.final.ERRloss);
    Cn = normalise01(ref.final.complexity);
    kFine = zeros(size(cfg.fineGrid));
    for q = 1:numel(cfg.fineGrid)
        w = cfg.fineGrid(q);
        [~,ii] = min(w*Ln+(1-w)*Cn);
        kFine(q) = ref.final.k(ii);
    end
    sw = find(diff(kFine) ~= 0);
    stage2(g).kFine = kFine;
    stage2(g).switchAt = cfg.fineGrid(sw+1);
    fprintf('  ordem do grupo %d: ',g);
    for s = sw(:)'
        fprintf('p %d->%d em w_E=%.3f  ',kFine(s),kFine(s+1),cfg.fineGrid(s+1));
    end
    fprintf('\n');
end

%% 7. DIAGNOSTICO C - NRMSE FREE-RUN DE CADA PREFIXO (so diagnostico)
logIdx = arrayfun(@(w) find(abs(cfg.sweep-w) < 1e-9,1),cfg.logWeights);
cache = containers.Map('KeyType','char','ValueType','any');
for c = logIdx
    fin = res(c).final;
    nrmse = nan(numel(fin.k),1);
    for q = 1:numel(fin.k)
        sel = terms(res(c).order(1:fin.k(q)),:);
        nrmse(q) = prefix_nrmse(sel,ut,yt,uv,yv,startIdx,cache);
    end
    res(c).final.valNRMSE = nrmse;
end
% NRMSE do modelo selecionado em cada peso da varredura
selNRMSE = nan(nW,1);
for c = 1:nW
    sel = terms(res(c).order(1:res(c).k),:);
    selNRMSE(c) = prefix_nrmse(sel,ut,yt,uv,yv,startIdx,cache);
end

% Checagem de reproducao do artigo (Tabelas 2-4)
for q = 1:size(cfg.paper,1)
    c = find(abs(cfg.sweep-cfg.paper(q,1)) < 1e-9,1);
    fprintf('Checagem w_E=%.2f: p=%d, NRMSE=%.3f (artigo: %d e %.3f)\n', ...
        cfg.paper(q,1),res(c).k,selNRMSE(c),cfg.paper(q,2),cfg.paper(q,3));
end

%% 8. DIAGNOSTICO D - PERDAS ALTERNATIVAS (so dados de identificacao)
% Previa da opcao (b). Mesma ordem, mesmo C, muda so a perda do estagio final.
alt.names = {'L (artigo)','sqrt(L)','log10(L)'};
alt.k = zeros(nW,numel(alt.names));
for c = 1:nW
    fin = res(c).final;  wE = cfg.sweep(c);
    L = fin.ERRloss;
    Lalt = {L, sqrt(L), log10(max(L,realmin))};
    Cn = normalise01(fin.complexity);
    for a = 1:numel(Lalt)
        [~,ii] = min(wE*normalise01(Lalt{a})+(1-wE)*Cn);
        alt.k(c,a) = fin.k(ii);
    end
end

%% 9. SALVAR TABELAS
T = table(cfg.sweep,[res.k]',[res.C]',selNRMSE,orderGroup, ...
    alt.k(:,1),alt.k(:,2),alt.k(:,3), ...
    'VariableNames',{'wE','p','C','valNRMSE','orderGroup', ...
    'p_L','p_sqrtL','p_log10L'});
writetable(T,fullfile(cfg.outDir,'fase0_sweep.csv'));

for c = logIdx
    tag = sprintf('%03d',round(100*cfg.sweep(c)));
    s = res(c).steps;
    writetable(struct2table(s),fullfile(cfg.outDir,['fase0_steps_wE' tag '.csv']));
    f = res(c).final;
    Tf = table(f.k,f.cumulativeERR,f.ERRloss,normalise01(f.ERRloss), ...
        f.complexity,normalise01(f.complexity),f.modelCost,f.valNRMSE, ...
        'VariableNames',{'p','cumERR','L','Ltil','C','Ctil','J','valNRMSE_diag'});
    writetable(Tf,fullfile(cfg.outDir,['fase0_final_wE' tag '.csv']));
end
save(fullfile(cfg.outDir,['fase0_' caseName '.mat']), ...
    'cfg','res','orderGroup','stage2','selNRMSE','alt','terms','-v7.3');

%% 10. RESUMO NA TELA
c07 = find(abs(cfg.sweep-0.70) < 1e-9,1);
f = res(c07).final;
fprintf('\nw_E = 0.70, estagio final:\n');
fprintf('  p   cumERR          L          L~      C~      J     NRMSE val\n');
for q = 1:numel(f.k)
    Ln = normalise01(f.ERRloss);  Cn = normalise01(f.complexity);
    fprintf('%3d  %.8f  %.3e  %.4f  %.4f  %.4f  %.3f\n', ...
        f.k(q),f.cumulativeERR(q),f.ERRloss(q),Ln(q),Cn(q),f.modelCost(q),f.valNRMSE(q));
end
fprintf('\nOrdem (w_E = 0.70):\n');
for r = 1:cfg.maxTerms
    st = res(c07).steps;
    fprintf('%2d  %-28s ERR=%.3e  rankERR=%4d/%4d  gapJ=%.2e  FROLS=%d\n', ...
        r,term_to_string(terms(res(c07).order(r),:)),st.errSelected(r), ...
        st.errRank(r),st.nAvailable(r),st.gapJ(r),st.sameAsFROLS(r));
end

%% 11. FIGURA
fg = figure('Color','w','Position',[100 100 1100 420]);
subplot(1,2,1);
yyaxis left;  semilogy(f.k,f.ERRloss,'o-','LineWidth',1.5); ylabel('L_p (identificacao)');
yyaxis right; plot(f.k,f.valNRMSE,'s-','LineWidth',1.5); ylabel('NRMSE free-run (validacao, diagnostico)');
xlabel('p'); title('w_E = 0.70: perda ERR x NRMSE do prefixo'); grid on
subplot(1,2,2);
plot(f.k,normalise01(f.ERRloss),'o-',f.k,normalise01(f.complexity),'s-','LineWidth',1.5);
xlabel('p'); legend('L~_p','C~_p','Location','best'); title('Valores normalizados no estagio final'); grid on
exportgraphics(fg,fullfile(cfg.outDir,['fase0_' caseName '.png']),'Resolution',150);
fprintf('\nSaidas em %s\n',cfg.outDir);


%% ========================================================================
%  FUNCOES LOCAIS
% =========================================================================

function [order,errSelected,S] = green_term_order_log( ...
    A,Y,model,dTerm,maxTerms,wE,wC,bits,multExponent)
% Igual a green_term_order do script do artigo, com registro por passo.
% Complexidade da estrutura tentativa: (r-1)*bits + (mult selecionadas +
% mult do candidato)*bits^alpha, identica a model_complexity.
    [m,n] = size(A);
    maxTerms = min([maxTerms,n]);
    sqy = Y'*Y;
    piv = 1:n;
    errSelected = zeros(maxTerms,1);
    S.step = (1:maxTerms)';
    [S.errSelected,S.errRank,S.nAvailable,S.errMin,S.errMax,S.compMin, ...
        S.compMax,S.errNormSel,S.compNormSel,S.Jsel,S.gapJ,S.nTies, ...
        S.sameAsFROLS,S.termIndex] = deal(zeros(maxTerms,1));
    multSel = 0;
    for r = 1:maxTerms
        available = r:n;
        nA = numel(available);
        errC = zeros(nA,1);
        for q = 1:nA
            j = available(q);
            aj = A(r:m,j)'*A(r:m,j);
            bj = A(r:m,j)'*Y(r:m);
            if aj <= eps
                errC(q) = 0;
            else
                errC(q) = (bj^2)/(aj*sqy);
            end
        end
        compC = (r-1)*bits+(multSel+dTerm(piv(available)))*(bits^multExponent);
        errN = normalise01(errC);
        compN = normalise01(compC);
        J = wE*(1-errN)+wC*compN;
        Jmin = min(J);
        tie = find(abs(J-Jmin) <= 1e-12);
        if numel(tie) > 1
            [~,bt] = max(errC(tie));
            lb = tie(bt);
        else
            lb = tie(1);
        end
        Js = sort(J);
        [~,frols] = max(errC);
        S.errSelected(r) = errC(lb);
        S.errRank(r)     = sum(errC > errC(lb))+1;
        S.nAvailable(r)  = nA;
        S.errMin(r) = min(errC);   S.errMax(r) = max(errC);
        S.compMin(r) = min(compC); S.compMax(r) = max(compC);
        S.errNormSel(r) = errN(lb); S.compNormSel(r) = compN(lb);
        S.Jsel(r) = J(lb);
        if nA > 1, S.gapJ(r) = Js(2)-Js(1); else, S.gapJ(r) = NaN; end
        S.nTies(r) = numel(tie);
        S.sameAsFROLS(r) = (frols == lb);

        k = available(lb);
        if k ~= r
            tmp = A(:,r); A(:,r) = A(:,k); A(:,k) = tmp;
            tmp = piv(r); piv(r) = piv(k); piv(k) = tmp;
        end
        S.termIndex(r) = piv(r);
        multSel = multSel+dTerm(piv(r));
        errSelected(r) = errC(lb);
        v = house(A(r:m,r));
        A(r:m,r:n) = rowhouse(A(r:m,r:n),v);
        Y(r:m) = rowhouse(Y(r:m),v);
    end
    order = piv(1:maxTerms);
    S.term = string(arrayfun(@(i) term_to_string(model(i,:)),S.termIndex,'UniformOutput',false));
end


function [kBest,details] = choose_green_size_log( ...
    terms,order,errOrder,minTerms,maxTerms,wE,wC,bits,multExponent)
% Identica a choose_green_size do script do artigo.
    kValues = (minTerms:maxTerms)';
    nK = numel(kValues);
    cumulativeERR = zeros(nK,1);
    errLoss = zeros(nK,1);
    complexity = zeros(nK,1);
    for q = 1:nK
        k = kValues(q);
        cumulativeERR(q) = sum(errOrder(1:k));
        errLoss(q) = max(1-cumulativeERR(q),0);
        complexity(q) = model_complexity(terms(order(1:k),:),bits,multExponent);
    end
    J = wE*normalise01(errLoss)+wC*normalise01(complexity);
    [~,idx] = min(J);
    kBest = kValues(idx);
    details.k = kValues;
    details.cumulativeERR = cumulativeERR;
    details.ERRloss = errLoss;
    details.complexity = complexity;
    details.modelCost = J;
    details.selectedIndex = idx;
end


function nr = prefix_nrmse(sel,ut,yt,uv,yv,startIdx,cache)
% Estima por orthreg nos dados de identificacao e simula em free res.
    key = mat2str(sortrows(sel));
    if isKey(cache,key), nr = cache(key); return, end
    k = size(sel,1);
    [mG,xG] = orthreg(sel,ut,yt,[k 0],0);
    [yh,ytrue] = simulate_continuation_free_run(mG,xG(:,1),uv,yv,startIdx);
    if any(~isfinite(yh))
        nr = Inf;
    else
        nr = norm(ytrue-yh)/norm(ytrue-mean(ytrue));
    end
    cache(key) = nr; %#ok<NASGU>
end


function C = model_complexity(model,bits,multExponent)
    [nAdd,nMult] = model_operation_count(model);
    C = nAdd*bits+nMult*(bits^multExponent);
end


function [nAdd,nMult] = model_operation_count(model)
    nTerms = size(model,1);
    nAdd = max(nTerms-1,0);
    nMult = 0;
    for i = 1:nTerms
        nMult = nMult+nnz(model(i,:) > 0);
    end
end


function z = normalise01(x)
    x = x(:);
    xmin = min(x);  xmax = max(x);
    if abs(xmax-xmin) <= eps(max(abs(x))+1)
        z = zeros(size(x));
    else
        z = (x-xmin)/(xmax-xmin);
    end
end


function [yhat,ytrue] = simulate_continuation_free_run(model,theta,u,y,startIndex)
% Igual ao script do artigo, com parada se divergir.
    u = u(:); y = y(:);
    positiveCodes = model(model > 0);
    lag = max([0;mod(positiveCodes(:),1000)]);
    firstIdx = startIndex-lag;
    uSeg = u(firstIdx:end);
    y0 = y(firstIdx:startIndex-1);
    n = length(uSeg);
    degree = size(model,2);
    ySim = zeros(n,1);
    if lag > 0, ySim(1:lag) = y0; end
    for nn = lag+1:n
        total = 0;
        for i = 1:size(model,1)
            value = theta(i);
            for j = 1:degree
                code = model(i,j);
                if code == 0 || code == -1, continue, end
                lagValue = mod(code,1000);
                switch floor(code/1000)
                    case 1, value = value*ySim(nn-lagValue);
                    case 2, value = value*uSeg(nn-lagValue);
                    case 3, value = 0;
                end
            end
            total = total+value;
        end
        ySim(nn) = total;
        if ~isfinite(total) || abs(total) > 1e6
            ySim(nn:end) = Inf;
            break
        end
    end
    yhat = ySim(lag+1:end);
    ytrue = y(startIndex:end);
end


function txt = term_to_string(termCode)
    factors = {};
    for j = 1:length(termCode)
        code = termCode(j);
        if code == 0 || code == -1, continue, end
        lagValue = mod(code,1000);
        switch floor(code/1000)
            case 1, factors{end+1} = sprintf('y(k-%d)',lagValue); %#ok<AGROW>
            case 2, factors{end+1} = sprintf('u(k-%d)',lagValue); %#ok<AGROW>
            case 3, factors{end+1} = sprintf('e(k-%d)',lagValue); %#ok<AGROW>
        end
    end
    if isempty(factors), txt = '1'; else, txt = strjoin(factors,' * '); end
end
