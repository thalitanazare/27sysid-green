%% ========================================================================
%  FASE 2 - E1 (sensibilidade a rho) e E2 (ablacao + enumeracao exaustiva)
%  Caso em FASE2_CASE ('dc', 'ced', 'silverbox').
%
%  E1
%     C = B*[(p-1) + rho*sum(d_i)], rho = B^(alpha-1). Roda o metodo publicado
%     com rho = 0.997 (medido no i5-8500, latencia double, e3_timing/),
%     1 (contagem uniforme), 6.92 (Toom-3), 11.39 (Karatsuba, artigo) e
%     64 (schoolbook). Checagem de invariancia: B = 32 com o mesmo rho deve
%     dar exatamente a mesma selecao.
%
%  E2
%     Ablacao com os mesmos pesos:
%       'Green (artigo)'    complexidade C nos dois estagios
%       'Termos'            complexidade = p nos dois estagios (no estagio
%                           sequencial vira FROLS puro, pois todo candidato
%                           acrescenta 1 termo)
%       'FROLS + final C'   estagio sequencial so com ERR, estagio final com C
%     Enumeracao exaustiva (so 'dc', 15 candidatos): todos os subconjuntos
%     com 2..15 termos, perda L = SSE/y'y nos dados de identificacao e C.
%     Frente de Pareto exata (L, C) e distancia dos modelos Green a ela.
%
%  Validacao: so para reportar NRMSE free-run, nunca para selecionar.
%  Saidas em fase2_outputs/<caso>/
% ========================================================================

clear; clc; close all; format long g
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
caseName = lower(getenv('FASE2_CASE'));
if isempty(caseName), caseName = 'dc'; end

%% 1. CONFIGURACAO
cfg.case = caseName; cfg.lage = 0; cfg.minTerms = 2; cfg.bits = 64;
switch caseName
    case 'dc',        cfg.degree = 2; cfg.lagy = 2;  cfg.lagu = 2;  cfg.maxTerms = 15;
    case 'ced',       cfg.degree = 3; cfg.lagy = 10; cfg.lagu = 10; cfg.maxTerms = 40;
    case 'silverbox', cfg.degree = 3; cfg.lagy = 10; cfg.lagu = 10; cfg.maxTerms = 20;
    otherwise, error('FASE2_CASE desconhecido: %s',caseName);
end
cfg.weights = [0.30 0.50 0.70];
cfg.rho = struct('name',{'medido PC (0.997)','uniforme','Toom-3','Karatsuba (artigo)','schoolbook'}, ...
                 'value',{0.9971, 1, 64^(log(5)/log(3)-1), 64^0.585, 64});
cfg.outDir = fullfile(scriptDir,'fase2_outputs',caseName);
if ~exist(cfg.outDir,'dir'), mkdir(cfg.outDir); end

%% 2. DADOS (igual aos scripts do artigo)
[ut,yt,uv,yv,startIdx] = load_case(caseName,scriptDir);
terms = genterms(cfg.degree,cfg.lagy,cfg.lagu,cfg.lage);
dTerm = sum(terms > 0,2);
[~,~,lag] = get_info(terms);
P = build_pr(terms,ut,yt);  A0 = P(lag+1:end,:);  clear P
Y0 = yt(lag+1:end);
cache = containers.Map('KeyType','char','ValueType','any');
fprintf('Caso %s | %d candidatos | treino %d\n',caseName,size(terms,1),numel(yt));

% Ordem FROLS pura (independe do peso): usada em E2
ordFROLS = term_order(A0,Y0,dTerm,cfg.maxTerms,1,0,cfg.bits,1,'C');

%% 3. E1 - SENSIBILIDADE A RHO
rows = {};
for iw = 1:numel(cfg.weights)
    wE = cfg.weights(iw); wC = 1-wE;
    for ir = 1:numel(cfg.rho)
        rho = cfg.rho(ir).value;
        tc = tic;
        mdl = run_green(A0,Y0,terms,dTerm,cfg,wE,wC,64,rho,'C','C',[]);
        nr = val_nrmse(mdl.sel,ut,yt,uv,yv,startIdx,cache);
        rows(end+1,:) = {wE,cfg.rho(ir).name,rho,64,mdl.p,mdl.nAdd,mdl.nMult, ...
            Ckar(mdl.nAdd,mdl.nMult),nr,mdl.key}; %#ok<AGROW>
        fprintf('E1 w_E=%.2f rho=%-20s p=%2d A/M=%d/%d NRMSE=%.3f (%.1f s)\n', ...
            wE,cfg.rho(ir).name,mdl.p,mdl.nAdd,mdl.nMult,nr,toc(tc));
    end
end
% Invariancia: B = 32 com o mesmo rho Karatsuba, peso 0.5
mdlB = run_green(A0,Y0,terms,dTerm,cfg,0.5,0.5,32,64^0.585,'C','C',[]);
mdlR = run_green(A0,Y0,terms,dTerm,cfg,0.5,0.5,64,64^0.585,'C','C',[]);
fprintf('Invariancia B=32 vs B=64 (mesmo rho): mesma ordem = %d, mesmo p = %d\n', ...
    isequal(mdlB.order,mdlR.order),mdlB.p == mdlR.p);
T1 = cell2table(rows,'VariableNames',{'wE','rho_name','rho','B','p','A','M', ...
    'C_karatsuba','valNRMSE','terms'});
writetable(T1,fullfile(cfg.outDir,'e1_rho.csv'));

%% 4. E2 - ABLACAO
rows = {};
variants = {'Green (artigo)','C','C',[]; 'Termos','p','p',[]; 'FROLS + final C','none','C',ordFROLS};
for iw = 1:numel(cfg.weights)
    wE = cfg.weights(iw); wC = 1-wE;
    for iv = 1:size(variants,1)
        mdl = run_green(A0,Y0,terms,dTerm,cfg,wE,wC,64,64^0.585, ...
            variants{iv,2},variants{iv,3},variants{iv,4});
        nr = val_nrmse(mdl.sel,ut,yt,uv,yv,startIdx,cache);
        rows(end+1,:) = {wE,variants{iv,1},mdl.p,mdl.nAdd,mdl.nMult, ...
            Ckar(mdl.nAdd,mdl.nMult),mdl.L,nr,mdl.key}; %#ok<AGROW>
        fprintf('E2 w_E=%.2f %-16s p=%2d A/M=%d/%d C=%9.2f L=%.3e NRMSE=%.3f\n', ...
            wE,variants{iv,1},mdl.p,mdl.nAdd,mdl.nMult,Ckar(mdl.nAdd,mdl.nMult),mdl.L,nr);
    end
end
T2 = cell2table(rows,'VariableNames',{'wE','variant','p','A','M','C_karatsuba', ...
    'L_ident','valNRMSE','terms'});
writetable(T2,fullfile(cfg.outDir,'e2_ablation.csv'));

%% 5. E2 - ENUMERACAO EXAUSTIVA (so DC motor)
if strcmp(caseName,'dc')
    nC = size(terms,1);
    yy = Y0'*Y0;
    nSub = 2^nC-1;
    Ls = nan(nSub,1); Cs = nan(nSub,1); ps = zeros(nSub,1);
    for s = 1:nSub
        idx = find(bitget(s,1:nC));
        if numel(idx) < cfg.minTerms, continue, end
        X = A0(:,idx);
        th = X\Y0;
        Ls(s) = max(sum((Y0-X*th).^2)/yy,0);
        nM = sum(dTerm(idx));
        Cs(s) = Ckar(numel(idx)-1,nM);
        ps(s) = numel(idx);
    end
    ok = find(~isnan(Ls));
    % Frente de Pareto exata em (L, C): nao dominados
    [~,o] = sortrows([Cs(ok) Ls(ok)]);
    cand = ok(o); front = []; bestL = inf;
    for s = cand'
        if Ls(s) < bestL - 1e-15, front(end+1) = s; bestL = Ls(s); end %#ok<AGROW>
    end
    fprintf('\nExaustivo DC: %d subconjuntos avaliados, %d na frente de Pareto (L, C)\n', ...
        numel(ok),numel(front));
    % NRMSE free-run da frente (diagnostico) e distancia dos Green
    frontRows = {};
    for s = front
        idx = find(bitget(s,1:nC));
        nr = val_nrmse(terms(idx,:),ut,yt,uv,yv,startIdx,cache);
        frontRows(end+1,:) = {ps(s),Cs(s),Ls(s),nr,strjoin(arrayfun(@(i) ...
            term_to_string(terms(i,:)),idx,'UniformOutput',false),' + ')}; %#ok<AGROW>
    end
    TF = cell2table(frontRows,'VariableNames',{'p','C','L_ident','valNRMSE','terms'});
    writetable(TF,fullfile(cfg.outDir,'e2_exhaustive_front.csv'));
    % Para cada modelo Green: menor L possivel com C <= C_green, e se ele esta na frente
    gRows = {};
    for iw = 1:numel(cfg.weights)
        g = T2(T2.wE == cfg.weights(iw) & strcmp(T2.variant,'Green (artigo)'),:);
        Lbest = min(Ls(Cs <= g.C_karatsuba + 1e-9));
        Cbest = min(Cs(Ls <= g.L_ident + 1e-12));
        gRows(end+1,:) = {g.wE,g.p,g.C_karatsuba,g.L_ident,Lbest, ...
            g.L_ident/Lbest,Cbest,g.C_karatsuba/Cbest}; %#ok<AGROW>
        fprintf('Green w_E=%.2f: L=%.3e | melhor L com C<=C_G: %.3e (razao %.2f) | menor C com L<=L_G: %.1f\n', ...
            g.wE,g.L_ident,Lbest,g.L_ident/Lbest,Cbest);
    end
    TG = cell2table(gRows,'VariableNames',{'wE','p','C','L','Lbest_sameC','L_ratio', ...
        'Cbest_sameL','C_ratio'});
    writetable(TG,fullfile(cfg.outDir,'e2_exhaustive_gap.csv'));
    save(fullfile(cfg.outDir,'e2_exhaustive_all.mat'),'Ls','Cs','ps','front');
end
fprintf('\nSaidas em %s\n',cfg.outDir);


%% ========================================================================
%  FUNCOES LOCAIS
% =========================================================================

function C = Ckar(nAdd,nMult)
% Complexidade do artigo (B = 64, alpha = 1.585), para comparar tudo na mesma escala.
    C = nAdd*64+nMult*64^1.585;
end


function mdl = run_green(A0,Y0,terms,dTerm,cfg,wE,wC,B,rho,stage1,stage2,fixedOrder)
% stage1/stage2: 'C' (operacoes), 'p' (numero de termos) ou 'none' (so ERR).
    if isempty(fixedOrder)
        if strcmp(stage1,'none'), w1 = [1 0]; else, w1 = [wE wC]; end
        [order,errSel] = term_order(A0,Y0,dTerm,cfg.maxTerms,w1(1),w1(2),B,rho,stage1);
    else
        order = fixedOrder;
        errSel = order_err(A0,Y0,order);
    end
    k = (cfg.minTerms:cfg.maxTerms)';
    L = max(1-cumsum(errSel),0);  L = L(k);
    nA = k-1;  nM = cumsum(dTerm(order(:)));  nM = nM(k);
    if strcmp(stage2,'p'), cost = k; else, cost = B*(nA+rho*nM); end
    J = wE*normalise01(L)+wC*normalise01(cost);
    [~,i] = min(J);
    mdl.order = order; mdl.p = k(i); mdl.L = L(i);
    mdl.sel = terms(order(1:mdl.p),:);
    mdl.nAdd = mdl.p-1; mdl.nMult = nM(i);
    mdl.key = strjoin(arrayfun(@(j) term_to_string(terms(j,:)),order(1:mdl.p),'UniformOutput',false),' + ');
end


function [order,errSelected] = term_order(A,Y,dTerm,maxTerms,wE,wC,B,rho,mode)
% Mesmo algoritmo de green_term_order (artigo). Complexidade da estrutura
% tentativa: mode 'C' -> B*[(r-1) + rho*mult]; 'p' ou 'none' -> constante
% entre candidatos (normalizacao zera o termo de custo).
    [m,n] = size(A);
    maxTerms = min(maxTerms,n);
    sqy = Y'*Y;  piv = 1:n;  errSelected = zeros(maxTerms,1);  multSel = 0;
    for r = 1:maxTerms
        available = r:n;  nA = numel(available);
        errC = zeros(nA,1);
        for q = 1:nA
            j = available(q);
            aj = A(r:m,j)'*A(r:m,j);  bj = A(r:m,j)'*Y(r:m);
            if aj <= eps, errC(q) = 0; else, errC(q) = (bj^2)/(aj*sqy); end
        end
        if strcmp(mode,'C')
            compC = B*((r-1)+rho*(multSel+dTerm(piv(available))));
        else
            compC = r*ones(nA,1);
        end
        J = wE*(1-normalise01(errC))+wC*normalise01(compC);
        Jmin = min(J);  tie = find(abs(J-Jmin) <= 1e-12);
        if numel(tie) > 1, [~,bt] = max(errC(tie)); lb = tie(bt); else, lb = tie(1); end
        k = available(lb);
        if k ~= r
            tmp = A(:,r); A(:,r) = A(:,k); A(:,k) = tmp;
            tmp = piv(r); piv(r) = piv(k); piv(k) = tmp;
        end
        multSel = multSel+dTerm(piv(r));
        errSelected(r) = errC(lb);
        v = house(A(r:m,r));
        A(r:m,r:n) = rowhouse(A(r:m,r:n),v);
        Y(r:m) = rowhouse(Y(r:m),v);
    end
    order = piv(1:maxTerms);
end


function errSel = order_err(A,Y,order)
% ERR de cada termo de uma ordem fixa (ortogonalizacao sequencial).
    A = A(:,order);  [m,n] = size(A);  sqy = Y'*Y;  errSel = zeros(n,1);
    for r = 1:n
        aj = A(r:m,r)'*A(r:m,r);  bj = A(r:m,r)'*Y(r:m);
        if aj > eps, errSel(r) = (bj^2)/(aj*sqy); end
        v = house(A(r:m,r));
        A(r:m,r:n) = rowhouse(A(r:m,r:n),v);
        Y(r:m) = rowhouse(Y(r:m),v);
    end
end


function [ut,yt,uv,yv,startIdx] = load_case(caseName,scriptDir)
    switch caseName
        case 'dc'
            Yraw = readmatrix(fullfile(scriptDir,'y_cc.csv'));
            Uraw = readmatrix(fullfile(scriptDir,'u_cc.csv'));
            Yraw = Yraw(:); Uraw = Uraw(:);
            idxD = 1:500:min(numel(Yraw),numel(Uraw));
            y = Yraw(idxD); u = Uraw(idxD);
            nTrain = numel(y)/2;
            ut = u(1:nTrain); yt = y(1:nTrain); uv = u; yv = y; startIdx = nTrain+1;
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
end


function nr = val_nrmse(sel,ut,yt,uv,yv,startIdx,cache)
    key = mat2str(sortrows(sel));
    if isKey(cache,key), nr = cache(key); return, end
    k = size(sel,1);
    ws = warning('off','MATLAB:nearlySingularMatrix');
    [mG,xG] = orthreg(sel,ut,yt,[k 0],0);
    warning(ws);
    [yh,ytrue] = simulate_free_run(mG,xG(:,1),uv,yv,startIdx);
    if any(~isfinite(yh)), nr = Inf; else, nr = norm(ytrue-yh)/norm(ytrue-mean(ytrue)); end
    cache(key) = nr; %#ok<NASGU>
end


function [yhat,ytrue] = simulate_free_run(model,theta,u,y,startIndex)
    u = u(:); y = y(:);
    positiveCodes = model(model > 0);
    lag = max([0;mod(positiveCodes(:),1000)]);
    firstIdx = startIndex-lag;
    uSeg = u(firstIdx:end);  y0 = y(firstIdx:startIndex-1);
    n = length(uSeg);  degree = size(model,2);  ySim = zeros(n,1);
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
        if ~isfinite(total) || abs(total) > 1e6, ySim(nn:end) = Inf; break, end
    end
    yhat = ySim(lag+1:end);  ytrue = y(startIndex:end);
end


function z = normalise01(x)
    x = x(:);  xmin = min(x);  xmax = max(x);
    if abs(xmax-xmin) <= eps(max(abs(x))+1), z = zeros(size(x));
    else, z = (x-xmin)/(xmax-xmin); end
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
        end
    end
    if isempty(factors), txt = '1'; else, txt = strjoin(factors,'*'); end
end
