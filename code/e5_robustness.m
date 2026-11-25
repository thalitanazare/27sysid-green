%% ========================================================================
%  FASE 2 - E5: robustez e determinismo
%  Metodo publicado (L linear, min-max,
%  B = 64, rho = 64^0.585), pesos w_E = 0.3, 0.5, 0.7.
%
%  Variaveis de ambiente
%    FASE5_CASE = dc | ced | silverbox
%    FASE5_PART = split | noise | det | all   (padrao all)
%
%  split : outras particoes dos dados
%          DC        treino nos primeiros 40, 50 e 60 % e treino na 2a metade
%                    validando na 1a (troca)
%          CED       exp2->exp2 (artigo), exp1->exp1, exp2->exp1, exp1->exp2
%          Silverbox treino completo (artigo), 1a metade, 2a metade do treino
%  noise : ruido branco gaussiano somado a saida de IDENTIFICACAO,
%          SNR = 40, 30, 20 dB, R realizacoes (DC e CED 30, Silverbox 10),
%          validacao free run nos dados limpos
%  det   : mesma selecao com 1 thread e com todas as threads, regra de
%          desempate do artigo ('err') e regra proposta ('cost'),
%          w_E = 0, 0.3, 0.5, 0.7, 1
%
%  Linhas de base em cada cenario (mesma ordem FROLS, so ERR):
%    FROLS-AIC  tamanho pelo minimo do AIC = N ln(sigma2_p) + 2p
%    FROLS-max  todos os maxTerms termos (extremo de acuracia, w_E = 1)
%
%  Saidas em fase2_outputs/e5/<caso>/
% ========================================================================

clear; clc; close all; format long g
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
addpath(fullfile(scriptDir,'fase2_lib'));

caseName = lower(strtrim(getenv('FASE5_CASE')));
part = lower(strtrim(getenv('FASE5_PART')));
if isempty(part), part = 'all'; end
assert(any(strcmp(caseName,{'dc','ced','silverbox'})),'FASE5_CASE invalido');
cfg = case_cfg(caseName);
outDir = fullfile(scriptDir,'fase2_outputs','e5',caseName);
if ~exist(outDir,'dir'), mkdir(outDir); end
terms = genterms(cfg.degree,cfg.lagy,cfg.lagu,cfg.lage);
dTerm = sum(terms > 0,2);
[~,~,lag] = get_info(terms);
W = [0.3 0.5 0.7];
fprintf('E5 %s | parte %s | %d candidatos\n',caseName,part,size(terms,1));

%% Particoes
if any(strcmp(part,{'split','all'}))
    maxNumCompThreads(1);
    P = partitions(caseName,scriptDir);
    rows = {};
    for ip = 1:numel(P)
        t0 = tic;
        R = fit_all(P(ip).ut,P(ip).yt,P(ip).uv,P(ip).yv,P(ip).start,terms,dTerm,lag,cfg,W);
        for i = 1:numel(R)
            rows(end+1,:) = {P(ip).name,R(i).label,R(i).p,R(i).nAdd,R(i).nMult, ...
                R(i).C,R(i).nrmse,R(i).key}; %#ok<SAGROW>
            fprintf('  %-22s %-10s p=%2d A/M=%2d/%3d C=%9.1f NRMSE=%.3f\n',P(ip).name, ...
                R(i).label,R(i).p,R(i).nAdd,R(i).nMult,R(i).C,R(i).nrmse);
        end
        fprintf('  (%.1f s)\n',toc(t0));
    end
    T = cell2table(rows,'VariableNames',{'partition','model','p','nAdd','nMult','C','NRMSE','structure'});
    writetable(T,fullfile(outDir,'e5_split.csv'));
end

%% Ruido (Monte Carlo)
if any(strcmp(part,{'noise','all'}))
    P = partitions(caseName,scriptDir);  P0 = P(1);      % particao do artigo
    R0 = fit_all(P0.ut,P0.yt,P0.uv,P0.yv,P0.start,terms,dTerm,lag,cfg,W);
    nomKey = containers.Map({R0.label},{R0.key});
    SNR = [40 30 20];
    if strcmp(caseName,'silverbox'), nR = 10; else, nR = 30; end
    labels = {R0.label};  nL = numel(labels);
    res = cell(numel(SNR),1);
    usePar = ~strcmp(caseName,'dc');
    if usePar
        nW = 6;  if strcmp(caseName,'silverbox'), nW = 4; end   % memoria
        pool = gcp('nocreate');
        if isempty(pool), parpool('Processes',nW); end
    end
    for is = 1:numel(SNR)
        out = cell(nR,1);  snr = SNR(is);
        ut = P0.ut; yt = P0.yt; uv = P0.uv; yv = P0.yv; st = P0.start;
        t0 = tic;
        if usePar
            parfor r = 1:nR
                out{r} = one_realisation(r,snr,ut,yt,uv,yv,st,terms,dTerm,lag,cfg,W);
            end
        else
            for r = 1:nR
                out{r} = one_realisation(r,snr,ut,yt,uv,yv,st,terms,dTerm,lag,cfg,W);
            end
        end
        res{is} = out;
        fprintf('SNR %d dB: %d realizacoes em %.1f s\n',snr,nR,toc(t0));
    end
    % tabela longa e resumo
    rows = {};  sumRows = {};
    for is = 1:numel(SNR)
        for il = 1:nL
            v = zeros(nR,5);  same = false(nR,1);
            for r = 1:nR
                Rr = res{is}{r};  m = Rr(strcmp({Rr.label},labels{il}));
                v(r,:) = [m.p m.nAdd m.nMult m.C m.nrmse];
                same(r) = strcmp(m.key,nomKey(labels{il}));
                rows(end+1,:) = {SNR(is),r,labels{il},m.p,m.nAdd,m.nMult,m.C,m.nrmse,same(r),m.key}; %#ok<AGROW>
            end
            q = prctile(v,[25 50 75]);
            sumRows(end+1,:) = {SNR(is),labels{il},q(2,1),q(1,1),q(3,1),q(2,3),q(2,4),q(1,4),q(3,4), ...
                q(2,5),q(1,5),q(3,5),mean(same),mean(isfinite(v(:,5)))}; %#ok<AGROW>
        end
    end
    T = cell2table(rows,'VariableNames',{'SNR_dB','rep','model','p','nAdd','nMult','C','NRMSE','sameAsNoiseFree','structure'});
    writetable(T,fullfile(outDir,'e5_noise_runs.csv'));
    S = cell2table(sumRows,'VariableNames',{'SNR_dB','model','p_med','p_q1','p_q3','nMult_med', ...
        'C_med','C_q1','C_q3','NRMSE_med','NRMSE_q1','NRMSE_q3','fracSameStructure','fracStable'});
    writetable(S,fullfile(outDir,'e5_noise_summary.csv'));
    % comparacao pareada Green x FROLS-AIC em cada realizacao
    pr = {};
    for is = 1:numel(SNR)
        for w = W
            lab = sprintf('Green %.1f',w);
            dC = zeros(nR,1); dN = zeros(nR,1);
            for r = 1:nR
                Rr = res{is}{r};
                g = Rr(strcmp({Rr.label},lab));  a = Rr(strcmp({Rr.label},'FROLS-AIC'));
                dC(r) = 100*(1-g.C/a.C);  dN(r) = g.nrmse-a.nrmse;
            end
            pr(end+1,:) = {SNR(is),lab,median(dC),min(dC),max(dC),median(dN), ...
                prctile(dN,25),prctile(dN,75),mean(dC > 0)}; %#ok<AGROW>
        end
    end
    Pt = cell2table(pr,'VariableNames',{'SNR_dB','model','Cred_vs_AIC_med_pct','Cred_min','Cred_max', ...
        'dNRMSE_med','dNRMSE_q1','dNRMSE_q3','fracCheaper'});
    writetable(Pt,fullfile(outDir,'e5_noise_paired.csv'));
    disp(S); disp(Pt)
end

%% Determinismo
if any(strcmp(part,{'det','all'}))
    P = partitions(caseName,scriptDir);  P0 = P(1);
    Pm = build_pr(terms,P0.ut,P0.yt);  A0 = Pm(lag+1:end,:);  clear Pm
    Y0 = P0.yt(lag+1:end);
    maxNumCompThreads('automatic');  nMax = maxNumCompThreads;
    rows = {};
    for tieRule = {'err','cost'}
        for wE = [0 0.3 0.5 0.7 1]
            keys = cell(1,3);  keys(:) = {''};  pp = zeros(1,3);  nm = zeros(1,3);
            thr = [1 nMax 1];                      % 1, todas, 1 de novo
            if strcmp(caseName,'silverbox'), thr = [1 nMax 1]; nT = 2; else, nT = 3; end
            for it = 1:nT
                maxNumCompThreads(thr(it));
                [order,errSel] = term_order(A0,Y0,dTerm,cfg.maxTerms,wE,1-wE,64,64^0.585,'C',tieRule{1});
                mdl = run_green(A0,Y0,terms,dTerm,cfg,wE,1-wE,64,64^0.585,'C','C',order);
                keys{it} = mdl.key;  pp(it) = mdl.p;  nm(it) = mdl.nMult;
                keys{it} = [keys{it} ' | ordem ' mat2str(order)];
            end
            rows(end+1,:) = {tieRule{1},wE,pp(1),nm(1),pp(2),nm(2), ...
                nT < 3 || strcmp(keys{1},keys{3}),strcmp(keys{1},keys{2})}; %#ok<AGROW>
            fprintf('  tie=%-4s wE=%.1f  1thr p=%2d M=%2d | %dthr p=%2d M=%2d | repete=%d | igual entre threads=%d\n', ...
                tieRule{1},wE,pp(1),nm(1),nMax,pp(2),nm(2),nT < 3 || strcmp(keys{1},keys{3}),strcmp(keys{1},keys{2}));
        end
    end
    maxNumCompThreads(nMax);
    T = cell2table(rows,'VariableNames',{'tieRule','wE','p_1thr','nMult_1thr','p_allthr','nMult_allthr', ...
        'repeatable_1thr','sameOrderAcrossThreads'});
    writetable(T,fullfile(outDir,'e5_determinism.csv'));
end
fprintf('E5 %s concluido.\n',caseName);


%% ========================================================================
function cfg = case_cfg(caseName)
    cfg.lage = 0; cfg.minTerms = 2; cfg.bits = 64;
    switch caseName
        case 'dc',        cfg.degree = 2; cfg.lagy = 2;  cfg.lagu = 2;  cfg.maxTerms = 15;
        case 'ced',       cfg.degree = 3; cfg.lagy = 10; cfg.lagu = 10; cfg.maxTerms = 40;
        case 'silverbox', cfg.degree = 3; cfg.lagy = 10; cfg.lagu = 10; cfg.maxTerms = 20;
    end
end


function P = partitions(caseName,scriptDir)
    P = struct('name',{},'ut',{},'yt',{},'uv',{},'yv',{},'start',{});
    switch caseName
        case 'dc'
            [ut,yt,uv,yv,st] = load_case('dc',scriptDir);
            P(end+1) = mk('50/50 (artigo)',ut,yt,uv,yv,st);
            n = numel(yv);
            for f = [0.4 0.6]
                nT = round(f*n);
                P(end+1) = mk(sprintf('%d/%d',round(100*f),round(100*(1-f))), ...
                    uv(1:nT),yv(1:nT),uv,yv,nT+1); %#ok<AGROW>
            end
            h = n/2;
            P(end+1) = mk('troca (2a -> 1a metade)',uv(h+1:end),yv(h+1:end),uv(1:h),yv(1:h),3);
        case 'ced'
            D1 = load(fullfile(scriptDir,'CED_outputs','CED_experiment_1_data.mat'));
            D2 = load(fullfile(scriptDir,'CED_outputs','CED_experiment_2_data.mat'));
            g = @(D) deal(double(D.u_train(:)),double(D.y_train(:)),double(D.u_test(:)), ...
                double(D.y_test(:)),double(D.n_initial)+1);
            [u1,y1,uv1,yv1,s1] = g(D1);  [u2,y2,uv2,yv2,s2] = g(D2);
            P(end+1) = mk('exp2 -> exp2 (artigo)',u2,y2,uv2,yv2,s2);
            P(end+1) = mk('exp1 -> exp1',u1,y1,uv1,yv1,s1);
            P(end+1) = mk('exp2 -> exp1',u2,y2,uv1,yv1,s1);
            P(end+1) = mk('exp1 -> exp2',u1,y1,uv2,yv2,s2);
        case 'silverbox'
            [ut,yt,uv,yv,st] = load_case('silverbox',scriptDir);
            P(end+1) = mk('treino completo (artigo)',ut,yt,uv,yv,st);
            h = floor(numel(yt)/2);
            P(end+1) = mk('1a metade do treino',ut(1:h),yt(1:h),uv,yv,st);
            P(end+1) = mk('2a metade do treino',ut(h+1:end),yt(h+1:end),uv,yv,st);
    end
end


function s = mk(name,ut,yt,uv,yv,st)
    s = struct('name',name,'ut',ut(:),'yt',yt(:),'uv',uv(:),'yv',yv(:),'start',st);
end


function R = one_realisation(r,snr,ut,yt,uv,yv,st,terms,dTerm,lag,cfg,W)
    s = RandStream('mt19937ar','Seed',1000*snr+r);
    ytn = yt+std(yt)*10^(-snr/20)*randn(s,size(yt));
    R = fit_all(ut,ytn,uv,yv,st,terms,dTerm,lag,cfg,W);
end


function R = fit_all(ut,yt,uv,yv,st,terms,dTerm,lag,cfg,W)
    Pm = build_pr(terms,ut,yt);  A0 = Pm(lag+1:end,:);  clear Pm
    Y0 = yt(lag+1:end);  N = numel(Y0);
    R = struct('label',{},'p',{},'nAdd',{},'nMult',{},'C',{},'nrmse',{},'key',{});
    for wE = W
        mdl = run_green(A0,Y0,terms,dTerm,cfg,wE,1-wE,64,64^0.585,'C','C',[]);
        R(end+1) = evaluate(sprintf('Green %.1f',wE),mdl.sel,ut,yt,uv,yv,st); %#ok<AGROW>
    end
    % FROLS so ERR (mesma implementacao, desempate de colineares pelo menor custo)
    [order,errSel] = term_order(A0,Y0,dTerm,cfg.maxTerms,1,0,64,64^0.585,'none','cost');
    L = max(1-cumsum(errSel),eps);
    aic = N*log(L*(Y0'*Y0)/N)+2*(1:numel(L))';
    [~,pA] = min(aic);
    R(end+1) = evaluate('FROLS-AIC',terms(order(1:pA),:),ut,yt,uv,yv,st);
    R(end+1) = evaluate('FROLS-max',terms(order,:),ut,yt,uv,yv,st);
end


function m = evaluate(label,sel,ut,yt,uv,yv,st)
    k = size(sel,1);
    ws = warning('off','all');
    [mG,xG] = orthreg(sel,ut,yt,[k 0],0);
    warning(ws);
    [yh,ytrue] = simulate_free_run(mG,xG(:,1),uv,yv,st);
    if any(~isfinite(yh)), nr = Inf; else, nr = norm(ytrue-yh)/norm(ytrue-mean(ytrue)); end
    m.label = label;  m.p = k;  m.nAdd = k-1;  m.nMult = nnz(sel > 0);
    m.C = m.nAdd*64+m.nMult*64^1.585;  m.nrmse = nr;
    m.key = strjoin(sort(arrayfun(@(j) term_to_string(sel(j,:)),1:k,'UniformOutput',false)),' + ');
end
