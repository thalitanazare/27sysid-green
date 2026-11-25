%% E7b - tempo e energia medidos da identificacao (RAPL)
%  Mesma implementacao (fase2_lib), uma thread. Para cada caso:
%    Green      : run_green com w_E = 0.5 (ordenacao Green + selecao final)
%    FROLS      : uma ordenacao so por ERR ate p_max (selecao final por L)
%    FROLS-AIC  : como no SysIdentPy, o tamanho e escolhido por AIC e a
%                 ordenacao e refeita para cada tamanho n = 1, ..., p_max
%  A matriz de candidatos e montada antes e nao entra nas medicoes.
%  Energia dinamica = energia (pacote + DRAM) - P_repouso x tempo.
%  Saida: fase2_outputs/e7/e7_ident_energy.csv
clear; clc
here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
main = fileparts(here);
addpath(fullfile(main,'fase2_lib'));  addpath(main);
maxNumCompThreads(1);
outDir = fullfile(main,'fase2_outputs','e7');  if ~exist(outDir,'dir'), mkdir(outDir); end
pkg  = System.Diagnostics.PerformanceCounter('Energy Meter','Energy','RAPL_Package0_PKG',true);
dram = System.Diagnostics.PerformanceCounter('Energy Meter','Energy','RAPL_Package0_DRAM',true);
readE = @() (double(pkg.RawValue) + double(dram.RawValue))*3.6e-9;   % J
rho = 64^0.585;
cases = {'dc','ced','silverbox'};
cycles = [6 5 3];  nIdle = [3 3 5];  maxSpread = 6;   % protocolo v2 (W)
rows = {};
for ic = 1:3
    c = cases{ic};
    switch c
        case 'dc',        cfg.degree = 2; cfg.lagy = 2;  cfg.lagu = 2;  cfg.maxTerms = 15;
        case 'ced',       cfg.degree = 3; cfg.lagy = 10; cfg.lagu = 10; cfg.maxTerms = 40;
        case 'silverbox', cfg.degree = 3; cfg.lagy = 10; cfg.lagu = 10; cfg.maxTerms = 20;
    end
    cfg.lage = 0; cfg.minTerms = 2;
    [ut,yt] = load_case(c,main);
    terms = genterms(cfg.degree,cfg.lagy,cfg.lagu,cfg.lage);
    dTerm = sum(terms > 0,2);  [~,~,lag] = get_info(terms);
    P = build_pr(terms,ut,yt);  A0 = P(lag+1:end,:);  Y0 = yt(lag+1:end);  clear P
    N = numel(Y0);
    procs = {'Green','FROLS','FROLS-AIC'};
    for ip = 1:3
        job = @() run_proc(procs{ip},A0,Y0,terms,dTerm,cfg,rho,N);
        % numero de repeticoes por ciclo para uma carga de pelo menos ~2 s
        if ic == 3
            reps = 1;   % Silverbox: cada execucao ja dura dezenas de segundos
        else
            t1 = tic;  job();  t1 = toc(t1);
            reps = max(1,ceil(2/t1));
        end
        K = cycles(ic);  if ic == 3 && ip == 3, K = 2; end
        d = zeros(K,1); tt = d; base = d; spread = d; prun = d;
        for k = 1:K
            [d(k),tt(k),base(k),spread(k),prun(k)] = bracketed(readE,job,reps,nIdle(ic));
        end
        ok = spread <= maxSpread;  if nnz(ok) < min(3,K), ok = true(K,1); end
        dyn = median(d(ok))/reps;  ts = median(tt)/reps;
        rows(end+1,:) = {c,procs{ip},reps,K,nnz(spread<=maxSpread),ts,dyn, ...
            median(prun(ok)-base(ok)),median(base(ok)),median(prun(ok))}; %#ok<SAGROW>
        fprintf('%-9s %-9s %9.4f s  %10.3f J dinamica  (dP %.2f W, base %.1f W, aceitos %d/%d)\n', ...
            c,procs{ip},ts,dyn,median(prun(ok)-base(ok)),median(base(ok)),nnz(spread<=maxSpread),K);
    end
    clear A0
end
T = cell2table(rows,'VariableNames',{'case','procedure','reps','cycles','accepted','time_s','dynamic_J','dP_W','P_idle_W','P_run_W'});
writetable(T,fullfile(outDir,'e7_ident_energy_v2.csv'));
disp(T)

function run_proc(proc,A0,Y0,terms,dTerm,cfg,rho,N)
switch proc
    case 'Green'
        mdl = run_green(A0,Y0,terms,dTerm,cfg,0.5,0.5,64,rho,'C','C',[]); %#ok<NASGU>
    case 'FROLS'
        mdl = run_green(A0,Y0,terms,dTerm,cfg,1,0,64,rho,'none','C',[]); %#ok<NASGU>
    case 'FROLS-AIC'
        aic = inf(cfg.maxTerms,1);
        for n = 1:cfg.maxTerms
            [~,errSel] = term_order(A0,Y0,dTerm,n,1,0,64,rho,'none');
            L = max(1-sum(errSel),eps);
            aic(n) = N*log(L*(Y0'*Y0)/N)+2*n;
        end
        [~,pA] = min(aic); %#ok<ASGLU>
end
end

function p = idle_windows(readE,n)
p = zeros(n,1);
e0 = readE();  t0 = tic;
for i = 1:n
    pause(1);
    e1 = readE();  dt = toc(t0);
    p(i) = (e1-e0)/dt;
    e0 = e1;  t0 = tic;
end
end

function [dyn,t,base,spread,prun] = bracketed(readE,job,reps,nIdle)
% repouso (nIdle janelas de 1 s) - carga - repouso; linha de base = mediana
before = idle_windows(readE,nIdle);
e2 = readE();  t2 = tic;
for r = 1:reps, job(); end
t = toc(t2);  E = readE()-e2;
after = idle_windows(readE,nIdle);
w = [before; after];
base = median(w);  spread = max(w)-min(w);
dyn = E - base*t;  prun = E/t;
end
