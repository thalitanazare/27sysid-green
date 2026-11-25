%% NRMSE do extremo ERR-only do CED com a regra de desempate revisada
clear; clc
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
addpath(fullfile(scriptDir,'fase2_lib'));
cfg.degree = 3; cfg.lagy = 10; cfg.lagu = 10; cfg.maxTerms = 40; cfg.lage = 0; cfg.minTerms = 2;
[ut,yt,uv,yv,st] = load_case('ced',scriptDir);
terms = genterms(cfg.degree,cfg.lagy,cfg.lagu,cfg.lage);
dTerm = sum(terms > 0,2);  [~,~,lag] = get_info(terms);
P = build_pr(terms,ut,yt);  A0 = P(lag+1:end,:);  Y0 = yt(lag+1:end);  clear P
for rule = {'cost','err'}
    [order,~] = term_order(A0,Y0,dTerm,cfg.maxTerms,1,0,64,64^0.585,'C',rule{1});
    mdl = run_green(A0,Y0,terms,dTerm,cfg,1,0,64,64^0.585,'C','C',order);
    k = size(mdl.sel,1);
    ws = warning('off','all');  [mG,xG] = orthreg(mdl.sel,ut,yt,[k 0],0);  warning(ws);
    [yh,ytrue] = simulate_free_run(mG,xG(:,1),uv,yv,st);
    nr = norm(ytrue-yh)/norm(ytrue-mean(ytrue));
    fprintf('regra %-4s p=%d A/M=%d/%d NRMSE=%.4f R2=%.4f\n',rule{1},mdl.p,mdl.p-1,mdl.nMult,nr,1-nr^2);
end
