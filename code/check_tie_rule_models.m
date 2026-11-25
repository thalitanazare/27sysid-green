%% Verificacao: a regra de desempate revisada ('cost', padrao do term_order)
%  reproduz termo a termo os modelos das Tabelas 2-5?
%  Compara com os modelos exportados no E3b (fase2_outputs/e3b/<caso>_models.json),
%  que reproduzem as tabelas do artigo (DC w_E = 1 na forma 12/20 do artigo).
clear; clc
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
addpath(fullfile(scriptDir,'fase2_lib'));
cases = {'dc','ced','silverbox'};
W = [0 0.3 0.5 0.7 1];
for ic = 1:3
    c = cases{ic};
    switch c
        case 'dc',        cfg.degree = 2; cfg.lagy = 2;  cfg.lagu = 2;  cfg.maxTerms = 15;
        case 'ced',       cfg.degree = 3; cfg.lagy = 10; cfg.lagu = 10; cfg.maxTerms = 40;
        case 'silverbox', cfg.degree = 3; cfg.lagy = 10; cfg.lagu = 10; cfg.maxTerms = 20;
    end
    cfg.lage = 0; cfg.minTerms = 2;
    [ut,yt] = load_case(c,scriptDir);
    terms = genterms(cfg.degree,cfg.lagy,cfg.lagu,cfg.lage);
    dTerm = sum(terms > 0,2);  [~,~,lag] = get_info(terms);
    P = build_pr(terms,ut,yt);  A0 = P(lag+1:end,:);  Y0 = yt(lag+1:end);  clear P
    D = jsondecode(fileread(fullfile(scriptDir,'fase2_outputs','e3b',[c '_models.json'])));
    for w = W
        mdl = run_green(A0,Y0,terms,dTerm,cfg,w,1-w,64,64^0.585,'C','C',[]);
        new = sort(arrayfun(@(j) term_to_string(mdl.sel(j,:)),1:size(mdl.sel,1),'UniformOutput',false));
        lab = sprintf('Green (%.2f)',w);
        m = D.models(strcmp({D.models.label},lab));
        codes = m.codes;  if size(codes,2) ~= size(terms,2), codes = codes'; end
        codes = [codes zeros(size(codes,1),size(terms,2)-size(codes,2))];
        old = sort(arrayfun(@(j) term_to_string(codes(j,:)),1:size(codes,1),'UniformOutput',false));
        same = isequal(new,old);
        fprintf('%-9s wE=%.2f  novo p=%2d M=%2d | tabela p=%2d M=%2d | mesmos termos: %d\n', ...
            c,w,mdl.p,mdl.nMult,size(codes,1),nnz(codes>0),same);
        if ~same
            fprintf('   so no novo: %s\n',strjoin(setdiff(new,old),', '));
            fprintf('   so na tabela: %s\n',strjoin(setdiff(old,new),', '));
        end
    end
end
