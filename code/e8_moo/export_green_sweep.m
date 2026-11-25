%% E8 - exporta os modelos Green da varredura de pesos para comparar com NSGA-II
%  Saida: e8_moo/green_sweep_<caso>.json com w_E, codigos, L = 1 - ERR acumulado,
%  C e NRMSE de validacao de cada peso. Tambem exporta a frente exata do DC.
clear
here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
main = fileparts(here);
files = struct('dc','green_dc_motor_sysidentpy_page72_results.mat', ...
               'ced','green_ced_exp2_sysidentpy_results.mat', ...
               'silverbox','green_silverbox_arrow_sysidentpy_results.mat');
for c = fieldnames(files)'
    S = load(fullfile(main,files.(c{1})));  g = S.results.green;
    out = struct('wE',{},'codes',{},'L',{},'C',{},'nrmse',{});
    for i = 1:numel(g)
        out(end+1) = struct('wE',g(i).w_E,'codes',g(i).model,'L',1-g(i).cumulativeERR, ...
            'C',g(i).complexity,'nrmse',g(i).metrics.NRMSE); %#ok<AGROW>
    end
    R = S.results.reference;
    ref = struct('codes',R.model,'C',R.complexity,'nrmse',R.metrics.NRMSE);
    fid = fopen(fullfile(here,['green_sweep_' c{1} '.json']),'w');
    fprintf(fid,'%s',jsonencode(struct('case',c{1},'green',out,'reference',ref)));  fclose(fid);
end
E = load(fullfile(main,'fase2_outputs','dc','e2_exhaustive_all.mat'));
ok = isfinite(E.Cs) & isfinite(E.Ls) & E.ps >= 2;
fid = fopen(fullfile(here,'dc_exhaustive.json'),'w');
fprintf(fid,'%s',jsonencode(struct('C',E.Cs(ok),'L',E.Ls(ok),'frontC',E.Cs(E.front),'frontL',E.Ls(E.front))));
fclose(fid);
disp('ok')
