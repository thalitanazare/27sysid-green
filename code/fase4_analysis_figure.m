%% ========================================================================
%  FASE 4 - figura de analise da selecao
%  %
%  (a) DC motor: todos os subconjuntos (C, L_p), frente de Pareto exata e
%      modelos Green (E2, fase2_outputs/dc/e2_exhaustive_all.mat e _gap.csv)
%  (b) perda normalizada L~_p ao longo da ordem Green com w_E = 0.70
%      (Fase 0, diag_fase0_outputs/**/fase0_final_wE070.csv)
%  (c,d) NRMSE de validacao (mediana e IQR) contra o SNR no CED e no
%      Silverbox (E5, fase2_outputs/e5/<caso>/e5_noise_summary.csv e e5_split.csv)
%  Saida: 27IFAC_greenSI_Thalita/V1/fig_analysis.pdf (8.4 cm de largura)
% ========================================================================

clear; close all
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
F2 = fullfile(scriptDir,'fase2_outputs');
outFile = fullfile(scriptDir,'..','27IFAC_greenSI_Thalita','V1','fig_analysis.pdf');
fs = 7;  W = 8.4;  H = 6.9;
cG5 = [0.227 0.431 0.647];  cG7 = [0.45 0.70 0.90];  cF = [0.80 0.40 0.10];
style = @(ax) set(ax,'FontName','Times New Roman','FontSize',fs,'LineWidth',0.5, ...
    'TickLabelInterpreter','latex','TickLength',[0.02 0.02]);

f = figure('Units','centimeters','Position',[2 2 W H],'Color','w');
t = tiledlayout(f,2,2,'TileSpacing','compact','Padding','tight');

% (a) busca exaustiva no DC motor
S = load(fullfile(F2,'dc','e2_exhaustive_all.mat'));
ok = isfinite(S.Cs) & isfinite(S.Ls) & S.ps >= 2;
G = readtable(fullfile(F2,'dc','e2_exhaustive_gap.csv'));
ax = nexttile(t);
semilogy(ax,S.Cs(ok),S.Ls(ok),'.','Color',[0.75 0.75 0.75],'MarkerSize',2);  hold(ax,'on');
[cf,ix] = sort(S.Cs(S.front));  lf = S.Ls(S.front);  lf = lf(ix);
stairs(ax,cf,lf,'-','Color','k','LineWidth',0.9);
mk = {'o','s','^'};  cc = {[0.30 0.62 0.30],cG5,cG7};  hG = gobjects(1,3);
for i = 1:3
    hG(i) = semilogy(ax,G.C(i),G.L(i),mk{i},'Color','k','MarkerFaceColor',cc{i},'MarkerSize',5,'LineWidth',0.5);
end
hold(ax,'off');  grid(ax,'on');  box(ax,'on');  style(ax);
xlim(ax,[0 2.1e4]);  ylim(ax,[1e-4 2e-2]);  ax.XAxis.Exponent = 4;
xlabel(ax,'Complexity $C$','Interpreter','latex','FontSize',fs);
ylabel(ax,'$L_p$','Interpreter','latex','FontSize',fs);
title(ax,'(a) DC motor, all subsets','FontWeight','normal','FontSize',fs,'Interpreter','latex');
lg = legend(ax,[ax.Children(end-1) hG],{'Pareto front','$w_E=0.3$','$w_E=0.5$','$w_E=0.7$'}, ...
    'Interpreter','latex','FontSize',fs-1,'Location','northeast','Box','on','Color','w','EdgeColor',[0.6 0.6 0.6]);
lg.ItemTokenSize = [10 6];

% (b) perda normalizada ao longo da ordem Green (w_E = 0.70)
src = {fullfile(scriptDir,'diag_fase0_outputs','dc','fase0_final_wE070.csv'), ...
       fullfile(scriptDir,'diag_fase0_outputs','ced','fase0_final_wE070.csv'), ...
       fullfile(scriptDir,'diag_fase0_outputs','fase0_final_wE070.csv')};
lab = {'DC motor','CED','Silverbox'};  col = {cF,cG5,[0.1 0.1 0.1]};  lst = {'-','--','-'};
ax = nexttile(t);  hold(ax,'on');
msz = [6 8 8];  lw = [0.8 1.0 1.0];  hL = gobjects(1,3);
for i = [2 3 1]                                    % a serie que termina primeiro fica por cima
    T = readtable(src{i});
    hL(i) = plot(ax,T.p,T.Ltil,lst{i},'Color',col{i},'LineWidth',lw(i),'Marker','.','MarkerSize',msz(i));
end
hold(ax,'off');  grid(ax,'on');  box(ax,'on');  style(ax);
xlim(ax,[2 40]);  ylim(ax,[0 1]);
xlabel(ax,'Number of terms $p$','Interpreter','latex','FontSize',fs);
ylabel(ax,'$\widetilde{L}_p$','Interpreter','latex','FontSize',fs);
title(ax,'(b) Normalised loss, $w_E=0.70$','FontWeight','normal','FontSize',fs,'Interpreter','latex');
lg = legend(ax,hL,lab,'Interpreter','latex','FontSize',fs-1,'Location','northeast','Box','off');
lg.ItemTokenSize = [10 6];

% (c, d) ruido
cases = {'ced','silverbox'};  ttl = {'(c) CED with noise','(d) Silverbox with noise'};
models = {'Green 0.5','Green 0.7','FROLS-max'};  mlab = {'$w_E=0.5$','$w_E=0.7$','ERR-only'};
mcol = {cG5,cG7,cF};  mmk = {'s','^','o'};  off = [-0.12 0 0.12];
for c = 1:2
    Sm = readtable(fullfile(F2,'e5',cases{c},'e5_noise_summary.csv'),'TextType','string');
    Sp = readtable(fullfile(F2,'e5',cases{c},'e5_split.csv'),'TextType','string');
    ax = nexttile(t);  hold(ax,'on');  h = gobjects(1,3);
    for m = 1:3
        r0 = Sp(Sp.model == models{m},:);  r0 = r0(1,:);            % particao do artigo, sem ruido
        r = Sm(Sm.model == models{m},:);  r = sortrows(r,'SNR_dB','descend');
        x = (1:4) + off(m);
        med = [r0.NRMSE; r.NRMSE_med];  lo = [r0.NRMSE; r.NRMSE_q1];  hi = [r0.NRMSE; r.NRMSE_q3];
        h(m) = errorbar(ax,x,med,med-lo,hi-med,mmk{m},'Color',mcol{m}, ...
            'MarkerFaceColor',mcol{m},'MarkerSize',3,'LineWidth',0.8,'CapSize',3);
    end
    for m = 1:3                                    % barras por cima dos marcadores
        r0 = Sp(Sp.model == models{m},:);  r0 = r0(1,:);
        r = Sm(Sm.model == models{m},:);  r = sortrows(r,'SNR_dB','descend');
        x = (1:4) + off(m);
        med = [r0.NRMSE; r.NRMSE_med];  lo = [r0.NRMSE; r.NRMSE_q1];  hi = [r0.NRMSE; r.NRMSE_q3];
        errorbar(ax,x,med,med-lo,hi-med,'LineStyle','none','Marker','none', ...
            'Color',mcol{m}*0.75,'LineWidth',0.8,'CapSize',3,'HandleVisibility','off');
    end
    hold(ax,'off');  grid(ax,'on');  box(ax,'on');  style(ax);
    xlim(ax,[0.5 4.5]);  xticks(ax,1:4);  xticklabels(ax,{'none','40','30','20'});
    ylim(ax,[0 0.4]);
    xlabel(ax,'SNR (dB)','Interpreter','latex','FontSize',fs);
    ylabel(ax,'NRMSE','Interpreter','latex','FontSize',fs);
    title(ax,ttl{c},'FontWeight','normal','FontSize',fs,'Interpreter','latex');
    if c == 2
        lg = legend(ax,h,mlab,'Interpreter','latex','FontSize',fs,'Box','off','NumColumns',3);
        lg.Layout.Tile = 'south';  lg.ItemTokenSize = [10 6];
    end
end

exportgraphics(f,outFile,'ContentType','vector');
fprintf('Figura salva em %s\n',outFile);
