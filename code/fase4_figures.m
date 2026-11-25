%% ========================================================================
%  FASE 4 - figura das varreduras de pesos no tamanho final da coluna
%  %
%  Usa os resultados ja salvos (nao roda identificacao):
%    green_dc_motor_sysidentpy_page72_results.mat   (49 pesos interiores)
%    green_ced_exp2_sysidentpy_results.mat          (51 pesos, extremos removidos)
%    green_silverbox_arrow_sysidentpy_results.mat   (49 pesos interiores)
%    fase2_outputs/e3b/ced_models.json              (free run CED, w_E = 0.70)
%  Paineis (a)-(c): complexidade relativa ao SysIdentPy (eixo esquerdo) e
%  NRMSE de validacao (eixo direito) contra w_E. Tracejados = SysIdentPy.
%  Saida: 27IFAC_greenSI_Thalita/V1/fig_sweeps.pdf (8.4 cm de largura)
% ========================================================================

clear; close all
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
outFile = fullfile(scriptDir,'..','27IFAC_greenSI_Thalita','V1','fig_sweeps.pdf');

fs = 7;                      % fonte em pontos, no tamanho impresso
W = 8.4;  H = 6.6;           % cm
cC = [0.227 0.431 0.647];  cE = [0.80 0.40 0.10];  cData = [0.05 0.09 0.13];

files = {'green_dc_motor_sysidentpy_page72_results.mat', ...
         'green_ced_exp2_sysidentpy_results.mat', ...
         'green_silverbox_arrow_sysidentpy_results.mat'};
names = {'(a) DC motor','(b) CED','(c) Silverbox'};

f = figure('Units','centimeters','Position',[2 2 W H],'Color','w');
t = tiledlayout(f,2,2,'TileSpacing','compact','Padding','tight');
for i = 1:3
    S = load(fullfile(scriptDir,files{i}));  r = S.results;
    w = r.tradeoff.weights(:);  C = r.tradeoff.complexity(:);  E = r.tradeoff.NRMSE(:);
    keep = w > 0 & w < 1 & isfinite(E);              % so pesos interiores
    w = w(keep);  C = C(keep);  E = E(keep);
    refC = r.reference.complexity;  refE = r.reference.metrics.NRMSE;
    ax = nexttile(t);
    yyaxis(ax,'left');
    plot(ax,w,C/refC,'o','Color',cC,'MarkerSize',3,'MarkerFaceColor',cC);
    hold(ax,'on');  yline(ax,1,'--','Color',cC,'LineWidth',0.6);  hold(ax,'off');
    ylim(ax,[0 1.15*max([C/refC;1])]);
    ylabel(ax,'$C/C_{\mathrm{SysIdentPy}}$','Interpreter','latex','FontSize',fs);
    ax.YAxis(1).Color = cC;
    yyaxis(ax,'right');
    plot(ax,w,E,'s','Color',cE,'MarkerSize',3,'MarkerFaceColor',cE);
    hold(ax,'on');  yline(ax,refE,'--','Color',cE,'LineWidth',0.6);  hold(ax,'off');
    ylim(ax,[0 1.15*max([E;refE])]);
    ylabel(ax,'NRMSE','Interpreter','latex','FontSize',fs);
    ax.YAxis(2).Color = cE;
    xlim(ax,[0 1]);  xticks(ax,0:0.2:1);  box(ax,'on');  grid(ax,'on');
    set(ax,'FontName','Times New Roman','FontSize',fs,'LineWidth',0.5, ...
        'TickLabelInterpreter','latex','TickLength',[0.02 0.02]);
    xlabel(ax,'$w_E$','Interpreter','latex','FontSize',fs);
    title(ax,names{i},'FontWeight','normal','FontSize',fs,'Interpreter','latex');
end

% (d) free run do CED
D = jsondecode(fileread(fullfile(scriptDir,'fase2_outputs','e3b','ced_models.json')));
s0 = D.start0;  y = D.y(:);  k = 400 + (s0+1:numel(y))';
lab = {D.models.label};
yRef = D.models(strcmp(lab,'SysIdentPy')).yhat(:);
yG = D.models(strcmp(lab,'Green (0.70)')).yhat(:);
ax = nexttile(t);
plot(ax,k,y(s0+1:end),'-','Color',cData,'LineWidth',1.0);  hold(ax,'on');
plot(ax,k,yRef,'-o','Color',[0.753 0.341 0.275],'LineWidth',0.6,'MarkerSize',2.2,'MarkerIndices',1:5:numel(k));
plot(ax,k,yG,'-s','Color',cC,'LineWidth',0.6,'MarkerSize',2.2,'MarkerIndices',3:5:numel(k));
hold(ax,'off');  box(ax,'on');  grid(ax,'on');  xlim(ax,[k(1) k(end)]);  ylim(ax,[0 3.3]);
set(ax,'FontName','Times New Roman','FontSize',fs,'LineWidth',0.5, ...
    'TickLabelInterpreter','latex','TickLength',[0.02 0.02]);
xlabel(ax,'Sample','Interpreter','latex','FontSize',fs);
ylabel(ax,'Pulley speed','Interpreter','latex','FontSize',fs);
title(ax,'(d) CED free run','FontWeight','normal','FontSize',fs,'Interpreter','latex');
lg = legend(ax,{'Measured','SysIdentPy','$w_E=0.70$'},'Interpreter','latex', ...
    'FontSize',fs-1,'Location','northwest','Box','off');
lg.ItemTokenSize = [10 6];

exportgraphics(f,outFile,'ContentType','vector');
fprintf('Figura salva em %s\n',outFile);
