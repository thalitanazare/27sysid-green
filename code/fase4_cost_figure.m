%% ========================================================================
%  FASE 4 - figura de custo medido e ciclo de vida
%  %
%  (a) tempo medido por amostra x complexidade C (log-log), 18 modelos
%      polinomiais (E3b, perfil direto), ajuste afim e redes neurais (E6c).
%  (b) CO2e acumulado x tempo de operacao na taxa do Silverbox
%      (identificacao/treino + avaliacoes), taxa da Tabela 1.
%  Entradas: fase2_outputs/e3b/e3b_timing.csv, fase2_outputs/e6c/e6c_timing.csv,
%            fase2_outputs/e6c/e6c_lifecycle.csv
%  Saida: 27IFAC_greenSI_Thalita/V1/fig_cost.pdf (8.4 cm de largura)
% ========================================================================

clear; close all
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
F2 = fullfile(scriptDir,'fase2_outputs');
outFile = fullfile(scriptDir,'..','27IFAC_greenSI_Thalita','V1','fig_cost.pdf');
fs = 7;  W = 8.4;  H = 5.0;
cP = [0.227 0.431 0.647];  cN = [0.80 0.40 0.10];
RATE = 121.81/60;                                    % mg CO2e por segundo
fsSB = 1e7/2^14;                                     % Hz

P = readtable(fullfile(F2,'e3b','e3b_timing.csv'),'TextType','string');
P = P(P.profile == "direto",:);
N = readtable(fullfile(F2,'e6c','e6c_timing.csv'),'TextType','string');
N = N(N.profile == "direto",:);
L = readtable(fullfile(F2,'e6c','e6c_lifecycle.csv'),'TextType','string');

f = figure('Units','centimeters','Position',[2 2 W H],'Color','w');
t = tiledlayout(f,1,2,'TileSpacing','compact','Padding','tight');

% (a) tempo x C
ax = nexttile(t);
cfit = polyfit(P.C,P.ns_median,1);
cc = logspace(log10(min(P.C)*0.8),log10(max(N.C)*1.3),100);
loglog(ax,P.C,P.ns_median,'o','Color',cP,'MarkerFaceColor',cP,'MarkerSize',3.5);  hold(ax,'on');
loglog(ax,cc,polyval(cfit,cc),'--','Color',cP,'LineWidth',0.6);
mk = struct('mlp','^','lstm','d','subnet','v');
fam = ["mlp","lstm","subnet"];  famLab = {'MLP','LSTM','SUBNET'};
h = gobjects(1,3);
for j = 1:3
    s = N(N.model == fam(j),:);
    h(j) = loglog(ax,s.C,s.ns_median,mk.(fam(j)),'Color',cN,'MarkerFaceColor',cN,'MarkerSize',4.5);
end
hold(ax,'off');  grid(ax,'on');  box(ax,'on');
xlim(ax,[5e2 3e7]);  ylim(ax,[1 3e4]);  xticks(ax,10.^(3:7));  yticks(ax,10.^(0:4));
set(ax,'FontName','Times New Roman','FontSize',fs,'LineWidth',0.5, ...
    'TickLabelInterpreter','latex','TickLength',[0.02 0.02],'XMinorGrid','off','YMinorGrid','off');
xlabel(ax,'Complexity $C$','Interpreter','latex','FontSize',fs);
ylabel(ax,'Measured time (ns/sample)','Interpreter','latex','FontSize',fs);
title(ax,'(a)','FontWeight','normal','FontSize',fs,'Interpreter','latex');
lg = legend(ax,[ax.Children(end) ax.Children(end-1) h],[{'Polynomial','Affine fit'} famLab], ...
    'Interpreter','latex','FontSize',fs-1,'Location','northwest','Box','on','Color','w','EdgeColor',[0.6 0.6 0.6]);
lg.ItemTokenSize = [10 6];

% (b) CO2e acumulado no Silverbox
ax = nexttile(t);
S = L(string(L{:,1}) == "silverbox",:);
show = ["Green (0.70)","Green (1.00)","MLP","LSTM","SUBNET"];
lab = {'Green $w_E=0.70$','ERR-only','MLP','LSTM','SUBNET'};
sty = {'-','--','-','--','-.'};
col = {cP,[0.45 0.70 0.90],[0.93 0.60 0.20],[0.80 0.40 0.10],[0.50 0.18 0.05]};
days = logspace(-3,3,200);
hold(ax,'on');
for j = 1:numel(show)
    r = S(S.model == show(j),:);
    g = (r.ident_s + days*86400*fsSB*r.ns_sample*1e-9)*RATE/1000;   % g CO2e
    plot(ax,days,g,sty{j},'Color',col{j},'LineWidth',1.1);
end
xline(ax,365.25,':','Color',[0.4 0.4 0.4],'LineWidth',0.5,'HandleVisibility','off');
hold(ax,'off');  set(ax,'XScale','log','YScale','log');  grid(ax,'on');  box(ax,'on');
xlim(ax,[1e-3 1e3]);  ylim(ax,[0.05 3e3]);  xticks(ax,10.^(-2:2:2));  yticks(ax,10.^(-1:1:3));  ax.XTickLabelRotation = 0;
set(ax,'FontName','Times New Roman','FontSize',fs,'LineWidth',0.5, ...
    'TickLabelInterpreter','latex','TickLength',[0.02 0.02],'XMinorGrid','off','YMinorGrid','off');
xlabel(ax,'Operation time (days)','Interpreter','latex','FontSize',fs);
ylabel(ax,'Cumulative CO$_2$e (g)','Interpreter','latex','FontSize',fs);
title(ax,'(b) Silverbox, 610 Hz','FontWeight','normal','FontSize',fs,'Interpreter','latex');
lg = legend(ax,lab,'Interpreter','latex','FontSize',fs-1,'Box','off','NumColumns',3);
lg.Layout.Tile = 'south';
text(ax,230,3,'1 year','FontSize',fs-1,'Interpreter','latex','Rotation',90,'HorizontalAlignment','center');
lg.ItemTokenSize = [12 6];

exportgraphics(f,outFile,'ContentType','vector');
fprintf('Figura salva em %s\n',outFile);
fprintf('Ajuste afim: t = %.4g + %.4g C (ns)\n',cfit(2),cfit(1));
