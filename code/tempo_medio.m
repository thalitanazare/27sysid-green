clear;
clc;

N = 1e7;                  % Multiplicacoes por bloco
repeticoes = 5;           % Numero de medicoes
tempos = zeros(repeticoes, 1);

a = 1.00000001;           % double: 64 bits

for r = 1:(repeticoes + 1)
    x = 1.0;

    tic;
    for k = 1:N
        x = x * a;
    end
    tempoBloco = toc;

    % Descartar a primeira execucao (aquecimento)
    if r > 1
        tempos(r - 1) = tempoBloco / N;
    end
end

tempoMedio = mean(tempos);

fprintf('Tempo medio: %.6e segundos/multiplicacao\n', tempoMedio);
fprintf('Tempo medio: %.3f nanossegundos/multiplicacao\n', tempoMedio * 1e9);
fprintf('Resultado final: %.8f\n', x);

%% %%
% depois de rodar esse code de cima eu tive o resultado: 
%Tempo medio: 1.072312e-09 segundos/unidade de complexidade (multiplicacao)
%entao agora: 
tempoMult = 1.072312e-9;       % segundos/multiplicacao
emissaoPorSegundo = 121.81/60; % mgCO2e/s (tirado do green calculator)

fator = (tempoMult / 64^1.585) * emissaoPorSegundo

%deu fator = 2.98577859424619e-12 to usando 3e-12
% ai usa essa regrina na complexidade total de cada modelo
% carbono_mg = complexidadeTotal * fator;