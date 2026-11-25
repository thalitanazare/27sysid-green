function [order,errSelected] = term_order(A,Y,dTerm,maxTerms,wE,wC,B,rho,mode,tieRule)
% Mesmo algoritmo de green_term_order (artigo). Complexidade da estrutura
% tentativa: mode 'C' -> B*[(r-1) + rho*mult]; 'p' ou 'none' -> constante
% entre candidatos (normalizacao zera o termo de custo).
% tieRule (opcional): 'cost' (padrao do artigo revisado: maior ERR,
%   menor custo para ERRs equivalentes e menor indice original).
%   'err' preserva a regra anterior para os testes comparativos de E5.
    if nargin < 10, tieRule = 'cost'; end
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
        if numel(tie) > 1
            if strcmp(tieRule,'cost')
                % maior ERR, mas ERRs iguais a menos de arredondamento
                % (colineares) sao desempatados pelo menor custo e depois
                % pelo menor indice original
                eT = errC(tie);
                tie = tie(eT >= max(eT)-1e-10*max(max(eT),eps));
                cand = [dTerm(piv(available(tie))) piv(available(tie))'];
                [~,bt] = sortrows(cand,[1 2]);  lb = tie(bt(1));
            else
                [~,bt] = max(errC(tie)); lb = tie(bt);
            end
        else
            lb = tie(1);
        end
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
