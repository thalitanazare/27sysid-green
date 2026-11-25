function mdl = run_green(A0,Y0,terms,dTerm,cfg,wE,wC,B,rho,stage1,stage2,fixedOrder)
% stage1/stage2: 'C' (operacoes), 'p' (numero de termos) ou 'none' (so ERR).
    if isempty(fixedOrder)
        if strcmp(stage1,'none'), w1 = [1 0]; else, w1 = [wE wC]; end
        [order,errSel] = term_order(A0,Y0,dTerm,cfg.maxTerms,w1(1),w1(2),B,rho,stage1);
    else
        order = fixedOrder;
        errSel = order_err(A0,Y0,order);
    end
    k = (cfg.minTerms:cfg.maxTerms)';
    L = max(1-cumsum(errSel),0);  L = L(k);
    nA = k-1;  nM = cumsum(dTerm(order(:)));  nM = nM(k);
    if strcmp(stage2,'p'), cost = k; else, cost = B*(nA+rho*nM); end
    J = wE*normalise01(L)+wC*normalise01(cost);
    [~,i] = min(J);
    mdl.order = order; mdl.p = k(i); mdl.L = L(i);
    mdl.sel = terms(order(1:mdl.p),:);
    mdl.nAdd = mdl.p-1; mdl.nMult = nM(i);
    mdl.key = strjoin(arrayfun(@(j) term_to_string(terms(j,:)),order(1:mdl.p),'UniformOutput',false),' + ');
end
