function C = Ckar(nAdd,nMult)
% Complexidade do artigo (B = 64, alpha = 1.585), para comparar tudo na mesma escala.
    C = nAdd*64+nMult*64^1.585;
end
