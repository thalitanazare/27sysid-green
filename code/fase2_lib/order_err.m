function errSel = order_err(A,Y,order)
% ERR de cada termo de uma ordem fixa (ortogonalizacao sequencial).
    A = A(:,order);  [m,n] = size(A);  sqy = Y'*Y;  errSel = zeros(n,1);
    for r = 1:n
        aj = A(r:m,r)'*A(r:m,r);  bj = A(r:m,r)'*Y(r:m);
        if aj > eps, errSel(r) = (bj^2)/(aj*sqy); end
        v = house(A(r:m,r));
        A(r:m,r:n) = rowhouse(A(r:m,r:n),v);
        Y(r:m) = rowhouse(Y(r:m),v);
    end
end
