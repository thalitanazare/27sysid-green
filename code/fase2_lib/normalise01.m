function z = normalise01(x)
    x = x(:);  xmin = min(x);  xmax = max(x);
    if abs(xmax-xmin) <= eps(max(abs(x))+1), z = zeros(size(x));
    else, z = (x-xmin)/(xmax-xmin); end
end
