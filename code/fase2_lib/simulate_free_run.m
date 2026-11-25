function [yhat,ytrue] = simulate_free_run(model,theta,u,y,startIndex)
    u = u(:); y = y(:);
    positiveCodes = model(model > 0);
    lag = max([0;mod(positiveCodes(:),1000)]);
    firstIdx = startIndex-lag;
    uSeg = u(firstIdx:end);  y0 = y(firstIdx:startIndex-1);
    n = length(uSeg);  degree = size(model,2);  ySim = zeros(n,1);
    if lag > 0, ySim(1:lag) = y0; end
    for nn = lag+1:n
        total = 0;
        for i = 1:size(model,1)
            value = theta(i);
            for j = 1:degree
                code = model(i,j);
                if code == 0 || code == -1, continue, end
                lagValue = mod(code,1000);
                switch floor(code/1000)
                    case 1, value = value*ySim(nn-lagValue);
                    case 2, value = value*uSeg(nn-lagValue);
                    case 3, value = 0;
                end
            end
            total = total+value;
        end
        ySim(nn) = total;
        if ~isfinite(total) || abs(total) > 1e6, ySim(nn:end) = Inf; break, end
    end
    yhat = ySim(lag+1:end);  ytrue = y(startIndex:end);
end
