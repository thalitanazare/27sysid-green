function nr = val_nrmse(sel,ut,yt,uv,yv,startIdx,cache)
    key = mat2str(sortrows(sel));
    if isKey(cache,key), nr = cache(key); return, end
    k = size(sel,1);
    ws = warning('off','MATLAB:nearlySingularMatrix');
    [mG,xG] = orthreg(sel,ut,yt,[k 0],0);
    warning(ws);
    [yh,ytrue] = simulate_free_run(mG,xG(:,1),uv,yv,startIdx);
    if any(~isfinite(yh)), nr = Inf; else, nr = norm(ytrue-yh)/norm(ytrue-mean(ytrue)); end
    cache(key) = nr; %#ok<NASGU>
end
