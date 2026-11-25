function txt = term_to_string(termCode)
    factors = {};
    for j = 1:length(termCode)
        code = termCode(j);
        if code == 0 || code == -1, continue, end
        lagValue = mod(code,1000);
        switch floor(code/1000)
            case 1, factors{end+1} = sprintf('y(k-%d)',lagValue); %#ok<AGROW>
            case 2, factors{end+1} = sprintf('u(k-%d)',lagValue); %#ok<AGROW>
        end
    end
    if isempty(factors), txt = '1'; else, txt = strjoin(factors,'*'); end
end
