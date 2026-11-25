function [ut,yt,uv,yv,startIdx] = load_case(caseName,scriptDir)
    switch caseName
        case 'dc'
            Yraw = readmatrix(fullfile(scriptDir,'y_cc.csv'));
            Uraw = readmatrix(fullfile(scriptDir,'u_cc.csv'));
            Yraw = Yraw(:); Uraw = Uraw(:);
            idxD = 1:500:min(numel(Yraw),numel(Uraw));
            y = Yraw(idxD); u = Uraw(idxD);
            nTrain = numel(y)/2;
            ut = u(1:nTrain); yt = y(1:nTrain); uv = u; yv = y; startIdx = nTrain+1;
        case 'ced'
            D = load(fullfile(scriptDir,'CED_outputs','CED_experiment_2_data.mat'));
            ut = double(D.u_train(:)); yt = double(D.y_train(:));
            uv = double(D.u_test(:));  yv = double(D.y_test(:));
            startIdx = double(D.n_initial)+1;
        case 'silverbox'
            D = load(fullfile(scriptDir,'Silverbox_arrow_validation_outputs', ...
                'Silverbox_arrow_full_data.mat'));
            ut = double(D.u_train(:)); yt = double(D.y_train(:));
            uv = double(D.u_test(:));  yv = double(D.y_test(:));
            startIdx = double(D.n_initial)+1;
    end
end
