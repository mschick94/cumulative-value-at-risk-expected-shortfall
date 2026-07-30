function EstOut = read_marg_est_results(MarginalModels, assets)
%READ_MARGINAL_EST_RESULTS Read in estimation results of univariate 
% variance models
%
%   EstOut = READ_MARGINAL_EST_RESULTS(MarginalModels,assets) reads in the
%   estimation resutls of the marginal distributions using variance models
%   and assets as specified.  UPDATE/IMPROVE HEADER!!!

    assets_str = strjoin(assets, '_');
    EstOut     = struct();
    for m = 1:length(MarginalModels)
        filename = sprintf('Output/Estimation/Marginals/EstOut_%s_%s.mat', ...
                           MarginalModels{m}, assets_str);
        if exist(filename, 'file')
            tmp                    = load(filename);
            EstOut.(MarginalModels{m}) = tmp.EstOut;
        else
            warning('read_marg_est_results: file not found: %s', filename);
        end
    end
end

% function EstOut = read_marg_est_results(models,assets)
% %READ_MARGINAL_EST_RESULTS Read in estimation results of univariate 
% % variance models
% %
% %   EstOut = READ_MARGINAL_EST_RESULTS(models,assets) reads in the
% %   estimation resutls of the marginal distributions using variance models
% %   and assets as specified. 
% 
% EstOut = struct();
% for m = 1:length(models)
%     filename = sprintf('Output/Estimation/Marginals/EstOut_%s_%s.mat', ...
%                        models{m}, strjoin(assets, '_'));
%     tmp = load(filename);
%     EstOut.(models{m}) = tmp.EstOut;
% end
% 
% end