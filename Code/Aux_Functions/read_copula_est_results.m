function EstOut = read_copula_est_results(MarginalModels, ...
    CopulaModels, assets)
%READ_COPULAL_EST_RESULTS Read in estimation results of univariate 
% variance models and copula
%
%   EstOut = READ_COPULA_EST_RESULTS(MarginalModels,CopulaModels,assets) 
%   reads in the estimation resutls of the marginal distributions and 
%   copula dependence structure using variance models, assets, copula, and 
%   correlation matrix model as specified. UPDATE/IMPROVE HEADER!!!

    assets_str = strjoin(assets, '_');
    EstOut     = struct();
    for m = 1:length(MarginalModels)
        for c = 1:length(CopulaModels)
            model_name = sprintf('%s_%s', MarginalModels{m}, CopulaModels{c});
            filename   = sprintf('Output/Estimation/Copula/EstOut_%s_%s.mat', ...
                                  model_name, assets_str);
            if exist(filename, 'file')
                tmp                    = load(filename);
                EstOut.(model_name)    = tmp.EstOut;
            else
                warning('read_cop_est_results: file not found: %s — skipping', filename);
            end
        end
    end

end