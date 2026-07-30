function EstOut = read_copula_est_results(MarginalModels, CopulaModels, assets)
%READ_COPULA_EST_RESULTS Read in estimation results for all combinations
% of marginal and copula model specifications.
%
%   EstOut = READ_COPULA_EST_RESULTS(MarginalModels, CopulaModels, assets)
%   reads in the copula estimation results for all combinations of marginal
%   distributions and copula models for the specified assets. Files are
%   expected in Output/Estimation/Copula/ following the naming convention
%   EstOut_<MarginalModel>_<CopulaModel>_<assets>.mat
%
%   INPUTS:
%       MarginalModels : Cell array of marginal model names
%                        e.g. {'GARCH_norm', 'GARCH_t', 'GJR_GARCH_norm'}
%       CopulaModels   : Cell array of copula model names
%                        e.g. {'CCC_norm', 'CCC_t', 'DCC_norm', 'DCC_t'}
%       assets         : Cell array of asset names
%                        e.g. {'AXP', 'BA'}
%
%   OUTPUT:
%       EstOut : Struct with one field per successfully loaded model
%                combination, e.g. EstOut.GARCH_norm_CCC_norm.
%                Missing files are skipped with a warning.
%
%   NOTES:
%       - Only combinations for which a file exists are loaded
%       - Asset names must match exactly those used during estimation

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