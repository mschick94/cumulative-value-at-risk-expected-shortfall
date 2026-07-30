function EstOut = read_marg_est_results(MarginalModels, assets)
%READ_MARG_EST_RESULTS Read in estimation results for all specified
% marginal model and asset combinations.
%
%   EstOut = READ_MARG_EST_RESULTS(MarginalModels, assets) reads in the
%   marginal estimation results for all specified variance models and
%   assets. Files are expected in Output/Estimation/Marginals/ following
%   the naming convention EstOut_<MarginalModel>_<assets>.mat
%
%   INPUTS:
%       MarginalModels : Cell array of marginal model names
%                        e.g. {'GARCH_norm', 'GARCH_t', 'GJR_GARCH_norm'}
%       assets         : Cell array of asset names
%                        e.g. {'AXP', 'BA'}
%
%   OUTPUT:
%       EstOut : Struct with one field per successfully loaded model,
%                e.g. EstOut.GARCH_norm, EstOut.GARCH_t.
%                Missing files are skipped with a warning.
%
%   NOTES:
%       - Only models for which a file exists are loaded
%       - Asset names must match exactly those used during estimation

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