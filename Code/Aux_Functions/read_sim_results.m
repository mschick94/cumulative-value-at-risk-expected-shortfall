function SimOut = read_sim_results(MarginalModels, CopulaModels, assets)
%READ_SIM_RESULTS Read in simulation results for all combinations of
% marginal and copula model specifications.
%
%   SimOut = READ_SIM_RESULTS(MarginalModels, CopulaModels, assets)
%   reads in the simulation results for all combinations of marginal
%   distributions, copula models, and assets as specified.
%
%   INPUTS:
%       MarginalModels : Cell array of marginal model names
%                        e.g. {'GARCH_norm', 'GARCH_t'}
%       CopulaModels   : Cell array of copula model names
%                        e.g. {'CCC_norm', 'CCC_t', 'DCC_norm'}
%       assets         : Cell array of asset names
%                        e.g. {'AXP', 'BA'}
%
%   OUTPUT:
%       SimOut : Struct with one field per model combination, e.g.
%                SimOut.GARCH_norm_CCC_norm

assets_str   = strjoin(assets, '_');
SimOut       = struct();
ModelsString = cell(length(MarginalModels) * length(CopulaModels), 1);

j = 1;
for m = 1:length(MarginalModels)
    for c = 1:length(CopulaModels)
        model_name = sprintf('%s_%s', MarginalModels{m}, CopulaModels{c});
        filename   = sprintf('Output/Simulation/SimOut_%s_%s.mat', ...
                              model_name, assets_str);
        if exist(filename, 'file')
            tmp                 = load(filename);
            SimOut.(model_name) = tmp.SimOut;
            ModelsString{j}     = model_name;
            j = j + 1;
        else
            warning('read_sim_results: file not found: %s — skipping', filename);
        end
    end
end

SimOut.ModelNames = ModelsString;

end