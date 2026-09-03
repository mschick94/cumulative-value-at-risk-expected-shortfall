function VaRESOut = simulate_all_var_es(EstOut, R, PFweights, alpha, ...
                                        varargin)
%SIMULATE_ALL_VAR_ES Compute portfolio VaR and ES for all models contained
% in EstOut by calling simulate_var_es for each model specification.
%
%   VaRESOut = SIMULATE_ALL_VAR_ES(EstOut, R, PFweights, alpha) loops over
%   all model specifications stored in EstOut, simulates H-step-ahead
%   cumulative returns for each, and collects VaR and ES forecasts into
%   a single output struct compatible with score_fz and var_uc_test.
%
%   INPUTS (required):
%       EstOut    : Struct, output from read_copula_est_results. Contains
%                   one field per model combination, 
%                   e.g., EstOut.GARCH_norm_CCC_norm
%       R         : (TxK) matrix of observed returns
%       PFweights : (Kx1) or (1xK) vector of portfolio weights
%       alpha     : p-vector, significance levels e.g. 0.025 for 97.5% VaR
%
%   INPUTS (optional name-value):
%       'H'          : Scalar, simulation horizon in days (default: 10)
%       'M'          : Scalar, number of simulation paths (default: 1000)
%       'NumWorkers' : Scalar, number of parallel workers (default: 1)
%       'SetSeed'    : Logical, set seed for reproducibility 
%                      (default: true)
%
%   OUTPUT:
%       VaRESOut : Struct containing:
%                  .VaR        - (T x J x H) VaR forecasts, negative values
%                  .ES         - (T x J x H) ES forecasts, negative values
%                  .alpha      - significance level
%                  .H          - simulation horizon
%                  .M          - number of simulation paths
%                  .Models     - (J x 1) cell array of model names
%                  .assets     - asset names
%                  .dates      - dates vector
%                  .PFweights  - (1xK) portfolio weights used
%                  .WindLength - estimation window length
%                  .ReestFreq  - re-estimation frequency
%
%   NOTES:
%       - Model names are read directly from EstOut field names
%       - Missing or skipped combinations are handled automatically since
%         read_copula_est_results only loads existing files
%       - Output is directly compatible with score_fz and var_uc_test
%       - PF benchmark models should be added separately via cat after
%         calling this function

% In inputParser
p = inputParser;
p.KeepUnmatched = true;
addParameter(p, 'OutputName', 'EquallyWeighted');
parse(p, varargin{:});
OutputName = p.Results.OutputName;

% Build filename and check existence to avoid accidentally overwriting it
first_model = fieldnames(EstOut);
assets_str  = strjoin(EstOut.(first_model{1}).assets, '_');
filename   = sprintf('Output/VaRandES/VaRandES_%s_%s.mat', ...
                     OutputName, assets_str);

if exist(filename, 'file')
    error('mycode:fileExists', ...
          ['simulate_all_var_es: output file already exists — ' ...
           'delete manually to overwrite:\n%s'], filename);
end


% Model specs
model_names = fieldnames(EstOut);
J           = length(model_names);
first       = true;
j           = 0;

p = length(alpha);
for m = 1:J
    model_name = model_names{m};
    result = simulate_var_es(EstOut.(model_name), R, PFweights, alpha, ...
                             varargin{:});
    j = j + 1;
    if first
        T   = size(result.VaR, 1);
        H_n = size(result.VaR, 3);
        VaRESOut.VaR        = NaN(T, J, H_n, p);
        VaRESOut.ES         = NaN(T, J, H_n, p);
        VaRESOut.alpha      = alpha;
        VaRESOut.H          = result.H;
        VaRESOut.M          = result.M;
        VaRESOut.assets     = result.assets;
        VaRESOut.dates      = result.dates;
        VaRESOut.PFweights  = PFweights;
        VaRESOut.WindLength = result.WindLength;
        VaRESOut.ReestFreq  = result.ReestFreq;
        VaRESOut.Models     = {};
        first               = false;
    end
    VaRESOut.VaR(:, j, :, :) = result.VaR(:, 1, :, :);
    VaRESOut.ES(:, j, :, :)  = result.ES(:, 1, :, :);
    VaRESOut.Models{j}    = model_name;

    save('Output/VaRandES/checkpoint.mat', 'VaRESOut');
    fprintf('Model %d/%d done: %s\n', j, J, model_name);
end


% Save final output
save(filename, 'VaRESOut');
fprintf('Output saved to %s\n', filename);

% Delete checkpoint only if save successful
if exist('Output/VaRandES/checkpoint.mat', 'file')
    delete('Output/VaRandES/checkpoint.mat');
    fprintf('Checkpoint deleted\n');
end


end