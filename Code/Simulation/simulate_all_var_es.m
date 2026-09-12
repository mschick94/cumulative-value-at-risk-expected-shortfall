function VaRESOut = simulate_all_var_es(EstOut, R, PFweights, ...
                                        alpha, varargin)
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
%       alpha     : (1xP) vector of significance levels
%                   e.g. [0.01, 0.025] for 99% and 97.5% VaR
%
%   INPUTS (optional name-value):
%       'RV'         : (Tx(K*(K+1)/2)) matrix of realized variances in vech
%                      format. Required when EstOut contains HEAVY models.
%                      Diagonal elements (realized variances per asset) are
%                      extracted internally. Pass [] for GARCH/GJR models
%                      (default: [])
%       'H'          : Scalar, simulation horizon in days (default: 10)
%       'M'          : Scalar, number of simulation paths (default: 1000)
%       'NumWorkers' : Scalar, number of parallel workers (default: 1)
%       'SetSeed'    : Logical, set seed for reproducibility (default: true)
%       'OutputName' : String, identifier for output filename
%                      (default: 'EquallyWeighted')
%
%   OUTPUT:
%       VaRESOut : Struct containing:
%                  .VaR        - (T x J x 1 x P) VaR forecasts, negative values
%                  .ES         - (T x J x 1 x P) ES forecasts, negative values
%                  .PITs       - (T x J) empirical PITs from simulated
%                                distribution. NaN for t > T-H+1.
%                  .alpha      - (1xP) significance levels
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
%       - Model names read directly from EstOut field names; no manual
%         specification needed
%       - Only h=H stored; VaR and ES collapsed over horizon dimension
%       - PITs computed as empirical CDF at each t for Bernoulli-corrected
%         ES test (Du & Escanciano, 2017). Requires ActualHStepPFRet
%         computed internally from R and PFweights
%       - PF benchmark models should be added separately via cat after
%         calling this function
%       - Output saved to Output/VaRandES/ — existing file raises error
%         forcing manual deletion to avoid accidental overwrite
%       - Supply NumWorkers > 1 to enable parallel simulation over t

% Name-value inputs
p = inputParser;
p.KeepUnmatched = true;
addParameter(p, 'RV',         []);
addParameter(p, 'H',          []);
addParameter(p, 'OutputName', 'EquallyWeighted');
addParameter(p, 'ResumeFrom', []); 
parse(p, varargin{:});

H_aux      = p.Results.H;
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


% Construct actual H-step ahead cumulative PF returns for PITs computation
ActualPFRet    = R * PFweights;   % (T x 1)
CumSum         = cumsum(ActualPFRet);
T              = size(ActualPFRet, 1);
ActualHStepPFRet = NaN(T, 1);
ActualHStepPFRet(1:T-H_aux+1) = CumSum(H_aux:T) - [0; CumSum(1:T-H_aux)];


% Model specs
model_names = fieldnames(EstOut);
J           = length(model_names);
j           = 0;
m_start     = 1;

% If crashed uncomment the following three lines and run again
load('Output\VaRandES\checkpoint.mat')
j = length(VaRESOut.Models); 
m_start = j + 1;

p = length(alpha);
for m = m_start:J
    model_name = model_names{m};
    result = simulate_var_es(EstOut.(model_name), R, PFweights, alpha, ...
                             'ActualHStepPFRet', ActualHStepPFRet, ...
                             varargin{:});
    j = j + 1;
    if ~exist('VaRESOut', 'var')
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
    VaRESOut.Models{j, 1}    = model_name;

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