function VaRESOut = compute_var_es(SimRet, PFweights, alpha)
%COMPUTE_VAR_ES Compute portfolio VaR and ES from simulated return paths
% for all models contained in SimRet.
%
%   VaRESOut = COMPUTE_VAR_ES(SimRet, PFweights, alpha) computes the
%   h-step ahead portfolio VaR and ES at confidence level alpha for all
%   models in SimRet by aggregating simulated asset returns using
%   PFweights and taking quantiles of the resulting path distribution.
%
%   INPUTS:
%       SimRet    : Struct, output from read_sim_results. Contains one
%                   field per model with HStepCumReturnSim (H x M x K x T)
%                   and a ModelNames cell array listing all loaded models.
%       PFweights : (K x 1) or (1 x K) vector of portfolio weights.
%                   Will be normalized to row vector internally.
%       alpha     : Scalar, significance level e.g. 0.025 for 97.5% VaR
%
%   OUTPUT:
%       VaRESOut : Struct containing:
%                  .VaR        - (T x J x H) VaR forecasts, negative values
%                  .ES         - (T x J x H) ES forecasts, negative values
%                  .alpha      - significance level of VaR and ES
%                  .H          - simulation horizon
%                  .M          - number of simulation paths used
%                  .Models     - (J x 1) cell array of model names
%                  .assets     - asset names
%                  .dates      - dates vector
%                  .PFweights  - (1 x K) portfolio weights used
%                  .WindLength - estimation window length
%
%   NOTES:
%       - VaR and ES are reported as negative numbers (left tail)
%       - Dimensions are consistent across all models since H, M, K, T
%         are read from the first model and assumed equal across all
%       - Actual returns are not stored here. Pass R separately to scoring
%         and evaluation functions
%       - Output not stored to disk since computation is fast


% Read out all model names considered
ModelNames = SimRet.ModelNames;
J          = max(size(ModelNames));

% Transform weight vector to row vector
[a, b] = size(PFweights);
if a > b
    PFweights = PFweights';
end

% Read out simulation results of the first model and the simulation setting
SimRetModel            = SimRet.(ModelNames{1});
HStepCumReturnSumModel = SimRetModel.HStepCumReturnSim;

% Read dimensions from first model (should be consistent across all models)
[H, M, K, T] = size(HStepCumReturnSumModel);

% Pre-allocate results matrices
VaR = NaN(T, J, H);
ES  = NaN(T, J, H);

% Compute VaR and ES
for j = 1:J

    % Read out simulation results of the j-th model and compute PF returns
    SimRetModel            = SimRet.(ModelNames{j});
    HStepCumReturnSumModel = SimRetModel.HStepCumReturnSim;
    HStepSimPFRet          = squeeze(sum(HStepCumReturnSumModel .* ...
                                     reshape(PFweights, 1, 1, K, 1), 3) );
 
    % h-step ahead VaR and ES for h = 1, ..., H
    for h = 1:H
        hStepSimPFRet = HStepSimPFRet(h, :, :);       
        sorted        = squeeze(sort(hStepSimPFRet))';  % sort ascending
        idx           = max(1, floor(alpha * M));       % left tail index
        VaR(:, j, h)  = sorted(:,idx);                  % VaR(alpha)
        ES(:, j, h)   = mean(sorted(:,1:idx), 2);       % ES
    end
end

% Pack VaRESOut
VaRESOut.VaR        = VaR;
VaRESOut.ES         = ES;
VaRESOut.alpha      = alpha;
VaRESOut.H          = SimRetModel.H;
VaRESOut.M          = M;
VaRESOut.Models     = ModelNames;
VaRESOut.assets     = SimRetModel.assets;
VaRESOut.dates      = SimRetModel.dates;
VaRESOut.PFweights  = PFweights;
VaRESOut.WindLength = SimRetModel.WindLength;

end