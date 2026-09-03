function VaRandES = pf_garch_benchmark(dist, R_pf, VaRandES, ...
    WindLength, ReestFreq, assets, dates, NumberWorkers, Hsim, Msim, ...
    varargin)
%PF_GARCH_BENCHMARK Estimate GARCH on equally weighted portfolio and add
% VaR and ES forecasts to the evaluation structure.
%
%   VaRandES = PF_GARCH_BENCHMARK(dist, R_pf, VaRandES, WindLength,
%   ReestFreq, assets, dates, NumberWorkers, Hsim, Msim) estimates a GARCH
%   model with distribution dist on the equally weighted portfolio return
%   R_pf, simulates H-step-ahead cumulative returns, and appends the
%   resulting VaR and ES forecasts to VaRandES for all alpha levels
%   stored in VaRandES.alpha.
%
%   INPUTS (required):
%       dist          : String, marginal distribution
%                       'norm', 't', 'skewt', 'laplace'
%       R_pf          : (T x 1) equally weighted portfolio returns
%       VaRandES      : Struct, output from simulate_all_var_es. Alpha
%                       levels are read from VaRandES.alpha
%       WindLength    : Scalar, estimation window length
%       ReestFreq     : Scalar, re-estimation frequency in days
%       assets        : Cell array of asset names
%       dates         : (T x 1) date vector
%       NumberWorkers : Scalar, number of parallel workers
%       Hsim          : Scalar, simulation horizon
%       Msim          : Scalar, number of simulation paths
%
%   INPUTS (optional name-value):
%       'EmpiricalPits' : Logical, use empirical PITs (default: false)
%       'GJR'           : Logical, use GJR-GARCH instead of GARCH
%                         (default: false)
%
%   OUTPUT:
%       VaRandES : Updated struct with PF benchmark VaR and ES appended
%                  for all alpha levels. Compatible with score_fz and
%                  var_uc_test.

% Name-value inputs
p = inputParser;
addParameter(p, 'EmpiricalPits', false);
addParameter(p, 'GJR',           false);
parse(p, varargin{:});
EmpiricalPits = p.Results.EmpiricalPits;
GJR           = p.Results.GJR;

% Model name for output struct
if GJR
    model_str = sprintf('EqualWeightedPF_GJRGARCH_%s', dist);
else
    model_str = sprintf('EqualWeightedPF_GARCH_%s', dist);
end
if EmpiricalPits
    model_str = [model_str '_empirical'];
end

% Estimate marginals
if GJR
    PF_marginal = estimate_gjr_garch(R_pf, 'dist', dist, ...
        'WindLength', WindLength, 'ReestFreq', ReestFreq, ...
        'assets', assets, 'NumWorkers', NumberWorkers, ...
        'dates', dates, 'Portfolio', 'EqualWeighted', 'SaveDisk', false);
else
    PF_marginal = estimate_garch(R_pf, 'dist', dist, ...
        'WindLength', WindLength, 'ReestFreq', ReestFreq, ...
        'assets', assets, 'NumWorkers', NumberWorkers, ...
        'dates', dates, 'Portfolio', 'EqualWeighted', 'SaveDisk', false);
end

% K=1 copula needed for simulation
PF_copula = estimate_copula(PF_marginal, 'empirical_pits', ...
                            EmpiricalPits, 'SaveDisk', false);

% Simulate for all alpha levels
alpha      = VaRandES.alpha;  
PFVaRandES = simulate_var_es(PF_copula, R_pf, 1, alpha, ...
                              'H', Hsim, 'M', Msim, ...
                              'NumWorkers', NumberWorkers);

% Add to evaluation structure
VaRandES.VaR    = cat(2, VaRandES.VaR, PFVaRandES.VaR);
VaRandES.ES     = cat(2, VaRandES.ES,  PFVaRandES.ES);
VaRandES.Models = [VaRandES.Models, {model_str}];


end