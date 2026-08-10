function [VaRandES_0010, VaRandES_0025] = pf_garch_benchmark(dist, ...
    R_pf, VaRandES_0010, VaRandES_0025, WindLength, ReestFreq, ...
    assets, dates, NumberWorkers, Hsim, Msim, varargin)
%PF_GARCH_BENCHMARK Estimate GARCH on equally weighted portfolio and add
% VaR and ES to evaluation structures.
%
%   [VaRandES_0010, VaRandES_0025] = PF_GARCH_BENCHMARK(dist, R_pf,
%   VaRandES_0010, VaRandES_0025, WindLength, ReestFreq, assets, dates,
%   NumberWorkers, Hsim, Msim) estimates a GARCH model with distribution
%   dist on the equally weighted portfolio return R_pf and adds the
%   resulting VaR and ES forecasts to VaRandES_0010 and VaRandES_0025.
%
%   INPUTS (required):
%       dist          : String, marginal distribution
%                       'norm', 't', 'skewt', 'laplace'
%       R_pf          : (T x 1) equally weighted portfolio returns
%       VaRandES_0010 : Struct, output from compute_var_es at alpha=0.01
%       VaRandES_0025 : Struct, output from compute_var_es at alpha=0.025
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
%       VaRandES_0010 : Updated struct with PF benchmark VaR and ES added
%       VaRandES_0025 : Updated struct with PF benchmark VaR and ES added

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

% Simulate return distribution
PF_sim             = simulate_return(PF_copula, R_pf, 'H', Hsim, ...
                                     'M', Msim, 'NumWorkers', ...
                                     NumberWorkers, 'SaveDisk', false);
PF_sim.(model_str) = PF_sim;
PF_sim.ModelNames  = {model_str};

% Compute VaR and ES at alpha = 0.01 and add to evaluation structure
PFVaRandES = compute_var_es(PF_sim, 1, 0.01);
VaRandES_0010.VaR    = cat(2, VaRandES_0010.VaR, PFVaRandES.VaR);
VaRandES_0010.ES     = cat(2, VaRandES_0010.ES,  PFVaRandES.ES);
VaRandES_0010.Models = [VaRandES_0010.Models; {model_str}];

% Compute VaR and ES at alpha = 0.025 and add to evaluation structure
PFVaRandES = compute_var_es(PF_sim, 1, 0.025);
VaRandES_0025.VaR    = cat(2, VaRandES_0025.VaR, PFVaRandES.VaR);
VaRandES_0025.ES     = cat(2, VaRandES_0025.ES,  PFVaRandES.ES);
VaRandES_0025.Models = [VaRandES_0025.Models; {model_str}];

end