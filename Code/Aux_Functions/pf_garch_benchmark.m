function VaRandES = pf_garch_benchmark(dist, R, PFweights, VaRandES, ...
    WindLength, ReestFreq, assets, dates, NumberWorkers, Hsim, Msim, varargin)
%PF_GARCH_BENCHMARK Estimate GARCH on equally weighted portfolio and add
% VaR, ES, and PITs to the evaluation structure.
%
%   VaRandES = PF_GARCH_BENCHMARK(dist, R, PFweights, VaRandES, WindLength,
%   ReestFreq, assets, dates, NumberWorkers, Hsim, Msim) estimates a GARCH
%   model with distribution dist on the equally weighted portfolio return,
%   simulates H-step-ahead cumulative returns, and appends the resulting
%   VaR, ES and PITs to VaRandES for all alpha levels in VaRandES.alpha.
%
%   INPUTS (required):
%       dist          : String, marginal distribution
%                       'norm', 't', 'skewt', 'laplace'
%       R             : (TxK) matrix of observed returns
%       PFweights     : (Kx1) or (1xK) vector of portfolio weights
%       VaRandES      : Struct, output from simulate_all_var_es. Alpha
%                       levels read from VaRandES.alpha
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
%       VaRandES : Updated struct with PF benchmark VaR, ES and empirical 
%                  PITs appended for all alpha levels. Compatible with
%                  score_fz and var_uc_test.
%
%   NOTES:
%       - R_pf = R * PFweights computed internally
%       - ActualHStepPFRet computed internally for PIT computation
%       - simulate_var_es called once with all alpha levels simultaneously
%       - Alpha levels inherited from VaRandES.alpha
%       - VaR, ES and PITs appended along J dimension via cat

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

% Construct actual H-step ahead cumulative PF returns for PITs computation
R_pf             = R * PFweights;   % (T x 1)
CumSum           = cumsum(R_pf);
T                = size(R_pf, 1);
ActualHStepPFRet = NaN(T, 1);
ActualHStepPFRet(1:T-Hsim+1) = CumSum(Hsim:T) - [0; CumSum(1:T-Hsim)];

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
                             'ActualHStepPFRet', ActualHStepPFRet,...
                             'H', Hsim, 'M', Msim, ...
                             'NumWorkers', NumberWorkers);

% Add to evaluation structure
VaRandES.VaR     = cat(2, VaRandES.VaR,     PFVaRandES.VaR);
VaRandES.ES      = cat(2, VaRandES.ES,      PFVaRandES.ES);
VaRandES.EmpPITs = cat(2, VaRandES.EmpPITs, PFVaRandES.PITs);
VaRandES.Models  = [VaRandES.Models; {model_str}];


end