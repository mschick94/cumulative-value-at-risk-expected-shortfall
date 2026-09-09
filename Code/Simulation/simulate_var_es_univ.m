function [VaR, ES] = simulate_var_es_univ(CopulaEst, Returns, alpha, varargin)
%SIMULATE_VAR_ES_UNIV Fast simulation of H-step-ahead VaR and ES for
% univariate GARCH models (K=1) with normal or t innovations.
%
%   [VaR, ES] = SIMULATE_VAR_ES_UNIV(CopulaEst, Returns, alpha) simulates
%   H-step-ahead cumulative returns and computes VaR and ES for K=1.
%
%   INPUTS (required):
%       CopulaEst : Struct, output from estimate_copula (K=1 only)
%       Returns   : (Tx1) vector of observed returns
%       alpha     : (1xP) vector of significance levels
%
%   INPUTS (optional name-value):
%       'H' : Scalar, simulation horizon (default: 10)
%       'M' : Scalar, number of simulation paths (default: 25000)
%
%   OUTPUT:
%       VaR : (T x P) VaR forecasts, negative values
%       ES  : (T x P) ES forecasts, negative values

% Name-value inputs
p = inputParser;
addParameter(p, 'H', 10);
addParameter(p, 'M', 25000);
parse(p, varargin{:});
H = p.Results.H;
M = p.Results.M;
P = length(alpha);

% Read out marginal setting
margDist  = CopulaEst.MargDist;
GARCHpars = CopulaEst.GARCHpars;
mu        = CopulaEst.mu;
H_last    = CopulaEst.H_last;
W         = CopulaEst.WindLength;
T         = size(Returns, 1);
t_start   = W + 1;

% Distribution parameters
switch margDist
    case 't'
        margNu = CopulaEst.nu;
    otherwise
        margNu = [];
end

% GARCHspec
if contains(CopulaEst.model, 'GJR')
    GARCHspec = 'gjr';
else
    GARCHspec = 'garch';
end

% Pre-draw standard normal random numbers
rng(1)
StdN = randn(H, M);   % (H x M) directly — no copula needed for K=1

% Pre-allocate
VaR = NaN(T, P);
ES  = NaN(T, P);

for t = t_start:T
    % Get parameters
    n_pars = size(GARCHpars, 2);
    pars   = reshape(GARCHpars(t,:,1), 1, n_pars);

    % Map to marginal distribution
    switch margDist
        case 'norm'
            z_sim = StdN;   % already standard normal
        case 't'
            nu_t  = margNu(t, 1);
            z_sim = trnd(nu_t, H, M) / sqrt(nu_t/(nu_t-2));
    end

    % Simulate H-step ahead returns
    omega = pars(1);
    alpha_g = pars(2);
    beta  = pars(3);
    h     = omega + alpha_g*(Returns(t-1,1) - mu(t,1))^2 + beta*H_last(t,1);
    r_sim = NaN(H, M);
    for j = 1:H
        r_sim(j,:) = mu(t,1) + sqrt(h) .* z_sim(j,:);
        if j < H
            eps2 = (r_sim(j,:) - mu(t,1)).^2;
            h    = omega + alpha_g*eps2 + beta*h;
        end
    end

    % Cumulative return at h=H
    cum_ret  = sum(r_sim, 1);   % (1 x M)
    sorted   = sort(cum_ret);

    % VaR and ES per alpha level
    for pp = 1:P
        idx       = max(1, floor(alpha(pp) * M));
        VaR(t,pp) = sorted(idx);
        ES(t,pp)  = mean(sorted(1:idx));
    end
end

end