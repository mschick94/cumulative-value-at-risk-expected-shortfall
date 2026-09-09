function [VaR, ES, EstPars] = simulate_var_es_univ(Returns, alpha, varargin)
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
addParameter(p, 'H',          10);
addParameter(p, 'M',          25000);
addParameter(p, 'WindLength', 1000);
addParameter(p, 'ReestFreq',  250);
addParameter(p, 'TruePars',   []);
addParameter(p, 'dist',       'norm');
addParameter(p, 'z_sim',      []);

parse(p, varargin{:});
H        = p.Results.H;
M        = p.Results.M;
WinL     = p.Results.WindLength;
ReFr     = p.Results.ReestFreq;
TruePars = p.Results.TruePars;
dist     = p.Results.dist;
z_sim    = p.Results.z_sim;
P        = length(alpha);

T         = size(Returns, 1);
t_start   = WinL + 1;

%%% GARCH estimation

% fmincon options
options = optimoptions(@fmincon, 'Algorithm', 'sqp', 'Display', 'off');

% Bounds on constraints
epsi = 1e-10;

% Pre-allocate
GARCHpars = NaN(T, 3);
H_last    = NaN(T, 1);
margNu    = NaN(T, 1);
mu        = zeros(T, 1);

% Pre-compute re-estimation dates
reest_dates = t_start:ReFr:T;

if isempty(TruePars)
    EstMean = true;    % estimate mu when estimating parameters
else
    EstMean = false;   % mu=0 known when using true parameters
end

switch dist
    case 'norm'
        A          = [     0,      1,      1];
        b          =  1-epsi;
        lb         = [     0;      0;      0];
        ub         = [   Inf; 1-epsi; 1-epsi];
        start      = [  0.05,   0.05,    0.9];
     
        for i = reest_dates
            r = Returns(i-WinL:i-1);
            if isempty(TruePars)
                ParsEst = fmincon(@(pars) ...
                    VarianceModels.univ_garch(pars, r, EstMean), start, ...
                    A, b, [], [], lb, ub, [], options); 
            else
                ParsEst = TruePars;
            end
            [~, HFilt, muEst] = VarianceModels.univ_garch(ParsEst, r, ...
                                                                  EstMean);
            
            GARCHpars(i, 1:3) = ParsEst;
            H_last(i)         = HFilt(end);
            if isempty(muEst)
                muEst = 0;
            end
            mu(i) = muEst;
        end

    case 't'
        A          = [     0,      1,      1,    0];
        b          =  1-epsi;
        lb         = [     0;      0;      0;  2.2];
        ub         = [   Inf; 1-epsi; 1-epsi; 1000];
        start      = [  0.05,   0.05,    0.9,    5];       

        for i = reest_dates
            r = Returns(i-WinL:i-1);
            if isempty(TruePars)
                ParsEst = fmincon(@(pars) ...
                    VarianceModels.univ_garch_t(pars, r, EstMean), start, ...
                    A, b, [], [], lb, ub, [], options); 
            else
                ParsEst = TruePars;
            end
            [~, HFilt, muEst] = VarianceModels.univ_garch_t(ParsEst, r, ...
                                                                  EstMean);
            
            GARCHpars(i, 1:3) = ParsEst(1:3);
            H_last(i)         = HFilt(end);
            margNu(i)         = ParsEst(4);
            if isempty(muEst)
                muEst = 0;
            end
            mu(i) = muEst;
        end        
end

% Carry-forward between re-estimation dates
for t = t_start:T
    if mod(t - t_start, ReFr) ~= 0
        GARCHpars(t, :) = GARCHpars(t-1, :);
        margNu(t)       = margNu(t-1);
        mu(t)           = mu(t-1);
        omega           = GARCHpars(t, 1);
        alpha_g         = GARCHpars(t, 2);
        beta            = GARCHpars(t, 3);
        eps2            = (Returns(t-1) - mu(t-1))^2; 
        H_last(t)       = omega + alpha_g*eps2 + beta*H_last(t-1);
    end
end


% Draw z_sim if not provided
if isempty(z_sim)
    rng(1)
    switch dist
        case 'norm'
            z_sim = randn(H, M);
        case 't'
            if ~isempty(TruePars)
                nu_fix = TruePars(4);
                z_sim  = trnd(nu_fix, H, M) / sqrt(nu_fix/(nu_fix-2));
            end
            % if TruePars empty: z_sim stays empty, redrawn per t below
    end
end

% Pre-allocate
VaR = NaN(T, P);
ES  = NaN(T, P);

for t = t_start:T
    % Get parameters
    pars = GARCHpars(t,:,1);

    % Draw random numbers
    switch dist
        case 'norm'
            z_t = z_sim;   % always fixed
        case 't'
            if ~isempty(z_sim)
                z_t = z_sim;
            else
                rng(1)
                nu_t = margNu(t, 1);
                z_t  = trnd(nu_t, H, M) / sqrt(nu_t/(nu_t-2));
            end
    end

    % Simulate H-step ahead returns
    omega = pars(1);
    alpha_g = pars(2);
    beta  = pars(3);
    h     = omega + alpha_g*(Returns(t-1,1) - mu(t,1))^2 + beta*H_last(t,1);
    r_sim = NaN(H, M);
    for j = 1:H
        r_sim(j,:) = mu(t,1) + sqrt(h) .* z_t(j,:);
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


EstPars.GARCHpars = GARCHpars;
EstPars.mu        = mu;
EstPars.margNu    = margNu;

end