function [UC_t_HAC, ES_t_HAC, ES_t_bern, HitRate] = ...
    VaR_UC_BacktestSim(VaR, EmpPITs, r_sim, alpha, H, WinL)
%VAR_UC_BACKTEST_SIM Minimal backtesting function for Monte Carlo 
% simulation study. Computes UC and ES test statistics only.
%
%   [UC_t_HAC, ES_t_HAC, ES_t_bern] = VAR_UC_BACKTEST_SIM(VaR, EmpPITs,
%   r_sim, alpha, H, WinL) computes test statistics for unconditional
%   coverage and ES adequacy at all alpha levels simultaneously.
%
%   INPUTS:
%       VaR     : (T x P) VaR forecasts, negative values
%       EmpPITs : (T x P) empirical PITs from simulated distribution
%       r_sim   : (T x 1) simulated returns
%       alpha   : (1 x P) vector of significance levels
%       H       : Scalar, forecast horizon
%       WinL    : Scalar, estimation window length for sample trimming
%
%   OUTPUTS:
%       UC_t_HAC  : (1 x P) UC t-statistics with HAC standard errors
%                   (Patton et al., 2019). Positive values indicate
%                   too many violations.
%       ES_t_HAC  : (1 x P) ES t-statistics with HAC standard errors
%                   (Du & Escanciano, 2017, PIT formulation). Positive
%                   values indicate ES underestimation.
%       ES_t_bern : (1 x P) ES t-statistics with Bernoulli-corrected
%                   standard errors using theoretical variance and ACF
%                   correction. Preferred for h-step ahead forecasts.
%       HitRate   : (1 x P) hit ratios.
%
%   NOTES:
%       - Evaluation sample trimmed to WinL+1:T-H+1
%       - HAC bandwidth set to H for all tests
%       - One-sided rejection: t > 1.645 for too many violations / ES 
%         underestimation
%       - Two-sided rejection: |t| > 1.96
%       - For full evaluation use VaR_UC_Backtest (p-values, LaTeX etc.)


% Cumulative H-step ahead returns
T = size(r_sim, 1);
CumActualRet = NaN(T, H);
CumSum       = cumsum(r_sim);
for h = 1:H
    CumActualRet(1:T-h+1, h) = CumSum(h:T) - [0; CumSum(1:T-h)];
end
CumActualRet = CumActualRet(:,H);

% Trim sample
CumActualRet = CumActualRet(WinL+1:end-H+1);
VaR          = VaR(WinL+1:end-H+1,:);
EmpPITs      = EmpPITs(WinL+1:end-H+1,:);

% (Demeaned) Hit sequence
hits = CumActualRet <= VaR; % logical
hitsdiff = hits - alpha;    % numeric
HitRate  = mean(hits);

% Unconditional coverage test from Patton et al. (2019) with HAC
T_eval = size(hitsdiff, 1);
q_nr   = length(alpha);
xbar   = HitRate - alpha;

se = NaN(1,q_nr);
for i = 1:q_nr
    [~, se(i)] = hac(ones(T_eval, 1), hitsdiff(:,i), 'type', 'HAC', ...
                     'Intercept', false, 'bandwidth', H, 'display', 'off');
end
UC_t_HAC = xbar ./ se;


% PIT-based ES test (Du & Escanciano via empirical PITs)
E_h   = alpha / 2;
Var_h = alpha/3 - alpha.^2/4;

H_mat  = (1./alpha) .* (alpha - EmpPITs) .* (EmpPITs <= alpha);
xbar_h = mean(H_mat);

% Version 1 — HAC
se = NaN(1,q_nr);
for i = 1:q_nr
    [~, se(i)] = hac(ones(T_eval, 1), H_mat(:,i), 'type', 'HAC', ...
                     'Intercept', false, 'bandwidth', H, 'display', 'off');
end
ES_t_HAC = (xbar_h - E_h) ./ se;

% Version 2 — Bernoulli corrected with ACF
var_bern = NaN(1,q_nr);
for i = 1:q_nr
    acf_h = autocorr(H_mat(:,i), 'NumLags', H+1);
    var_bern(i) = (Var_h(i) / T_eval) * (1 + 2*sum(acf_h(2:H+1)));
end
ES_t_bern = (xbar_h - E_h) ./ sqrt(var_bern);


end