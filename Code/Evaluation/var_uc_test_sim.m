function [TestOut, HitRate, hits] = var_uc_test_sim(VaR, ES, r_sim, ...
                                                            alpha, H, WinL)
% ....!



T = size(r_sim, 1);

% Cumulative H-step ahead returns
CumActualRet = NaN(T, H);
CumSum         = cumsum(r_sim);
for h = 1:H
    CumActualRet(1:T-h+1, h) = CumSum(h:T) - [0; CumSum(1:T-h)];
end
CumActualRet = CumActualRet(:,H);

% Trim sample
CumActualRet = CumActualRet(WinL+1:end-H+1);
VaR          = VaR(WinL+1:end-H+1);
ES           = ES(WinL+1:end-H+1);

% (Demeaned) Hit sequence
hits = CumActualRet <= VaR; % logical
hitsdiff = hits - alpha;    % numeric
HitRate  = mean(hits);


% Unconditional coverage test from Patton et al. (2019)
T_eval = size(hitsdiff, 1);
xbar = HitRate - alpha;
[~, se] = hac(ones(T_eval, 1), hitsdiff, 'type', 'HAC', 'Intercept', ...
                                  false, 'bandwidth', H, 'display', 'off');
UC_t           = xbar / se;
UC_p_one_sided = 1 - normcdf(UC_t);
UC_p_two_sided = 2 * (1 - normcdf(abs(UC_t)));


% Bernoulli-corrected standard error
acf_hits       = autocorr(hitsdiff, 'NumLags', H);
var_bern       = alpha * (1 - alpha);
se_bern        = sqrt((var_bern / T_eval) * (1 + 2*sum(acf_hits(2:end))));
UC_t_bern      = xbar / se_bern;
UC_bern_p_one_sided_bern = 1 - normcdf(UC_t_bern);
UC_bern_p_two_sided_bern = 2 * (1 - normcdf(abs(UC_t_bern)));


% Du & Escanciano - HAC version
hits_numeric = hitsdiff + alpha;
xi_centered = hits_numeric .* (CumActualRet - ES) / alpha;
xbar_es     = mean(xi_centered);
[~, se_es, ~] = hac(ones(T_eval, 1), xi_centered, 'type', 'HAC', ...
                     'Intercept', false, 'bandwidth', H, 'display', 'off');

ES_reg_t = xbar_es / se_es;
ES_reg_p_one_sided = normcdf(ES_reg_t);
ES_reg_p_two_sided = 2 * (1 - normcdf(abs(ES_reg_t)));


% Pack Output
TestOut.UC_t                = UC_t;
TestOut.UC_p_one_sided      = UC_p_one_sided;
TestOut.UC_p_two_sided      = UC_p_two_sided;
TestOut.UC_bern_t           = UC_t_bern;
TestOut.UC_bern_p_one_sided = UC_bern_p_one_sided_bern;
TestOut.UC_bern_p_two_sided = UC_bern_p_two_sided_bern;
TestOut.ES_t                = ES_reg_t;
TestOut.ES_p_one_sided      = ES_reg_p_one_sided;
TestOut.ES_p_two_sided      = ES_reg_p_two_sided;

end