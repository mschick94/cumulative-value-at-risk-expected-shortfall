function SimBacktest = BackTestSimulation(TruePars, varargin)
%BACKTESTSIMULATION Monte Carlo simulation study for VaR and ES backtests.
%
%   SimBacktest = BACKTESTSIMULATION(TruePars) runs B Monte Carlo
%   replications under a GARCH(1,1) DGP and collects UC and ES test
%   statistics for size analysis.
%
%   INPUTS (required):
%       TruePars : Parameter vector of true DGP parameters
%                  [omega, alpha, beta] for 'norm'
%                  [omega, alpha, beta, nu] for 't'
%
%   INPUTS (optional name-value):
%       'T_sim'         : Scalar, time series length (default: 5000)
%       'M_sim'         : Scalar, simulation paths for VaR/ES (default: 25000)
%       'H'             : Scalar, forecast horizon (default: 10)
%       'B_sim'         : Scalar, Monte Carlo replications (default: 1000)
%       'alpha'         : (1xP) significance levels (default: [0.01 0.025])
%       'dist'          : String, DGP and model distribution
%                         'norm' - GARCH-Normal (default)
%                         't'    - GARCH-t
%       'EstPars'       : Logical, estimate parameters (true) or use
%                         true parameters (false) (default: true)
%       'WindLength_mc' : Scalar, estimation window length (default: 1000)
%       'reest_freq_mc' : Scalar, re-estimation frequency (default: 250)
%
%   OUTPUT:
%       SimBacktest : Struct containing test statistics and settings.
%                     Saved to Output/SimBacktests/ automatically.
%
%   NOTES:
%       - z_sim pre-drawn once outside loop for speed up
%       - rng(b) per replication for reproducibility
%       - Results saved as SimBacktest_<EstPars|TruePars>_GARCH_<dist>_
%         B<B>_T<T>_H<H>_M<M>.mat

% Save caller's random stream state
stream_state = rng;

% Name-value inputs
p = inputParser;
addParameter(p, 'T_sim',         5000); 
addParameter(p, 'M_sim',         25000); 
addParameter(p, 'H',             10); 
addParameter(p, 'B_sim',         1000); 
addParameter(p, 'alpha',         [0.01 0.25]);
addParameter(p, 'dist',          'norm');
addParameter(p, 'EstPars',       true);
addParameter(p, 'WindLength_mc', 1000); 
addParameter(p, 'reest_freq_mc', 250);
parse(p, varargin{:});

T_sim         = p.Results.T_sim;
M_sim         = p.Results.M_sim;
H             = p.Results.H;
B             = p.Results.B_sim;
alpha         = p.Results.alpha;
dist          = p.Results.dist;
EstPars       = p.Results.EstPars;
WindLength_mc = p.Results.WindLength_mc;
reest_freq_mc = p.Results.reest_freq_mc;

% Read out parameters
omega_true = TruePars(1);
alpha_true = TruePars(2);
beta_true  = TruePars(3);

% Additional distribution specific settings
switch dist
    case 'norm'
        pars_n = 4;
        tDist  = false;

        if EstPars
            TrueParsProvide = [];
        else
            TrueParsProvide = TruePars(1:3);
        end

    case 't'
        tDist   = true;
        nu_true = TruePars(4);
        pars_n  = 5;

        if EstPars
            TrueParsProvide = [];
        else
            TrueParsProvide = TruePars(1:4);
        end
end

% Draw random numbers for PF return simulation (also possible for 
% h-step-ahead simulation using the t-distribution iff true parameters are 
% used, otherwise simulate_var_es_univ.m will ignore z_sim)
if tDist
    rng(1)
    z_sim = trnd(nu_true, M_sim, H) / sqrt(nu_true/(nu_true-2));
else
    rng(1)
    z_sim = randn(M_sim, H);
end

% Output matrices
UC_t_HAC_b   = NaN(B, 2);
ES_t_HAC_b   = NaN(B, 2);
ES_t_bern_b  = NaN(B, 2);
HitRate_b    = NaN(B, 2);
GARCH_pars_b = NaN(B, pars_n);

% MC Simulations
for b = 1:B

    rng(b)
    if tDist
        % Simulate GARCH(1,1)-t
        z_t = trnd(nu_true, T_sim, 1) / sqrt(nu_true/(nu_true-2));
        
        r_sim    = NaN(T_sim, 1);
        h_sim    = NaN(T_sim, 1);
        h_sim(1) = omega_true / (1 - alpha_true - beta_true);
        r_sim(1) = sqrt(h_sim(1)) * z_t(1);
        for t = 2:T_sim
            h_sim(t) = omega_true + alpha_true*r_sim(t-1)^2 ...
                       + beta_true*h_sim(t-1);
            r_sim(t) = sqrt(h_sim(t)) * z_t(t);
        end
    else
        % Simulate GARCH(1,1)-Normal
        r_sim    = NaN(T_sim, 1);
        h_sim    = NaN(T_sim, 1);
        h_sim(1) = omega_true / (1 - alpha_true - beta_true);
        r_sim(1) = sqrt(h_sim(1)) * randn;
        for t = 2:T_sim
            h_sim(t) = omega_true + alpha_true*r_sim(t-1)^2 ...
                       + beta_true*h_sim(t-1);
            r_sim(t) = sqrt(h_sim(t)) * randn;
        end
    end

    % Simulate VaR and ES 
    [VaR, ~, EmpPITs, Pars] = simulate_var_es_univ(r_sim, alpha, ...
                                           'H', H, 'M', M_sim, ...,
                                           'dist', dist, ...
                                           'WindLength', WindLength_mc, ...
                                           'ReestFreq', reest_freq_mc, ...
                                           'TruePars', TrueParsProvide, ...
                                           'z_sim', z_sim);

    % US and ES test
    [UC_t_HAC, ES_t_HAC, ES_t_bern, HitRate] = VaR_UC_BacktestSim(VaR, ...
                                  EmpPITs, r_sim, alpha, H, WindLength_mc);

    % Collect test results
    GARCH_pars_b(b, 1:4) = [mean(Pars.mu(WindLength_mc+1:end)), ...
                            mean(Pars.GARCHpars(WindLength_mc+1:end,:))];
    if tDist
        GARCH_pars_b(b, 5) = mean(Pars.margNu(WindLength_mc+1:end,:));
    end

    UC_t_HAC_b(b,:)  = UC_t_HAC;
    ES_t_HAC_b(b,:)  = ES_t_HAC;
    ES_t_bern_b(b,:) = ES_t_bern;
    HitRate_b(b,:)   = HitRate;

    fprintf('Replication %d/%d done\n', b, B);

end


% Pack simulation results
SimBacktest.UC_t_HAC      = UC_t_HAC_b;
SimBacktest.ES_t_HAC      = ES_t_HAC_b;
SimBacktest.ES_t_bern     = ES_t_bern_b;
SimBacktest.HitRate       = HitRate_b;
SimBacktest.GARCH_pars    = GARCH_pars_b;
SimBacktest.alpha         = alpha;
SimBacktest.B             = B;
SimBacktest.T_sim         = T_sim;
SimBacktest.H_mc          = H;
SimBacktest.Msim          = M_sim;
SimBacktest.WindLength_mc = WindLength_mc;
SimBacktest.reest_freq_mc = reest_freq_mc;
SimBacktest.omega_true    = omega_true;
SimBacktest.alpha_true    = alpha_true;
SimBacktest.beta_true     = beta_true;

if tDist
    SimBacktest.nu_true = nu_true;
end

if EstPars
    ParEstTrue = 'EstPars';
else
    ParEstTrue = 'TruePars';
end

% Save simulation results
filename = sprintf(['Output/SimBacktests/SimBacktest_%s_GARCH_%s_' ...
                   'B%d_T%d_H%d_M%d.mat'], ParEstTrue, dist, B, T_sim, ...
                    H, M_sim);
save(filename, 'SimBacktest');
fprintf('Results saved to %s\n', filename);
                
% Restore caller's random stream state
rng(stream_state);


end