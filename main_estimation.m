%% Portfolio VaR and ES - cumulative h-step-ahead 

% Change directory
cd 'C:/Users/Schick/Documents/Forschung/8_Portfolio_Var_ES_hstep'

close all
clc

% Directories
addpath("Data")
addpath(genpath('Code'));


%% Setup

% Load return data
clear
ReturnData = readtable('CTC_RET.xlsx','VariableNamingRule','preserve');
RVData     = load('Start_data_10_30_50_2002_2023.mat');
RV         = RVData.RCOV_CTC_vech_10;

% Specify assets (manually or by position in ReturnData) and extract data
% assets = {'AXP', 'BA'};
VarNames = ReturnData.Properties.VariableNames;
VarNames = VarNames(2:11);
assets = cellfun(@(x) x(2:end-1), VarNames, 'UniformOutput', false);
assets_q = strcat("'", assets, "'");
dates    = ReturnData.Var1;
R        = table2array(ReturnData(:, assets_q));

% Equally weighted portfolio
weightMat = ones(max(size(assets)), 1) / max(size(assets));

% Rolling-window setting
reest_freq = 21;    % re-estimate every 21 observations
WindLength = 1000;  % Window length

% Simulation set-up
Hsim = 10;
Msim = 25000;

% Number of workers for parallel computing 
NumberWorkers = 4;


%%% Specify strings to handle model names 

% Specify models depending on marginal specifications estimated previously
MarginalModels = {'GARCH_norm', 'GARCH_t', 'GARCH_skewt', ...
                  'GARCH_laplace', 'RiskMetrics_GARCH_norm', ...
                  'GJRGARCH_norm', 'GJRGARCH_t', 'GJRGARCH_skewt', ...
                  'GJRGARCH_laplace', ...
                  'HEAVY_norm', 'HEAVY_laplace'}; % Add marginal models here

% Specify models depending on copula specifications estimated previously
CopulaModels = {'CCC_norm', 'CCC_t', 'CCC_norm_empirical', ...
                'CCC_t_empirical',  ...
                'DCC_norm_empirical', ...
                %'DCC_t'
                }; % Add Copula models here 


%% Rolling-window estimation of the marginals

% GARCH-Normal
estimate_garch(R, 'dist', 'norm', 'WindLength', WindLength, ...
                  'ReestFreq', reest_freq, 'assets', assets, ...
                  'NumWorkers', NumberWorkers, 'dates', dates);

% GARCH-t
estimate_garch(R, 'dist', 't', 'WindLength', WindLength, ...
                  'ReestFreq', reest_freq, 'assets', assets, ...
                  'NumWorkers', NumberWorkers, 'dates', dates);

% GARCH-Skew-t
estimate_garch(R, 'dist', 'skewt', 'WindLength', WindLength, ...
                  'ReestFreq', reest_freq, 'assets', assets, ...
                  'NumWorkers', NumberWorkers, 'dates', dates);

% GARCH-Laplace
estimate_garch(R, 'dist', 'laplace', 'WindLength', WindLength, ...
                  'ReestFreq', reest_freq, 'assets', assets, ...
                  'NumWorkers', NumberWorkers, 'dates', dates);


% GJR-GARCH-Normal
estimate_gjr_garch(R, 'dist', 'norm', 'WindLength', WindLength, ...
                      'ReestFreq', reest_freq, 'assets', assets, ...
                      'NumWorkers', NumberWorkers, 'dates', dates);

% GJR-GARCH-t
estimate_gjr_garch(R, 'dist', 't', 'WindLength', WindLength, ...
                      'ReestFreq', reest_freq, 'assets', assets, ...
                      'NumWorkers', NumberWorkers, 'dates', dates);

% GJR-GARCH-Skew-t
estimate_gjr_garch(R, 'dist', 'skewt', 'WindLength', WindLength, ...
                      'ReestFreq', reest_freq, 'assets', assets, ...
                      'NumWorkers', NumberWorkers, 'dates', dates);

% GJR-GARCH-Laplace
estimate_gjr_garch(R, 'dist', 'laplace', 'WindLength', WindLength, ...
                      'ReestFreq', reest_freq, 'assets', assets, ...
                      'NumWorkers', NumberWorkers, 'dates', dates);


% RiskMetrics
estimate_garch(R, 'dist', 'norm', 'WindLength', WindLength, ...
                  'ReestFreq', reest_freq, 'assets', assets, ...
                  'NumWorkers', NumberWorkers, 'dates', dates, ...
                  'RiskMetrics', true);


% Heavy-Normal
estimate_heavy(R, RV, 'dist', 'norm', 'WindLength', WindLength, ...
                      'ReestFreq', reest_freq, 'assets', assets, ...
                      'NumWorkers', NumberWorkers, 'dates', dates);

% Heavy-Laplace
estimate_heavy(R, RV, 'dist', 'laplace', 'WindLength', WindLength, ...
                      'ReestFreq', reest_freq, 'assets', assets, ...
                      'NumWorkers', NumberWorkers, 'dates', dates);



%% Rolling-window estimation of the copula
clearvars -except R RV assets weightMat NumberWorkers dates Hsim Msim ...
                  MarginalModels CopulaModels

% Load all combinations of univariate variance models and assets
EstOut = read_marg_est_results(MarginalModels,assets);


%%%% GARCH(1,1) with various distributions and Gaussian CCC Copula

% GARCH(1,1)-Normal
estimate_copula(EstOut.GARCH_norm, 'copula_dist', 'norm', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% GARCH(1,1)-t
estimate_copula(EstOut.GARCH_t, 'copula_dist', 'norm', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% GARCH(1,1)-Skew-t
estimate_copula(EstOut.GARCH_skewt, 'copula_dist', 'norm', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% GARCH(1,1)-Laplace
estimate_copula(EstOut.GARCH_laplace, 'copula_dist', 'norm', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% GARCH(1,1)-Empirical
estimate_copula(EstOut.GARCH_norm, 'copula_dist', 'norm', ...
                'corr_model', 'CCC', 'empirical_pits', true, ...
                'NumWorkers', NumberWorkers);


%%%% GARCH(1,1) with various distributions and Student-t CCC Copula

% GARCH(1,1)-Normal
estimate_copula(EstOut.GARCH_norm, 'copula_dist', 't', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% GARCH(1,1)-t
estimate_copula(EstOut.GARCH_t, 'copula_dist', 't', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% GARCH(1,1)-Skew-t
estimate_copula(EstOut.GARCH_skewt, 'copula_dist', 't', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% GARCH(1,1)-Laplace
estimate_copula(EstOut.GARCH_laplace, 'copula_dist', 't', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);


%%% Risk metrics with Gaussian and Student-t CCC Copula

% Gaussian Copula
estimate_copula(EstOut.RiskMetrics_GARCH_norm, 'copula_dist', 'norm', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% Student-t Copula
estimate_copula(EstOut.RiskMetrics_GARCH_norm, 'copula_dist', 't', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);


%%%% GJR-GARCH(1,1) with various distributions and Gaussian CCC Copula

% GJR-GARCH(1,1)-Normal
estimate_copula(EstOut.GJRGARCH_norm, 'copula_dist', 'norm', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% GJR-GARCH(1,1)-t
estimate_copula(EstOut.GJRGARCH_t, 'copula_dist', 'norm', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% GJR-GARCH(1,1)-Skew-t
estimate_copula(EstOut.GJRGARCH_skewt, 'copula_dist', 'norm', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% GJR-GARCH(1,1)-Laplace
estimate_copula(EstOut.GJRGARCH_laplace, 'copula_dist', 'norm', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% GJR-GARCH(1,1)-Empirical
estimate_copula(EstOut.GJRGARCH_norm, 'copula_dist', 'norm', ...
                'corr_model', 'CCC', 'empirical_pits', true, ...
                'NumWorkers', NumberWorkers);


%%%% GJR-GARCH(1,1) with various distributions and Student-t CCC Copula

% GJR-GARCH(1,1)-Normal
estimate_copula(EstOut.GJRGARCH_norm, 'copula_dist', 't', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% GJR-GARCH(1,1)-t
estimate_copula(EstOut.GJRGARCH_t, 'copula_dist', 't', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% GJR-GARCH(1,1)-Skew-t
estimate_copula(EstOut.GJRGARCH_skewt, 'copula_dist', 't', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% GJR-GARCH(1,1)-Laplace
estimate_copula(EstOut.GJRGARCH_laplace, 'copula_dist', 't', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);



%%%% GJR-GARCH(1,1) with various distributions and DCC Copula

% GJR-GARCH(1,1)-Empirical
estimate_copula(EstOut.GJRGARCH_norm, 'copula_dist', 'norm', ...
                'corr_model', 'DCC', 'empirical_pits', true, ...
                'NumWorkers', NumberWorkers);



%%%% HEAVY-model with various distributions and Gaussian CCC Copula

% HEAVY-Laplace
estimate_copula(EstOut.HEAVY_laplace, 'copula_dist', 'norm', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% HEAVY-Empirical
estimate_copula(EstOut.HEAVY_norm, 'copula_dist', 'norm', ...
                'corr_model', 'CCC', 'empirical_pits', true, ...
                'NumWorkers', NumberWorkers);



%% Simulation of H-step ahead portfolio VaR and ES from copula models
clearvars -except R RV assets MarginalModels weightMat NumberWorkers ...
                  dates Hsim Msim MarginalModels CopulaModels

% Load all combinations of variance models, assets, and copulas
EstOut = read_copula_est_results(MarginalModels,CopulaModels,assets);
% Check warnings, some specs are supposed to be missing, e.g. the empirical 
% distribution should only feature normal marginals and a Gaussian Copula!

% Simulate H-step ahead returns for all models in EstOut and compute Var/ES
alpha = [0.01, 0.025];
VaRandES = simulate_all_var_es(EstOut, R, weightMat, alpha, ...
                               'H', Hsim, 'M', Msim, ...
                               'NumWorkers', NumberWorkers, ...
                               'RV', RV, ...
                               'OutputName', 'EquallyWeighted');


% GARCH on equally weighted portfolio

% Equally weighted PF returns
R_pf  = R * weightMat;
dists = {'norm', 't', 'skewt', 'laplace'};

% GARCH benchmarks
for d = 1:length(dists)
    VaRandES = pf_garch_benchmark(dists{d}, R_pf, VaRandES, ...
        VaRandES.WindLength, VaRandES.ReestFreq, assets, dates, ...
        NumberWorkers, Hsim, Msim);
end

% GARCH empirical
VaRandES = pf_garch_benchmark('norm', R_pf, VaRandES, ...
    VaRandES.WindLength, VaRandES.ReestFreq, assets, dates, ...
    NumberWorkers, Hsim, Msim, 'EmpiricalPits', true);

% GJR-GARCH benchmarks
for d = 1:length(dists)
    VaRandES = pf_garch_benchmark(dists{d}, R_pf, VaRandES, ...
        VaRandES.WindLength, VaRandES.ReestFreq, assets, dates, ...
        NumberWorkers, Hsim, Msim, 'GJR', true);
end

% GJR-GARCH empirical
VaRandES = pf_garch_benchmark('norm', R_pf, VaRandES, ...
    VaRandES.WindLength, VaRandES.ReestFreq, assets, dates, ...
    NumberWorkers, Hsim, Msim, 'EmpiricalPits', true, 'GJR', true);


% Shut down parallel pool
if NumberWorkers > 1
    pool = gcp('nocreate');
    delete(pool);
end


% Add additional forecasts from other models or weighting schemes here

% VaRandES already saved inside simulate_all_var_es
% Re-save with PF benchmarks added
assets_str = strjoin(assets, '_');
filename   = sprintf('Output/VaRandES/VaRandES_EquallyWeighted_%s.mat', ...
                     assets_str);
save(filename, 'VaRandES');



%% Forecast evaluation
clearvars -except R VaRandESpath assets

% Read in VaR and ES forecasts per model
clearvars -except R assets
load(sprintf('Output/VaRandES/VaRandES_EquallyWeighted_%s.mat', ...
             strjoin(assets, '_')));


%%%% Fissler-Ziegel loss function for model comparison

% Fissler-Ziegel (FZ) loss and Model Confidence Set of 1% VaR and ES
VaRandES_0010 = VaRandES; 
VaRandES_0010.alpha = VaRandES_0010.alpha(1);
VaRandES_0010.VaR(:,:,:,2)=[]; VaRandES_0010.VaR = squeeze(VaRandES_0010.VaR);
VaRandES_0010.ES(:,:,:,2)=[]; VaRandES_0010.ES = squeeze(VaRandES_0010.ES);
MCSTable_0010 = score_fz(VaRandES_0010, R, 'HEval', 10, ...
                         'DateStart', 20060201, 'DateEnd', 20221230, ...
                         'Decimals', 3, 'PrintTable', false);

% Fissler-Ziegel (FZ) loss and Model Confidence Set of 2.5% VaR and ES
MCSTable_0025 = score_fz(VaRandES_0025, R, 'HEval', 10, ...
                         'DateStart', 20060201, 'DateEnd', 20221230, ...
                         'Decimals', 3, 'PrintTable', false);


%%%% Backtesting of VaR and ES
var_uc_test(VaRandES_0010, R)

var_uc_test(VaRandES_0025, R)



%% Monte Carlo - GARCH-N
T_sim      = 5000;
omega_true = 0.01;
alpha_true = 0.03;
beta_true  = 0.95;
nu_true    = 5;

B     = 500;
Msim  = 25000;
H_mc  = 10;
alpha = [0.01, 0.025];

% Estimate GARCH-Normal/t
WindLength_mc = 1000;
reest_freq_mc = 250;

UC_t                = NaN(B, 2);
UC_p_one_sided      = NaN(B, 2);
UC_p_two_sided      = NaN(B, 2);
UC_bern_t           = NaN(B, 2);
UC_bern_p_one_sided = NaN(B, 2);
UC_bern_p_two_sided = NaN(B, 2);
ES_t                = NaN(B, 2);
ES_p_one_sided      = NaN(B, 2);
ES_p_two_sided      = NaN(B, 2);
GARCH_pars          = NaN(B, 4);

% Draw random numbers
rng(1)
z_sim = randn(H_mc, Msim);

for b = 1:B

    % Simulate GARCH(1,1)-Normal
    rng(b)
    r_sim    = NaN(T_sim, 1);
    h_sim    = NaN(T_sim, 1);
    h_sim(1) = omega_true / (1 - alpha_true - beta_true);
    r_sim(1) = sqrt(h_sim(1)) * randn;
    for t = 2:T_sim
        h_sim(t) = omega_true + alpha_true*r_sim(t-1)^2 ...
                   + beta_true*h_sim(t-1);
        r_sim(t) = sqrt(h_sim(t)) * randn;
    end

    % Simulate VaR and ES 
    [VaR, ES, EstPars] = simulate_var_es_univ(r_sim, alpha,'H', H_mc, ...
                                           'M', Msim, ...
                                           'WindLength', WindLength_mc, ...
                                           'ReestFreq', reest_freq_mc, ...
                                           'TruePars', [], 'z_sim', z_sim);

    % US and ES test
    TestRes_0010 = var_uc_test_sim(VaR(:,1), ES(:,1), r_sim, alpha(1), ...
                                   10, WindLength_mc);
    TestRes_0025 = var_uc_test_sim(VaR(:,2), ES(:,2), r_sim, alpha(2), ...
                                   10, WindLength_mc);

    % Collect test results
    GARCH_pars(b, :) = [mean(EstPars.mu(WindLength_mc+1:end)), ...
                        mean(EstPars.GARCHpars(WindLength_mc+1:end,:))];
    UC_t(b,:)                = [TestRes_0010.UC_t,                TestRes_0025.UC_t];
    UC_bern_t(b,:)           = [TestRes_0010.UC_bern_t,           TestRes_0025.UC_bern_t];
    UC_p_one_sided(b,:)      = [TestRes_0010.UC_p_one_sided,      TestRes_0025.UC_p_one_sided];
    UC_p_two_sided(b,:)      = [TestRes_0010.UC_p_two_sided,      TestRes_0025.UC_p_two_sided];
    UC_bern_p_one_sided(b,:) = [TestRes_0010.UC_bern_p_one_sided, TestRes_0025.UC_bern_p_one_sided];
    UC_bern_p_two_sided(b,:) = [TestRes_0010.UC_bern_p_two_sided, TestRes_0025.UC_bern_p_two_sided];
    ES_t(b,:)                = [TestRes_0010.ES_t,                TestRes_0025.ES_t];
    ES_p_one_sided(b,:)      = [TestRes_0010.ES_p_one_sided,      TestRes_0025.ES_p_one_sided];
    ES_p_two_sided(b,:)      = [TestRes_0010.ES_p_two_sided,      TestRes_0025.ES_p_two_sided];

    fprintf('Replication %d/%d done\n', b, B);

end

% Pack simulation results
SimBacktest.UC_t                = UC_t;
SimBacktest.UC_p_one_sided      = UC_p_one_sided;
SimBacktest.UC_p_two_sided      = UC_p_two_sided;
SimBacktest.UC_bern_t           = UC_bern_t;
SimBacktest.UC_bern_p_one_sided = UC_bern_p_one_sided;
SimBacktest.UC_bern_p_two_sided = UC_bern_p_two_sided;
SimBacktest.ES_t                = ES_t;
SimBacktest.ES_p_one_sided      = ES_p_one_sided;
SimBacktest.ES_p_two_sided      = ES_p_two_sided;
SimBacktest.GARCH_pars          = GARCH_pars;
SimBacktest.B                   = B;
SimBacktest.T_sim               = T_sim;
SimBacktest.H_mc                = H_mc;
SimBacktest.Msim                = Msim;
SimBacktest.WindLength_mc       = WindLength_mc;
SimBacktest.reest_freq_mc       = reest_freq_mc;
SimBacktest.omega_true          = omega_true;
SimBacktest.alpha_true          = alpha_true;
SimBacktest.beta_true           = beta_true;
SimBacktest.alpha               = alpha;

filename = sprintf(['Output/SimBacktests/SimBacktest_EstPars_GARCHN_' ...
                    'B%d_T%d_H%d_M%d.mat'], B, T_sim, H_mc, Msim);
save(filename, 'SimBacktest');
fprintf('Results saved to %s\n', filename);


% True parameters
true_vec = [omega_true, alpha_true, beta_true];

UC_t                = NaN(B, 2);
UC_p_one_sided      = NaN(B, 2);
UC_p_two_sided      = NaN(B, 2);
UC_bern_t           = NaN(B, 2);
UC_bern_p_one_sided = NaN(B, 2);
UC_bern_p_two_sided = NaN(B, 2);
ES_t                = NaN(B, 2);
ES_p_one_sided      = NaN(B, 2);
ES_p_two_sided      = NaN(B, 2);
GARCH_pars          = NaN(B, 4);

% Draw random numbers
rng(1)
z_sim = randn(H_mc, Msim);

for b = 1:B

    % Simulate GARCH(1,1)-Normal
    rng(b)
    r_sim    = NaN(T_sim, 1);
    h_sim    = NaN(T_sim, 1);
    h_sim(1) = omega_true / (1 - alpha_true - beta_true);
    r_sim(1) = sqrt(h_sim(1)) * randn;
    for t = 2:T_sim
        h_sim(t) = omega_true + alpha_true*r_sim(t-1)^2 ...
                   + beta_true*h_sim(t-1);
        r_sim(t) = sqrt(h_sim(t)) * randn;
    end

    % Simulate VaR and ES 
    [VaR, ES, EstPars] = simulate_var_es_univ(r_sim, alpha,'H', H_mc, ...
                                           'M', Msim, ...
                                           'WindLength', WindLength_mc, ...
                                           'ReestFreq', reest_freq_mc, ...
                                           'TruePars', true_vec, ...
                                           'z_sim', z_sim);

    % US and ES test
    TestRes_0010 = var_uc_test_sim(VaR(:,1), ES(:,1), r_sim, alpha(1), ...
                                   10, WindLength_mc);
    TestRes_0025 = var_uc_test_sim(VaR(:,2), ES(:,2), r_sim, alpha(2), ...
                                   10, WindLength_mc);

    % Collect test results
    GARCH_pars(b, :) = [mean(EstPars.mu(WindLength_mc+1:end)), ...
                        mean(EstPars.GARCHpars(WindLength_mc+1:end,:))];
    UC_t(b,:)                = [TestRes_0010.UC_t,                TestRes_0025.UC_t];
    UC_bern_t(b,:)           = [TestRes_0010.UC_bern_t,           TestRes_0025.UC_bern_t];
    UC_p_one_sided(b,:)      = [TestRes_0010.UC_p_one_sided,      TestRes_0025.UC_p_one_sided];
    UC_p_two_sided(b,:)      = [TestRes_0010.UC_p_two_sided,      TestRes_0025.UC_p_two_sided];
    UC_bern_p_one_sided(b,:) = [TestRes_0010.UC_bern_p_one_sided, TestRes_0025.UC_bern_p_one_sided];
    UC_bern_p_two_sided(b,:) = [TestRes_0010.UC_bern_p_two_sided, TestRes_0025.UC_bern_p_two_sided];
    ES_t(b,:)                = [TestRes_0010.ES_t,                TestRes_0025.ES_t];
    ES_p_one_sided(b,:)      = [TestRes_0010.ES_p_one_sided,      TestRes_0025.ES_p_one_sided];
    ES_p_two_sided(b,:)      = [TestRes_0010.ES_p_two_sided,      TestRes_0025.ES_p_two_sided];

    fprintf('Replication %d/%d done\n', b, B);

end

% Pack simulation results
SimBacktest.UC_t                = UC_t;
SimBacktest.UC_p_one_sided      = UC_p_one_sided;
SimBacktest.UC_p_two_sided      = UC_p_two_sided;
SimBacktest.UC_bern_t           = UC_bern_t;
SimBacktest.UC_bern_p_one_sided = UC_bern_p_one_sided;
SimBacktest.UC_bern_p_two_sided = UC_bern_p_two_sided;
SimBacktest.ES_t                = ES_t;
SimBacktest.ES_p_one_sided      = ES_p_one_sided;
SimBacktest.ES_p_two_sided      = ES_p_two_sided;
SimBacktest.GARCH_pars          = GARCH_pars;
SimBacktest.B                   = B;
SimBacktest.T_sim               = T_sim;
SimBacktest.H_mc                = H_mc;
SimBacktest.Msim                = Msim;
SimBacktest.WindLength_mc       = WindLength_mc;
SimBacktest.reest_freq_mc       = reest_freq_mc;
SimBacktest.omega_true          = omega_true;
SimBacktest.alpha_true          = alpha_true;
SimBacktest.beta_true           = beta_true;
SimBacktest.alpha               = alpha;

filename = sprintf(['Output/SimBacktests/SimBacktest_TruePars_GARCHN_' ...
                    'B%d_T%d_H%d_M%d.mat'], B, T_sim, H_mc, Msim);
save(filename, 'SimBacktest');
fprintf('Results saved to %s\n', filename);


% Monte Carlo - GARCH-t
UC_t                = NaN(B, 2);
UC_p_one_sided      = NaN(B, 2);
UC_p_two_sided      = NaN(B, 2);
UC_bern_t           = NaN(B, 2);
UC_bern_p_one_sided = NaN(B, 2);
UC_bern_p_two_sided = NaN(B, 2);
ES_t                = NaN(B, 2);
ES_p_one_sided      = NaN(B, 2);
ES_p_two_sided      = NaN(B, 2);
GARCH_pars          = NaN(B, 5);

% Draw random numbers
rng(1)
z_sim = trnd(nu_true, H_mc, Msim) / sqrt(nu_true/(nu_true-2));

for b = 1:B

    % Simulate GARCH(1,1)-t
    rng(b)
    r_sim    = NaN(T_sim, 1);
    h_sim    = NaN(T_sim, 1);
    h_sim(1) = omega_true / (1 - alpha_true - beta_true);
    r_sim(1) = sqrt(h_sim(1)) * trnd(nu_true) / sqrt(nu_true/(nu_true-2));
    for t = 2:T_sim
        h_sim(t) = omega_true + alpha_true*r_sim(t-1)^2 ...
                   + beta_true*h_sim(t-1);
        r_sim(t) = sqrt(h_sim(t))*trnd(nu_true)/sqrt(nu_true/(nu_true-2));
    end

    % Simulate VaR and ES 
    [VaR, ES, EstPars] = simulate_var_es_univ(r_sim, alpha,'H', H_mc, ...
                                           'M', Msim, 'dist', 't', ...
                                           'WindLength', WindLength_mc, ...
                                           'ReestFreq', reest_freq_mc, ...
                                           'TruePars', [], 'z_sim', z_sim);

    % US and ES test
    TestRes_0010 = var_uc_test_sim(VaR(:,1), ES(:,1), r_sim, alpha(1), ...
                                   10, WindLength_mc);
    TestRes_0025 = var_uc_test_sim(VaR(:,2), ES(:,2), r_sim, alpha(2), ...
                                   10, WindLength_mc);

    % Collect test results
    GARCH_pars(b, :) = [mean(EstPars.mu(WindLength_mc+1:end)), ...
                        mean(EstPars.GARCHpars(WindLength_mc+1:end,:)), ...
                        mean(EstPars.margNu(WindLength_mc+1:end,:))];
    UC_t(b,:)                = [TestRes_0010.UC_t,                TestRes_0025.UC_t];
    UC_bern_t(b,:)           = [TestRes_0010.UC_bern_t,           TestRes_0025.UC_bern_t];
    UC_p_one_sided(b,:)      = [TestRes_0010.UC_p_one_sided,      TestRes_0025.UC_p_one_sided];
    UC_p_two_sided(b,:)      = [TestRes_0010.UC_p_two_sided,      TestRes_0025.UC_p_two_sided];
    UC_bern_p_one_sided(b,:) = [TestRes_0010.UC_bern_p_one_sided, TestRes_0025.UC_bern_p_one_sided];
    UC_bern_p_two_sided(b,:) = [TestRes_0010.UC_bern_p_two_sided, TestRes_0025.UC_bern_p_two_sided];
    ES_t(b,:)                = [TestRes_0010.ES_t,                TestRes_0025.ES_t];
    ES_p_one_sided(b,:)      = [TestRes_0010.ES_p_one_sided,      TestRes_0025.ES_p_one_sided];
    ES_p_two_sided(b,:)      = [TestRes_0010.ES_p_two_sided,      TestRes_0025.ES_p_two_sided];

    fprintf('Replication %d/%d done\n', b, B);

end

% Pack simulation results
SimBacktest.UC_t                = UC_t;
SimBacktest.UC_p_one_sided      = UC_p_one_sided;
SimBacktest.UC_p_two_sided      = UC_p_two_sided;
SimBacktest.UC_bern_t           = UC_bern_t;
SimBacktest.UC_bern_p_one_sided = UC_bern_p_one_sided;
SimBacktest.UC_bern_p_two_sided = UC_bern_p_two_sided;
SimBacktest.ES_t                = ES_t;
SimBacktest.ES_p_one_sided      = ES_p_one_sided;
SimBacktest.ES_p_two_sided      = ES_p_two_sided;
SimBacktest.GARCH_pars          = GARCH_pars;
SimBacktest.B                   = B;
SimBacktest.T_sim               = T_sim;
SimBacktest.H_mc                = H_mc;
SimBacktest.Msim                = Msim;
SimBacktest.WindLength_mc       = WindLength_mc;
SimBacktest.reest_freq_mc       = reest_freq_mc;
SimBacktest.omega_true          = omega_true;
SimBacktest.alpha_true          = alpha_true;
SimBacktest.beta_true           = beta_true;
SimBacktest.alpha               = alpha;

filename = sprintf(['Output/SimBacktests/SimBacktest_EstPars_GARCHt_' ...
                    'B%d_T%d_H%d_M%d.mat'], B, T_sim, H_mc, Msim);
save(filename, 'SimBacktest');
fprintf('Results saved to %s\n', filename);


% True parameters
true_vec = [omega_true, alpha_true, beta_true, nu_true];

UC_t                = NaN(B, 2);
UC_p_one_sided      = NaN(B, 2);
UC_p_two_sided      = NaN(B, 2);
UC_bern_t           = NaN(B, 2);
UC_bern_p_one_sided = NaN(B, 2);
UC_bern_p_two_sided = NaN(B, 2);
ES_t                = NaN(B, 2);
ES_p_one_sided      = NaN(B, 2);
ES_p_two_sided      = NaN(B, 2);
GARCH_pars          = NaN(B, 5);

% Draw random numbers
rng(1)
z_sim = trnd(nu_true, H_mc, Msim) / sqrt(nu_true/(nu_true-2));

for b = 1:B

    % Simulate GARCH(1,1)-Normal
    rng(b)
    r_sim    = NaN(T_sim, 1);
    h_sim    = NaN(T_sim, 1);
    h_sim(1) = omega_true / (1 - alpha_true - beta_true);
    r_sim(1) = sqrt(h_sim(1)) * trnd(nu_true) / sqrt(nu_true/(nu_true-2));
    for t = 2:T_sim
        h_sim(t) = omega_true + alpha_true*r_sim(t-1)^2 ...
                   + beta_true*h_sim(t-1);
        r_sim(t) = sqrt(h_sim(t))*trnd(nu_true)/sqrt(nu_true/(nu_true-2));
    end

    % Simulate VaR and ES 
    [VaR, ES, EstPars] = simulate_var_es_univ(r_sim, alpha,'H', H_mc, ...
                                           'M', Msim, 'dist', 't', ...
                                           'WindLength', WindLength_mc, ...
                                           'ReestFreq', reest_freq_mc, ...
                                           'TruePars', true_vec, ...
                                           'z_sim', z_sim);

    % US and ES test
    TestRes_0010 = var_uc_test_sim(VaR(:,1), ES(:,1), r_sim, alpha(1), ...
                                   10, WindLength_mc);
    TestRes_0025 = var_uc_test_sim(VaR(:,2), ES(:,2), r_sim, alpha(2), ...
                                   10, WindLength_mc);

    % Collect test results
    GARCH_pars(b, :) = [mean(EstPars.mu(WindLength_mc+1:end)), ...
                        mean(EstPars.GARCHpars(WindLength_mc+1:end,:)), ...
                        mean(EstPars.margNu(WindLength_mc+1:end,:))];
    UC_t(b,:)                = [TestRes_0010.UC_t,                TestRes_0025.UC_t];
    UC_bern_t(b,:)           = [TestRes_0010.UC_bern_t,           TestRes_0025.UC_bern_t];
    UC_p_one_sided(b,:)      = [TestRes_0010.UC_p_one_sided,      TestRes_0025.UC_p_one_sided];
    UC_p_two_sided(b,:)      = [TestRes_0010.UC_p_two_sided,      TestRes_0025.UC_p_two_sided];
    UC_bern_p_one_sided(b,:) = [TestRes_0010.UC_bern_p_one_sided, TestRes_0025.UC_bern_p_one_sided];
    UC_bern_p_two_sided(b,:) = [TestRes_0010.UC_bern_p_two_sided, TestRes_0025.UC_bern_p_two_sided];
    ES_t(b,:)                = [TestRes_0010.ES_t,                TestRes_0025.ES_t];
    ES_p_one_sided(b,:)      = [TestRes_0010.ES_p_one_sided,      TestRes_0025.ES_p_one_sided];
    ES_p_two_sided(b,:)      = [TestRes_0010.ES_p_two_sided,      TestRes_0025.ES_p_two_sided];

    fprintf('Replication %d/%d done\n', b, B);

end

% Pack simulation results
SimBacktest.UC_t                = UC_t;
SimBacktest.UC_p_one_sided      = UC_p_one_sided;
SimBacktest.UC_p_two_sided      = UC_p_two_sided;
SimBacktest.UC_bern_t           = UC_bern_t;
SimBacktest.UC_bern_p_one_sided = UC_bern_p_one_sided;
SimBacktest.UC_bern_p_two_sided = UC_bern_p_two_sided;
SimBacktest.ES_t                = ES_t;
SimBacktest.ES_p_one_sided      = ES_p_one_sided;
SimBacktest.ES_p_two_sided      = ES_p_two_sided;
SimBacktest.GARCH_pars          = GARCH_pars;
SimBacktest.B                   = B;
SimBacktest.T_sim               = T_sim;
SimBacktest.H_mc                = H_mc;
SimBacktest.Msim                = Msim;
SimBacktest.WindLength_mc       = WindLength_mc;
SimBacktest.reest_freq_mc       = reest_freq_mc;
SimBacktest.omega_true          = omega_true;
SimBacktest.alpha_true          = alpha_true;
SimBacktest.beta_true           = beta_true;
SimBacktest.alpha               = alpha;

filename = sprintf(['Output/SimBacktests/SimBacktest_TruePars_GARCHt_' ...
                    'B%d_T%d_H%d_M%d.mat'], B, T_sim, H_mc, Msim);
save(filename, 'SimBacktest');
fprintf('Results saved to %s\n', filename);


load('Output/SimBacktests/SimBacktest_TruePars_GARCHt_B500_T5000_H10_M25000.mat')

mean(SimBacktest.UC_p_one_sided < 0.05)
mean(SimBacktest.UC_p_two_sided < 0.05)

mean(SimBacktest.UC_bern_p_one_sided < 0.05)
mean(SimBacktest.UC_bern_p_two_sided < 0.05)

mean(SimBacktest.ES_p_one_sided < 0.05)
mean(SimBacktest.ES_p_two_sided < 0.05)

figure
histogram(SimBacktest.UC_t(:,1), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
hold on
x_range = linspace(-5, 5, 1000);
plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
xline(0, 'k--', 'LineWidth', 1.5)
xlabel('t-statistic')
ylabel('Density')
title('1% UC t-statistics vs Standard Normal')
legend('Simulated', 'N(0,1)', 'Location', 'northwest')

figure
histogram(SimBacktest.UC_t(:,2), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
hold on
x_range = linspace(-5, 5, 1000);
plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
xline(0, 'k--', 'LineWidth', 1.5)
xlabel('t-statistic')
ylabel('Density')
title('2.5% UC t-statistics vs Standard Normal')
legend('Simulated', 'N(0,1)', 'Location', 'northwest')

figure
histogram(SimBacktest.UC_bern_t(:,1), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
hold on
x_range = linspace(-5, 5, 1000);
plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
xline(0, 'k--', 'LineWidth', 1.5)
xlabel('t-statistic')
ylabel('Density')
title('1% UC-Bernoulli t-statistics vs Standard Normal')
legend('Simulated', 'N(0,1)', 'Location', 'northwest')

figure
histogram(SimBacktest.UC_bern_t(:,2), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
hold on
x_range = linspace(-5, 5, 1000);
plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
xline(0, 'k--', 'LineWidth', 1.5)
xlabel('t-statistic')
ylabel('Density')
title('2.5% UC-Bernoulli t-statistics vs Standard Normal')
legend('Simulated', 'N(0,1)', 'Location', 'northwest')


figure
histogram(SimBacktest.ES_t(:,1), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
hold on
x_range = linspace(-5, 5, 1000);
plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
xline(0, 'k--', 'LineWidth', 1.5)
xlabel('t-statistic')
ylabel('Density')
title('1% ES t-statistics vs Standard Normal')
legend('Simulated', 'N(0,1)', 'Location', 'northwest')

figure
histogram(SimBacktest.ES_t(:,2), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
hold on
x_range = linspace(-5, 5, 1000);
plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
xline(0, 'k--', 'LineWidth', 1.5)
xlabel('t-statistic')
ylabel('Density')
title('2.5% ES t-statistics vs Standard Normal')
legend('Simulated', 'N(0,1)', 'Location', 'northwest')



%%

% figure
% histogram(tstat_uc(1:b), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
% hold on
% x_range = linspace(-5, 5, 1000);
% plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
% xline(0, 'k--', 'LineWidth', 1.5)
% xlabel('t-statistic')
% ylabel('Density')
% title('UC test t-statistics vs Standard Normal')
% legend('Simulated', 'N(0,1)', 'Location', 'northwest')







%% Monte Carlo size check for backtesting functions - 1-day-ahead
T_sim      = 5000;
omega_true = 0.02;
alpha_true = 0.05;
beta_true  = 0.93;
mu_true    = 0;

% Settings
WindLength_mc = 1000;
reest_freq_mc = 21;
assets_mc     = {'SimAsset'};
dates_mc      = (1:T_sim)';
alpha_mc      = 0.025;
B             = 500;

% Pre-allocate
pValues_UC     = NaN(B, 1);
pValues_UC_bin = NaN(B, 1);
pValues_DQ     = NaN(B, 1);
pValues_ES     = NaN(B, 1);
tstat_uc       = NaN(B, 1);

for b = 1:B
    rng(b)

    % Simulate GARCH(1,1)-Normal
    r_sim    = NaN(T_sim, 1);
    h_sim    = NaN(T_sim, 1);
    h_sim(1) = omega_true / (1 - alpha_true - beta_true);
    r_sim(1) = sqrt(h_sim(1)) * randn;
    for t = 2:T_sim
        h_sim(t) = omega_true + alpha_true*r_sim(t-1)^2 + beta_true*h_sim(t-1);
        r_sim(t) = sqrt(h_sim(t)) * randn;
    end

    % Analytical VaR and ES using true parameters — no simulation needed
    VaR_true = norminv(alpha_mc) * sqrt(h_sim);
    ES_true  = -normpdf(norminv(alpha_mc)) / alpha_mc * sqrt(h_sim);

    % Build VaRandES_mc struct
    VaRandES_mc.VaR        = VaR_true;
    VaRandES_mc.ES         = ES_true;
    VaRandES_mc.alpha      = alpha_mc;
    VaRandES_mc.H          = 1;
    VaRandES_mc.M          = NaN;
    VaRandES_mc.Models     = {'SimAsset'};
    VaRandES_mc.assets     = {'SimAsset'};
    VaRandES_mc.dates      = dates_mc;
    VaRandES_mc.PFweights  = 1;
    VaRandES_mc.WindLength = WindLength_mc;
    VaRandES_mc.ReestFreq  = reest_freq_mc;

    % Collect p-values
    [pUC, pDQ, pES, pUCbinomial, tuc] = var_uc_test(VaRandES_mc, r_sim);
    pValues_UC(b)     = pUC;
    pValues_UC_bin(b) = pUCbinomial;
    pValues_DQ(b)     = pDQ;
    pValues_ES(b)     = pES;
    tstat_uc(b)       = tuc;

    fprintf('Replication %d/%d done\n', b, B);
    fprintf('UC test size:     %.3f\n', mean(pValues_UC(1:b)     < 0.05));
    fprintf('UC Bin test size: %.3f\n', mean(pValues_UC_bin(1:b) < 0.05));
    fprintf('DQ test size:     %.3f\n', mean(pValues_DQ(1:b)     < 0.05));
    fprintf('ES test size:     %.3f\n', mean(pValues_ES(1:b)     < 0.05));
end

% Final empirical size
fprintf('UC test size:     %.3f\n', mean(pValues_UC     < 0.05));
fprintf('UC Bin test size: %.3f\n', mean(pValues_UC_bin < 0.05));
fprintf('DQ test size:     %.3f\n', mean(pValues_DQ     < 0.05));
fprintf('ES test size:     %.3f\n', mean(pValues_ES     < 0.05));

% Plot t-stat distribution
figure
histogram(tstat_uc(1:b), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
hold on
x_range = linspace(-5, 5, 1000);
plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
xline(0, 'k--', 'LineWidth', 1.5)
xlabel('t-statistic')
ylabel('Density')
title('UC test t-statistics vs Standard Normal')
legend('Simulated', 'N(0,1)', 'Location', 'northwest')




% % Insert DCC result at position j=X
% j_insert = X;
% 
% % Shift existing models from j=15 onwards to j=16 onwards
% VaRandES.VaR    = cat(2, VaRandES.VaR(:,1:j_insert-1,:,:), ...
%                           VaRESOut.VaR, ...
%                           VaRandES.VaR(:,j_insert:end,:,:));
% VaRandES.ES     = cat(2, VaRandES.ES(:,1:j_insert-1,:,:), ...
%                           VaRESOut.ES, ...
%                           VaRandES.ES(:,j_insert:end,:,:));
% VaRandES.Models = [VaRandES.Models(1:j_insert-1), ...
%                    {'GJRGARCH_norm_DCC_norm_empirical'}, ...
%                    VaRandES.Models(j_insert:end)];


% new=load('C:\Users\Schick\Documents\Forschung\8_Portfolio_Var_ES_hstep\Output\SimBacktests\checkpoint_simbacktest481_500.mat');
% old=load('C:\Users\Schick\Documents\Forschung\8_Portfolio_Var_ES_hstep\Output\SimBacktests\checkpoint_simbacktest480repMerge.mat');
% 
% merge = new;
% merge.pValues_DQ(1:480,:) = old.pValues_DQ(1:480,:);
% merge.pValues_DQ_sub(1:480,:) = old.pValues_DQ_sub(1:480,:);
% merge.pValues_UC_bin(1:480,:) = old.pValues_UC_bin(1:480,:);
% merge.pValues_UC_bin_sub(1:480,:) = old.pValues_UC_bin_sub(1:480,:);
% merge.tValues_ES(1:480,:) = old.tValues_ES(1:480,:);
% merge.tValues_ES_sub(1:480,:) = old.tValues_ES_sub(1:480,:);
% merge.tValues_UC(1:480,:) = old.tValues_UC(1:480,:);
% merge.tValues_UC_sub(1:480,:) = old.tValues_UC_sub(1:480,:);
% 
% 
% pValues_DQ = merge.pValues_DQ;
% pValues_DQ_sub = merge.pValues_DQ_sub;
% pValues_UC_bin= merge.pValues_UC_bin;
% pValues_UC_bin_sub = merge.pValues_UC_bin_sub;
% tValues_ES = merge.tValues_ES;
% tValues_ES_sub = merge.tValues_ES_sub;
% tValues_UC = merge.tValues_UC;
% tValues_UC_sub = merge.tValues_UC_sub;


%%  UC test two sided
load('C:\Users\Schick\Documents\Forschung\8_Portfolio_Var_ES_hstep\Output\SimBacktests\SimBacktest_B500_T5000_H10_M25000.mat')
t = SimBacktest.tValues_UC;
tsub = SimBacktest.tValues_UC_sub;

% UC no subsampling alpha = 0.01
figure
histogram(t(:,1), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
hold on
x_range = linspace(-5, 5, 1000);
plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
xline(0, 'k--', 'LineWidth', 1.5)
xlabel('t-statistic')
ylabel('Density')
title('UC test t-statistics vs Standard Normal')
legend('Simulated', 'N(0,1)', 'Location', 'northwest')

mean(abs(t(:,1)) >=1.96, 'omitnan')


% UC no subsampling alpha = 0.025
figure
histogram(t(:,2), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
hold on
x_range = linspace(-5, 5, 1000);
plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
xline(0, 'k--', 'LineWidth', 1.5)
xlabel('t-statistic')
ylabel('Density')
title('UC test t-statistics vs Standard Normal')
legend('Simulated', 'N(0,1)', 'Location', 'northwest')

mean(abs(t(:,2)) >=1.96, 'omitnan')


% UC subsampling alpha = 0.01 - Not enough violations!!!!!!!!!!!!!!
figure
histogram(max(min(10,tsub(:,1)),-10), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
hold on
x_range = linspace(-5, 5, 1000);
plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
xline(0, 'k--', 'LineWidth', 1.5)
xlabel('t-statistic')
ylabel('Density')
title('UC test t-statistics vs Standard Normal')
legend('Simulated', 'N(0,1)', 'Location', 'northwest')

mean(abs(tsub(:,1)) >=1.96, 'omitnan')


% UC subsampling alpha = 0.025
figure
histogram(tsub(:,2), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
hold on
x_range = linspace(-5, 5, 1000);
plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
xline(0, 'k--', 'LineWidth', 1.5)
xlabel('t-statistic')
ylabel('Density')
title('UC test t-statistics vs Standard Normal')
legend('Simulated', 'N(0,1)', 'Location', 'northwest')

mean(abs(tsub(:,2)) >=1.96, 'omitnan')


%% UC binomial test

p = SimBacktest.pValues_UC_bin;
psub = SimBacktest.pValues_UC_bin_sub;

% UC w and w/o subsampling alpha = 0.01
mean(p(:,1) < 0.05, 'omitnan')
mean(psub(:,1) < 0.05, 'omitnan')

% UC w and w/o subsampling alpha = 0.025
mean(p(:,2) < 0.05, 'omitnan')
mean(psub(:,2) < 0.05, 'omitnan')


%% DQ test

p = SimBacktest.pValues_DQ;
psub = SimBacktest.pValues_DQ_sub;

% UC w and w/o subsampling alpha = 0.01
mean(p(:,1) < 0.05, 'omitnan')
mean(psub(:,1) < 0.05, 'omitnan')

% UC w and w/o subsampling alpha = 0.025
mean(p(:,2) < 0.05, 'omitnan')
mean(psub(:,2) < 0.05, 'omitnan')



%%  ES test
t = SimBacktest.tValues_ES;
tsub = SimBacktest.tValues_ES_sub;

% ES no subsampling alpha = 0.01
figure
histogram(t(1:500,1), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
hold on
x_range = linspace(-5, 5, 1000);
plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
xline(0, 'k--', 'LineWidth', 1.5)
xlabel('t-statistic')
ylabel('Density')
title('ES test t-statistics vs Standard Normal')
legend('Simulated', 'N(0,1)', 'Location', 'northwest')

mean(abs(t(1:500,1)) >=1.96, 'omitnan')


% ES no subsampling alpha = 0.025
figure
histogram(t(1:500,2), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
hold on
x_range = linspace(-5, 5, 1000);
plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
xline(0, 'k--', 'LineWidth', 1.5)
xlabel('t-statistic')
ylabel('Density')
title('ES test t-statistics vs Standard Normal')
legend('Simulated', 'N(0,1)', 'Location', 'northwest')

mean(abs(t(1:500,2)) >=1.96, 'omitnan')


% ES subsampling alpha = 0.01 - Not enough violations!!!!!!!!!!!!!!
figure
histogram(max(min(10,tsub(1:480,1)),-10), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
hold on
x_range = linspace(-5, 5, 1000);
plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
xline(0, 'k--', 'LineWidth', 1.5)
xlabel('t-statistic')
ylabel('Density')
title('ES test t-statistics vs Standard Normal')
legend('Simulated', 'N(0,1)', 'Location', 'northwest')

mean(abs(tsub(1:480,1)) >=1.96, 'omitnan')


% ES subsampling alpha = 0.025
figure
histogram(tsub(1:500,2), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
hold on
x_range = linspace(-5, 5, 1000);
plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
xline(0, 'k--', 'LineWidth', 1.5)
xlabel('t-statistic')
ylabel('Density')
title('ES test t-statistics vs Standard Normal')
legend('Simulated', 'N(0,1)', 'Location', 'northwest')

mean(abs(tsub(1:500,2)) >=1.96, 'omitnan')