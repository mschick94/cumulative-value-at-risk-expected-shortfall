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
                  'GJRGARCH_laplace'}; % Add marginal models here

% Specify models depending on copula specifications estimated previously
CopulaModels = {'CCC_norm', 'CCC_t', 'CCC_norm_empirical', ...
                'CCC_t_empirical',  ...
                %'DCC_norm', ...
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



%% Rolling-window estimation of the copula
clearvars -except R assets weightMat NumberWorkers dates Hsim Msim ...
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




%% Simulation of H-step ahead portfolio VaR and ES from copula models
clearvars -except R assets MarginalModels weightMat NumberWorkers dates ...
                  EstOut Hsim Msim MarginalModels CopulaModels

% Load all combinations of univariate variance models, assets, and copulas
EstOut = read_copula_est_results(MarginalModels,CopulaModels,assets);
% Check warnings, some specs are supposed to be missing, e.g. the empirical 
% distribution should only feature normal marginals and a Gaussian Copula!

% Simulate H-step ahead returns for all models in EstOut and compute Var/ES
alpha = [0.01, 0.025];
VaRandES = simulate_all_var_es(EstOut, R, weightMat, alpha, ...
                               'H', Hsim, 'M', Msim, ...
                               'NumWorkers', NumberWorkers, ...
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

ReadName     = ['VaRandES_0025_EquallyWeighted_' strjoin(assets, '_')];
ReadNamePath = sprintf('Output/VaRandES/%s.mat', ReadName);
load(ReadNamePath);


%%%% Fissler-Ziegel loss function for model comparison

% Fissler-Ziegel (FZ) loss and Model Confidence Set of 1% VaR and ES
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



%% Monte Carlo size check for backtesting functions

T_sim      = 5000;
omega_true = 0.02;
alpha_true = 0.05;
beta_true  = 0.93;
mu_true    = 0;

NumberWorkers = 4;

% Estimate GARCH-Normal in our workflow
WindLength_mc = 1000;
reest_freq_mc = 21;
assets_mc     = {'SimAsset'};
dates_mc      = (1:T_sim)';   % integer dates
alpha_mc      = 0.05;

B              = 500;
pValues_UC     = NaN(B, 1);
pValues_UC_bin = NaN(B, 1);
pValues_DQ     = NaN(B, 1);
pValues_ES     = NaN(B, 1);
tstat_uc     = NaN(B, 1);

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

    % Full pipeline
    MargEst_mc = estimate_garch(r_sim, 'dist', 'norm', ...
        'WindLength', WindLength_mc, 'ReestFreq', reest_freq_mc, ...
        'assets', assets_mc, 'dates', dates_mc, 'SaveDisk', false, ...
        'NumWorkers', NumberWorkers, 'RiskMetrics', true);
    CopEst_mc  = estimate_copula(MargEst_mc, 'SaveDisk', false);
    SimOut_mc  = simulate_return(CopEst_mc, r_sim, 'H', 1, 'M', 10000, ...
                                 'SaveDisk', false, ...
                                 'NumWorkers', NumberWorkers);
    SimOut_mc.SimAsset   = SimOut_mc;
    SimOut_mc.ModelNames = {'SimAsset'};

    VaRandES_mc = compute_var_es(SimOut_mc, 1, alpha_mc);

    % Collect p-values
    [pUC, pDQ, pES, pUCbinomial, tuc]  = var_uc_test(VaRandES_mc, r_sim);
    pValues_UC(b) = pUC;
    pValues_UC_bin(b) = pUCbinomial;
    pValues_DQ(b) = pDQ;
    pValues_ES(b) = pES;
tstat_uc(b) = tuc;
    fprintf('Replication %d/%d done\n', b, B);

    % Empirical size at 5% level
    fprintf('UC test size:     %.3f\n', mean(pValues_UC(1:b)     < 0.05));
    fprintf('UC Bin test size: %.3f\n', mean(pValues_UC_bin(1:b) < 0.05));
    fprintf('DQ test size:     %.3f\n', mean(pValues_DQ(1:b)     < 0.05));
    fprintf('ES test size:     %.3f\n', mean(pValues_ES(1:b)     < 0.05));
end

% Empirical size at 5% level
fprintf('UC test size:     %.3f\n', mean(pValues_UC     < 0.05));
fprintf('UC Bin test size: %.3f\n', mean(pValues_UC_bin < 0.05));
fprintf('DQ test size:     %.3f\n', mean(pValues_DQ     < 0.05));
fprintf('ES test size:     %.3f\n', mean(pValues_ES     < 0.05));




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







%% Monte Carlo size check for backtesting functions
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