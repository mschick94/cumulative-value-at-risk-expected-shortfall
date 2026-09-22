%% Estimation of portfolio h-step-ahead VaR and ES - Copula specifications

% Change directory
cd 'C:/Users/Schick/Documents/Forschung/8_Portfolio_Var_ES_hstep'

close all
clc

% Directories
addpath('Data')
addpath(genpath('Code'));


%% Setup

% Load return data
clear
ReturnData = readtable('CTC_RET.xlsx','VariableNamingRule','preserve');
RVData     = load('Start_data_10_30_50_2002_2023.mat');

% K = 10, 30, or 50
K = 10;

% Specify assets (manually or by position in ReturnData) and extract data
% assets = {'AXP', 'BA'}; % Manually selecting assets by name
VarNames = ReturnData.Properties.VariableNames;
VarNames = VarNames(2:K+1);
assets = cellfun(@(x) x(2:end-1), VarNames, 'UniformOutput', false);
assets_q = strcat("'", assets, "'");
dates    = ReturnData.Var1;
R        = table2array(ReturnData(:, assets_q));

% Realized covariance matrices for HEAVY-model
if K == 10
    RV = RVData.RCOV_CTC_vech_10;
elseif K == 30
    RV = RVData.RCOV_CTC_vech_20;
elseif K == 50
    RV = RVData.RCOV_CTC_vech_30;
end

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
                  'GJRGARCH_laplace', ... % 'HEAVY_laplace'
                  }; % Add marginal models here

% Specify models depending on copula specifications estimated previously
CopulaModels = {'CCC_norm', 'CCC_t', 'CCC_norm_empirical', ...
                'CCC_t_empirical',  ...
                % 'DCC_norm_empirical', ...
                % 'DCC_t'
                }; % Add Copula models here 



%% Rolling-window estimation of marginals and copula

%%%% GARCH(1,1) with various distributions and CCC Copula

%%% GARCH-Normal 
EstOut = estimate_garch(R, 'dist', 'norm', 'WindLength', WindLength, ...
                        'ReestFreq', reest_freq, 'assets', assets, ...
                        'NumWorkers', NumberWorkers, 'dates', dates);

% Gaussian-CCC
estimate_copula(EstOut, 'copula_dist', 'norm',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);

% t-CCC 
estimate_copula(EstOut, 'copula_dist', 't',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);

% Empirical 
estimate_copula(EstOut, 'copula_dist', 'norm', 'corr_model', 'CCC', ...
                'empirical_pits', true, 'NumWorkers', NumberWorkers);


%%% GARCH-t
EstOut = estimate_garch(R, 'dist', 't', 'WindLength', WindLength, ...
                        'ReestFreq', reest_freq, 'assets', assets, ...
                        'NumWorkers', NumberWorkers, 'dates', dates);

% Gaussian-CCC
estimate_copula(EstOut, 'copula_dist', 'norm',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);

% t-CCC 
estimate_copula(EstOut, 'copula_dist', 't',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);

%%% GARCH-Skew-t
EstOut = estimate_garch(R, 'dist', 'skewt', 'WindLength', WindLength, ...
                        'ReestFreq', reest_freq, 'assets', assets, ...
                        'NumWorkers', NumberWorkers, 'dates', dates);

% Gaussian-CCC
estimate_copula(EstOut, 'copula_dist', 'norm',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);

% t-CCC 
estimate_copula(EstOut, 'copula_dist', 't',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);


% GARCH-Laplace
EstOut = estimate_garch(R, 'dist', 'laplace', 'WindLength', WindLength, ...
                        'ReestFreq', reest_freq, 'assets', assets, ...
                        'NumWorkers', NumberWorkers, 'dates', dates);

% Gaussian-CCC
estimate_copula(EstOut, 'copula_dist', 'norm',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);

% t-CCC 
estimate_copula(EstOut, 'copula_dist', 't',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);


%%% RiskMetrics with Gaussian and Student-t CCC Copula
EstOut = estimate_garch(R, 'dist', 'norm', 'WindLength', WindLength, ...
                        'ReestFreq', reest_freq, 'assets', assets, ...
                        'NumWorkers', NumberWorkers, 'dates', dates, ...
                        'RiskMetrics', true);

% Gaussian-CCC
estimate_copula(EstOut, 'copula_dist', 'norm',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);

% t-CCC 
estimate_copula(EstOut, 'copula_dist', 't',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);



%%%% GJR-GARCH(1,1) with various distributions and CCC Copula

%%% GJR-GARCH-Normal 
EstOut = estimate_gjr_garch(R, 'dist', 'norm', ...
                            'WindLength', WindLength, ...
                            'ReestFreq', reest_freq, 'assets', assets, ...
                            'NumWorkers', NumberWorkers, 'dates', dates);

% Gaussian-CCC
estimate_copula(EstOut, 'copula_dist', 'norm',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);

% t-CCC 
estimate_copula(EstOut, 'copula_dist', 't',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);

% Empirical 
estimate_copula(EstOut, 'copula_dist', 'norm', 'corr_model', 'CCC', ...
                'empirical_pits', true, 'NumWorkers', NumberWorkers);


%%% GJR-GARCH-t
EstOut = estimate_gjr_garch(R, 'dist', 't', 'WindLength', WindLength, ...
                            'ReestFreq', reest_freq, 'assets', assets, ...
                            'NumWorkers', NumberWorkers, 'dates', dates);

% Gaussian-CCC
estimate_copula(EstOut, 'copula_dist', 'norm',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);

% t-CCC 
estimate_copula(EstOut, 'copula_dist', 't',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);

%%% GJR-GARCH-Skew-t
EstOut = estimate_gjr_garch(R, 'dist', 'skewt', ...
                            'WindLength', WindLength, ...
                            'ReestFreq', reest_freq, 'assets', assets, ...
                            'NumWorkers', NumberWorkers, 'dates', dates);

% Gaussian-CCC
estimate_copula(EstOut, 'copula_dist', 'norm',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);

% t-CCC 
estimate_copula(EstOut, 'copula_dist', 't',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);


% GJR-GARCH-Laplace
EstOut = estimate_gjr_garch(R, 'dist', 'laplace', ...
                            'WindLength', WindLength, ...
                            'ReestFreq', reest_freq, 'assets', assets, ...
                            'NumWorkers', NumberWorkers, 'dates', dates);

% Gaussian-CCC
estimate_copula(EstOut, 'copula_dist', 'norm',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);

% t-CCC 
estimate_copula(EstOut, 'copula_dist', 't',  'corr_model', 'CCC', ...
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
dists = {'norm', 't', 'skewt', 'laplace'};

% GARCH benchmarks
for d = 1:length(dists)
    VaRandES = pf_garch_benchmark(dists{d}, R, weightMat, VaRandES, ...
        VaRandES.WindLength, VaRandES.ReestFreq, assets, dates, ...
        NumberWorkers, Hsim, Msim);
end

% GARCH empirical
VaRandES = pf_garch_benchmark('norm', R, weightMat, VaRandES, ...
    VaRandES.WindLength, VaRandES.ReestFreq, assets, dates, ...
    NumberWorkers, Hsim, Msim, 'EmpiricalPits', true);

% GJR-GARCH benchmarks
for d = 1:length(dists)
    VaRandES = pf_garch_benchmark(dists{d}, R, weightMat, VaRandES, ...
        VaRandES.WindLength, VaRandES.ReestFreq, assets, dates, ...
        NumberWorkers, Hsim, Msim, 'GJR', true);
end

% GJR-GARCH empirical
VaRandES = pf_garch_benchmark('norm', R, weightMat, VaRandES, ...
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



%% Monte Carlo - GARCH-N
T_sim      = 5000;
omega_true = 0.01;
alpha_true = 0.03;
beta_true  = 0.95;
nu_true    = 5;

B     = 1000;
Msim  = 25000;
H_mc  = 10;
alpha = [0.01, 0.025];

% Estimate GARCH-Normal/t
WindLength_mc = 1000;
reest_freq_mc = 250;

% GARCH-N with estimated parameters
TruePars  = [omega_true, alpha_true, beta_true, nu_true];
BackTestSimulation(TruePars, 'T_sim', T_sim, 'M_sim', Msim, 'H', H_mc, ...
                   'B_sim', B, 'alpha', alpha, 'dist', 'norm', ...
                   'EstPars', true, 'WindLength_mc', WindLength_mc, ...
                   'reest_freq_mc', reest_freq_mc);

% GARCH-N with true parameters
TruePars  = [omega_true, alpha_true, beta_true, nu_true];
BackTestSimulation(TruePars, 'T_sim', T_sim, 'M_sim', Msim, 'H', H_mc, ...
                   'B_sim', B, 'alpha', alpha, 'dist', 'norm', ...
                   'EstPars', false, 'WindLength_mc', WindLength_mc, ...
                   'reest_freq_mc', reest_freq_mc);

% GARCH-t with estimated parameters
TruePars  = [omega_true, alpha_true, beta_true, nu_true];
BackTestSimulation(TruePars, 'T_sim', T_sim, 'M_sim', Msim, 'H', H_mc, ...
                   'B_sim', B, 'alpha', alpha, 'dist', 't', ...
                   'EstPars', true, 'WindLength_mc', WindLength_mc, ...
                   'reest_freq_mc', reest_freq_mc);

% GARCH-t with true parameters
TruePars  = [omega_true, alpha_true, beta_true, nu_true];
BackTestSimulation(TruePars, 'T_sim', T_sim, 'M_sim', Msim, 'H', H_mc, ...
                   'B_sim', B, 'alpha', alpha, 'dist', 't', ...
                   'EstPars', false, 'WindLength_mc', WindLength_mc, ...
                   'reest_freq_mc', reest_freq_mc);

