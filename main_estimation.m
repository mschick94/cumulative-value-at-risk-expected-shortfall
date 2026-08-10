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

% Specify assets and extract data
assets    = {'AXP', 'BA', 'CAT', 'GE', 'HD', 'HON', 'IBM', 'JPM', 'KO', ...
              'MCD'};
assets_q  = strcat("'", assets, "'");
dates     = ReturnData.Var1;
R         = table2array(ReturnData(:, assets_q));

% Equally weighted portfolio
weightMat = ones(max(size(assets)), 1) / max(size(assets));

% Rolling-window setting
reest_freq = 21;    % re-estimate every 21 observations
WindLength = 1000;  % Window length

% Simulation set-up
Hsim = 10;
Msim = 1000;

% Number of workers for parallel computing 
NumberWorkers = 4;



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
clearvars -except R assets weightMat NumberWorkers dates Hsim Msim

% Specify models depending on marginal specifications estimated previously
MarginalModels = {'GARCH_norm', 'GARCH_t', 'GARCH_skewt', ...
                  'GARCH_laplace', 'RiskMetrics_GARCH_norm', ...
                  'GJRGARCH_norm', 'GJRGARCH_t', 'GJRGARCH_skewt', ...
                  'GJRGARCH_laplace'}; % Add marginal models here

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




%% Simulation of H-step ahead forecast distribution 
clearvars -except R assets MarginalModels weightMat NumberWorkers dates ...
                  Hsim Msim

% Specify models depending on copula specifications estimated previously
CopulaModels = {'CCC_norm', 'CCC_t', 'CCC_norm_empirical', ...
                'CCC_t_empirical',  ...
                %'DCC_norm', ...
                %'DCC_t'
                }; % Add Copula models here 


% Load all combinations of univariate variance models, assets, and copulas
EstOut = read_copula_est_results(MarginalModels,CopulaModels,assets);
% Check warnings, some specs are supposed to be missing, e.g. the empirical 
% distribution should only feature normal marginals and a Gaussian Copula!

% GARCH specifications
simulate_return(EstOut.GARCH_norm_CCC_norm, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);
simulate_return(EstOut.GARCH_t_CCC_norm, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);
simulate_return(EstOut.GARCH_skewt_CCC_norm, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);
simulate_return(EstOut.GARCH_laplace_CCC_norm, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);
simulate_return(EstOut.GARCH_norm_CCC_norm_empirical, R, 'H', Hsim, ...
                'M', Msim, 'NumWorkers', NumberWorkers);
simulate_return(EstOut.GARCH_norm_CCC_t, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);
simulate_return(EstOut.GARCH_t_CCC_t, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);
simulate_return(EstOut.GARCH_skewt_CCC_t, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);
simulate_return(EstOut.GARCH_laplace_CCC_t, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);


% GJR-GARCH specifications
simulate_return(EstOut.GJRGARCH_norm_CCC_norm, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);
simulate_return(EstOut.GJRGARCH_t_CCC_norm, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);
simulate_return(EstOut.GJRGARCH_skewt_CCC_norm, R, 'H', Hsim, ...
                'M', Msim, 'NumWorkers', NumberWorkers);
simulate_return(EstOut.GJRGARCH_laplace_CCC_norm, R, 'H', Hsim, ...
                'M', Msim, 'NumWorkers', NumberWorkers);
simulate_return(EstOut.GJRGARCH_norm_CCC_norm_empirical, R, 'H', Hsim, ...
                'M', Msim, 'NumWorkers', NumberWorkers);
simulate_return(EstOut.GJRGARCH_norm_CCC_t, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);
simulate_return(EstOut.GJRGARCH_t_CCC_t, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);
simulate_return(EstOut.GJRGARCH_skewt_CCC_t, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);
simulate_return(EstOut.GJRGARCH_laplace_CCC_t, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);


% RiskMetrics
simulate_return(EstOut.RiskMetrics_GARCH_norm_CCC_norm, R, 'H', Hsim, ...
                'M', Msim, 'NumWorkers', NumberWorkers);
simulate_return(EstOut.RiskMetrics_GARCH_norm_CCC_t, R, 'H', Hsim, ...
                'M', Msim, 'NumWorkers', NumberWorkers);



%% Compute Portfolio VaR and ES from copula models
clearvars -except R assets MarginalModels weightMat NumberWorkers dates ...
                  Hsim Msim

% Specify models depending on copula specifications estimated previously
CopulaModels = {'CCC_norm', 'CCC_t', 'CCC_norm_empirical', ...
                'CCC_t_empirical'}; 

% Load simulation results of all variance models, assets, and copulas
SimOut = read_sim_results(MarginalModels, CopulaModels, assets);
% Check warnings, some specs are supposed to be missing, e.g. the empirical 
% distribution should only feature normal marginals and a Gaussian Copula 
% based on a GARCH and no GJR-GARCH!

% Specify alpha-quantile
alpha = 0.01;
VaRandES_0010 = compute_var_es(SimOut, weightMat, alpha);

alpha = 0.025;
VaRandES_0025 = compute_var_es(SimOut, weightMat, alpha);



%% GARCH on equally weighted portfolio
R_pf  = R * weightMat;
dists = {'norm', 't', 'skewt', 'laplace'};

% GARCH benchmarks
for d = 1:length(dists)
    [VaRandES_0010, VaRandES_0025] = pf_garch_benchmark(dists{d}, ...
        R_pf, VaRandES_0010, VaRandES_0025, VaRandES_0010.WindLength, ...
        VaRandES_0010.ReestFreq, assets, dates, NumberWorkers, Hsim, Msim);
end

% GARCH empirical
[VaRandES_0010, VaRandES_0025] = pf_garch_benchmark('norm', ...
    R_pf, VaRandES_0010, VaRandES_0025, VaRandES_0010.WindLength, ...
    VaRandES_0010.ReestFreq, assets, dates, NumberWorkers, Hsim, Msim, ...
    'EmpiricalPits', true);

% GJR-GARCH benchmarks
for d = 1:length(dists)
    [VaRandES_0010, VaRandES_0025] = pf_garch_benchmark(dists{d}, ...
        R_pf, VaRandES_0010, VaRandES_0025, VaRandES_0010.WindLength, ...
        VaRandES_0010.ReestFreq, assets, dates, NumberWorkers, Hsim, ...
        Msim, 'GJR', true);
end

% GJR-GARCH empirical
[VaRandES_0010, VaRandES_0025] = pf_garch_benchmark('norm', ...
    R_pf, VaRandES_0010, VaRandES_0025, VaRandES_0010.WindLength, ...
    VaRandES_0010.ReestFreq, assets, dates, NumberWorkers, Hsim, Msim, ...
    'EmpiricalPits', true, 'GJR', true);


% % Shut down parallel pool
% if NumberWorkers > 1
%     pool = gcp('nocreate');
%     delete(pool);
% end

% R_pf = R * weightMat;
% PFweight = 1;
% 
% %%% GARCH-Normal
% EqualWeightedPF_marginal = estimate_garch(R_pf, 'dist', 'norm', ...
%     'WindLength', VaRandES_0010.WindLength, 'ReestFreq', ...
%     VaRandES_0010.ReestFreq, 'assets', assets, 'NumWorkers', ...
%     NumberWorkers, 'dates', dates, 'Portfolio', 'EqualWeighted', ...
%     'SaveDisk', false);
% 
% % K = 1 copula needed for simulation of returns
% GARCH_norm_EqualWeightedPF = estimate_copula(EqualWeightedPF_marginal, ...
%                                              'SaveDisk', false);
% 
% % Simulate return distribution
% EqualWeightedPF = simulate_return(GARCH_norm_EqualWeightedPF, R_pf, ...
%     'H', Hsim, 'M', Msim, 'NumWorkers', NumberWorkers, 'SaveDisk', false);
% EqualWeightedPF.EqualWeightedPF = EqualWeightedPF;
% EqualWeightedPF.ModelNames = {'EqualWeightedPF'};
% 
% % Compute equally weighted portfolio 1% VaR and ES
% alpha = 0.01;
% EqualWeightedPFVaRandES = compute_var_es(EqualWeightedPF, PFweight, ...
%                                          alpha);
% 
% % Add equally weighted portfolio VaR and ES to evaluation structure
% VaRandES_0010.VaR    = cat(2, VaRandES_0010.VaR, EqualWeightedPFVaRandES.VaR);
% VaRandES_0010.ES     = cat(2, VaRandES_0010.ES,  EqualWeightedPFVaRandES.ES);
% VaRandES_0010.Models = [VaRandES_0010.Models; {'EqualWeightedPF_GARCH_norm'}];
% 
% % Compute equally weighted portfolio 2.5% VaR and ES
% alpha = 0.025;
% EqualWeightedPFVaRandES = compute_var_es(EqualWeightedPF, PFweight, ...
%                                          alpha);
% 
% % Add equally weighted portfolio VaR and ES to evaluation structure
% VaRandES_0025.VaR    = cat(2, VaRandES_0025.VaR, EqualWeightedPFVaRandES.VaR);
% VaRandES_0025.ES     = cat(2, VaRandES_0025.ES,  EqualWeightedPFVaRandES.ES);
% VaRandES_0025.Models = [VaRandES_0025.Models; {'EqualWeightedPF_GARCH_norm'}];
% 
% 
% %%% GARCH-t
% EqualWeightedPF_marginal = estimate_garch(R_pf, 'dist', 't', ...
%     'WindLength', VaRandES_0010.WindLength, 'ReestFreq', ...
%     VaRandES_0010.ReestFreq, 'assets', assets, 'NumWorkers', ...
%     NumberWorkers, 'dates', dates, 'Portfolio', 'EqualWeighted', ...
%     'SaveDisk', false);
% 
% % K = 1 copula needed for simulation of returns
% GARCH_norm_EqualWeightedPF = estimate_copula(EqualWeightedPF_marginal, ...
%                                              'SaveDisk', false);
% 
% % Simulate return distribution
% EqualWeightedPF = simulate_return(GARCH_norm_EqualWeightedPF, R_pf, ...
%     'H', Hsim, 'M', Msim, 'NumWorkers', NumberWorkers, 'SaveDisk', false);
% EqualWeightedPF.EqualWeightedPF = EqualWeightedPF;
% EqualWeightedPF.ModelNames = {'EqualWeightedPF'};
% 
% % Compute equally weighted portfolio 1% VaR and ES
% alpha = 0.01;
% EqualWeightedPFVaRandES = compute_var_es(EqualWeightedPF, PFweight, ...
%                                          alpha);
% 
% % Add equally weighted portfolio VaR and ES to evaluation structure
% VaRandES_0010.VaR    = cat(2, VaRandES_0010.VaR, EqualWeightedPFVaRandES.VaR);
% VaRandES_0010.ES     = cat(2, VaRandES_0010.ES,  EqualWeightedPFVaRandES.ES);
% VaRandES_0010.Models = [VaRandES_0010.Models; {'EqualWeightedPF_GARCH_t'}];
% 
% % Compute equally weighted portfolio 2.5% VaR and ES
% alpha = 0.025;
% EqualWeightedPFVaRandES = compute_var_es(EqualWeightedPF, PFweight, ...
%                                          alpha);
% 
% % Add equally weighted portfolio VaR and ES to evaluation structure
% VaRandES_0025.VaR    = cat(2, VaRandES_0025.VaR, EqualWeightedPFVaRandES.VaR);
% VaRandES_0025.ES     = cat(2, VaRandES_0025.ES,  EqualWeightedPFVaRandES.ES);
% VaRandES_0025.Models = [VaRandES_0025.Models; {'EqualWeightedPF_GARCH_t'}];
% 
% 
% %%% GARCH-Skew-t
% EqualWeightedPF_marginal = estimate_garch(R_pf, 'dist', 'skewt', ...
%     'WindLength', VaRandES_0010.WindLength, 'ReestFreq', ...
%     VaRandES_0010.ReestFreq, 'assets', assets, 'NumWorkers', ...
%     NumberWorkers, 'dates', dates, 'Portfolio', 'EqualWeighted', ...
%     'SaveDisk', false);
% 
% % K = 1 copula needed for simulation of returns
% GARCH_norm_EqualWeightedPF = estimate_copula(EqualWeightedPF_marginal, ...
%                                              'SaveDisk', false);
% 
% % Simulate return distribution
% EqualWeightedPF = simulate_return(GARCH_norm_EqualWeightedPF, R_pf, ...
%     'H', Hsim, 'M', Msim, 'NumWorkers', NumberWorkers, 'SaveDisk', false);
% EqualWeightedPF.EqualWeightedPF = EqualWeightedPF;
% EqualWeightedPF.ModelNames = {'EqualWeightedPF'};
% 
% % Compute equally weighted portfolio 1% VaR and ES
% alpha = 0.01;
% EqualWeightedPFVaRandES = compute_var_es(EqualWeightedPF, PFweight, ...
%                                          alpha);
% 
% % Add equally weighted portfolio VaR and ES to evaluation structure
% VaRandES_0010.VaR    = cat(2, VaRandES_0010.VaR, EqualWeightedPFVaRandES.VaR);
% VaRandES_0010.ES     = cat(2, VaRandES_0010.ES,  EqualWeightedPFVaRandES.ES);
% VaRandES_0010.Models = [VaRandES_0010.Models; {'EqualWeightedPF_GARCH_skewt'}];
% 
% % Compute equally weighted portfolio 2.5% VaR and ES
% alpha = 0.025;
% EqualWeightedPFVaRandES = compute_var_es(EqualWeightedPF, PFweight, ...
%                                          alpha);
% 
% % Add equally weighted portfolio VaR and ES to evaluation structure
% VaRandES_0025.VaR    = cat(2, VaRandES_0025.VaR, EqualWeightedPFVaRandES.VaR);
% VaRandES_0025.ES     = cat(2, VaRandES_0025.ES,  EqualWeightedPFVaRandES.ES);
% VaRandES_0025.Models = [VaRandES_0025.Models; {'EqualWeightedPF_GARCH_skewt'}];
% 
% 
% %%% GARCH-Laplace
% EqualWeightedPF_marginal = estimate_garch(R_pf, 'dist', 'laplace', ...
%     'WindLength', VaRandES_0010.WindLength, 'ReestFreq', ...
%     VaRandES_0010.ReestFreq, 'assets', assets, 'NumWorkers', ...
%     NumberWorkers, 'dates', dates, 'Portfolio', 'EqualWeighted', ...
%     'SaveDisk', false);
% 
% % K = 1 copula needed for simulation of returns
% GARCH_norm_EqualWeightedPF = estimate_copula(EqualWeightedPF_marginal, ...
%                                              'SaveDisk', false);
% 
% % Simulate return distribution
% EqualWeightedPF = simulate_return(GARCH_norm_EqualWeightedPF, R_pf, ...
%     'H', Hsim, 'M', Msim, 'NumWorkers', NumberWorkers, 'SaveDisk', false);
% EqualWeightedPF.EqualWeightedPF = EqualWeightedPF;
% EqualWeightedPF.ModelNames = {'EqualWeightedPF'};
% 
% % Compute equally weighted portfolio 1% VaR and ES
% alpha = 0.01;
% EqualWeightedPFVaRandES = compute_var_es(EqualWeightedPF, PFweight, ...
%                                          alpha);
% 
% % Add equally weighted portfolio VaR and ES to evaluation structure
% VaRandES_0010.VaR    = cat(2, VaRandES_0010.VaR, EqualWeightedPFVaRandES.VaR);
% VaRandES_0010.ES     = cat(2, VaRandES_0010.ES,  EqualWeightedPFVaRandES.ES);
% VaRandES_0010.Models = [VaRandES_0010.Models; {'EqualWeightedPF_GARCH_laplace'}];
% 
% % Compute equally weighted portfolio 2.5% VaR and ES
% alpha = 0.025;
% EqualWeightedPFVaRandES = compute_var_es(EqualWeightedPF, PFweight, ...
%                                          alpha);
% 
% % Add equally weighted portfolio VaR and ES to evaluation structure
% VaRandES_0025.VaR    = cat(2, VaRandES_0025.VaR, EqualWeightedPFVaRandES.VaR);
% VaRandES_0025.ES     = cat(2, VaRandES_0025.ES,  EqualWeightedPFVaRandES.ES);
% VaRandES_0025.Models = [VaRandES_0025.Models; {'EqualWeightedPF_GARCH_laplace'}];
% 
% 
% %%% GARCH-empirical
% EqualWeightedPF_marginal = estimate_garch(R_pf, 'dist', 'norm', ...
%     'WindLength', VaRandES_0010.WindLength, 'ReestFreq', ...
%     VaRandES_0010.ReestFreq, 'assets', assets, 'NumWorkers', ...
%     NumberWorkers, 'dates', dates, 'Portfolio', 'EqualWeighted', ...
%     'SaveDisk', false);
% 
% % K = 1 copula needed for simulation of returns
% GARCH_norm_EqualWeightedPF = estimate_copula(EqualWeightedPF_marginal, ...
%                                              'empirical_pits', true, ...
%                                              'SaveDisk', false);
% 
% % Simulate return distribution
% EqualWeightedPF = simulate_return(GARCH_norm_EqualWeightedPF, R_pf, ...
%     'H', Hsim, 'M', Msim, 'NumWorkers', NumberWorkers, 'SaveDisk', false);
% EqualWeightedPF.EqualWeightedPF = EqualWeightedPF;
% EqualWeightedPF.ModelNames = {'EqualWeightedPF'};
% 
% % Compute equally weighted portfolio 1% VaR and ES
% alpha = 0.01;
% EqualWeightedPFVaRandES = compute_var_es(EqualWeightedPF, PFweight, ...
%                                          alpha);
% 
% % Add equally weighted portfolio VaR and ES to evaluation structure
% VaRandES_0010.VaR    = cat(2, VaRandES_0010.VaR, EqualWeightedPFVaRandES.VaR);
% VaRandES_0010.ES     = cat(2, VaRandES_0010.ES,  EqualWeightedPFVaRandES.ES);
% VaRandES_0010.Models = [VaRandES_0010.Models; {'EqualWeightedPF_GARCH_empirical'}];
% 
% % Compute equally weighted portfolio 2.5% VaR and ES
% alpha = 0.025;
% EqualWeightedPFVaRandES = compute_var_es(EqualWeightedPF, PFweight, ...
%                                          alpha);
% 
% % Add equally weighted portfolio VaR and ES to evaluation structure
% VaRandES_0025.VaR    = cat(2, VaRandES_0025.VaR, EqualWeightedPFVaRandES.VaR);
% VaRandES_0025.ES     = cat(2, VaRandES_0025.ES,  EqualWeightedPFVaRandES.ES);
% VaRandES_0025.Models = [VaRandES_0025.Models; {'EqualWeightedPF_GARCH_empirical'}];





%% Add additional forecasts from other models or weighting schemes here!

% Store VaaR and ES forecasts
VaRandESName = ['VaRandES_0010_Equally_Weighted_' strjoin(assets, '_')];
VaRandESpath = sprintf('Output/VaRandES/%s.mat', VaRandESName);
save(VaRandESpath, 'VaRandES_0010');

VaRandESName = ['VaRandES_0025_Equally_Weighted_' strjoin(assets, '_')];
VaRandESpath = sprintf('Output/VaRandES/%s.mat', VaRandESName);
save(VaRandESpath, 'VaRandES_0025');



%% Forecast evaluation
clearvars -except R VaRandESpath assets

% Read in VaR and ES forecasts per model
ReadName     = ['VaRandES_0010_Equally_Weighted_' strjoin(assets, '_')];
ReadNamePath = sprintf('Output/VaRandES/%s.mat', ReadName);
load(ReadNamePath);

ReadName     = ['VaRandES_0025_Equally_Weighted_' strjoin(assets, '_')];
ReadNamePath = sprintf('Output/VaRandES/%s.mat', ReadName);
load(ReadNamePath);


% Fissler-Ziegel (FZ) loss and Model Confidence Set of 1% VaR and ES
MCSTable_0010 = score_fz(VaRandES_0010, R, 'HEval', 10, ...
                         'DateStart', 20060201, 'DateEnd',   20221230, ...
                         'Decimals',  3);

% Fissler-Ziegel (FZ) loss and Model Confidence Set of 2.5% VaR and ES
MCSTable_0025 = score_fz(VaRandES_0025, R, 'HEval', 10, ...
                         'DateStart', 20060201, 'DateEnd',   20221230, ...
                         'Decimals',  3);
