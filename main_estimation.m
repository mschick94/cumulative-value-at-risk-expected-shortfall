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
assets    = {'AXP', 'BA'};
assets_q  = strcat("'", assets, "'");
dates     = ReturnData.Var1;
R         = table2array(ReturnData(:, assets_q));
weightMat = ones(max(size(assets)), 1) / max(size(assets));

% Rolling-window setting
reest_freq = 500;   % re-estimate every 21 observations
WindLength = 1000;  % Window length

% Number of workers for parallel computing 
NumberWorkers = 1;



%% Rolling-window estimation of the marginals

% GARCH-Normal
estimate_garch(R, 'dist', 'norm', 'WindLength', WindLength, ...
                  'ReestFreq', reest_freq, 'assets', assets, ...
                  'NumWorkers', NumberWorkers, 'dates', dates);

% GARCH-t
estimate_garch(R, 'dist', 't', 'WindLength', WindLength, ...
                  'ReestFreq', reest_freq, 'assets', assets, ...
                  'NumWorkers', NumberWorkers, 'dates', dates);

% % GARCH-Skew-t
% estimate_garch(R, 'dist', 'skewt', 'WindLength', WindLength, ...
%                   'ReestFreq', reest_freq, 'assets', assets, ...
%                   'NumWorkers', NumberWorkers, 'dates', dates);
% 
% % GARCH-Laplace
% estimate_garch(R, 'dist', 'laplace', 'WindLength', WindLength, ...
%                   'ReestFreq', reest_freq, 'assets', assets, ...
%                   'NumWorkers', NumberWorkers, 'dates', dates);
% 
% 
% % GJR-GARCH-Normal
% estimate_gjr_garch(R, 'dist', 'norm', 'WindLength', WindLength, ...
%                       'ReestFreq', reest_freq, 'assets', assets, ...
%                       'NumWorkers', NumberWorkers, 'dates', dates);




%% Rolling-window estimation of the copula
clearvars -except R assets weightMat NumberWorkers

% Specify models depending on marginal specifications estimated previously
MarginalModels = {'GARCH_norm', ...
                  'GARCH_t', ...
                  %'GARCH_skewt', ...
                  %'GARCH_laplace', ...
                  %'GJRGARCH_norm'
                  }; % Add marginal models here

% Load all combinations of univariate variance models and assets
EstOut = read_marg_est_results(MarginalModels,assets);


% GARCH(1,1)-Normal and Gaussian Copula with CCC matrix
estimate_copula(EstOut.GARCH_norm, 'copula_dist', 'norm', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);

% GARCH(1,1)-t and Gaussian Copula with CCC matrix
estimate_copula(EstOut.GARCH_t, 'copula_dist', 'norm', ...
                'corr_model', 'CCC', 'NumWorkers', NumberWorkers);


% % GARCH(1,1)-Normal and Student-t Copula with CCC matrix
% estimate_copula(EstOut.GARCH_norm, 'copula_dist', 't', ...
%                 'corr_model', 'CCC', 'NumWorkers', NumberWorkers);
% 
% % GARCH(1,1)-Normal and Gaussian Copula with DCC matrix
% estimate_copula(EstOut.GARCH_norm, 'copula_dist', 'norm', ...
%                 'corr_model', 'DCC', 'NumWorkers', NumberWorkers);
% 
% % GARCH(1,1)-Normal and Student-t Copula with DCC matrix
% estimate_copula(EstOut.GARCH_norm, 'copula_dist', 't', ...
%                 'corr_model', 'DCC', 'NumWorkers', NumberWorkers);
% 
% % GARCH(1,1)-Skew-t and Gaussian Copula with CCC matrix
% estimate_copula(EstOut.GARCH_skewt, 'copula_dist', 'norm', ...
%                 'corr_model', 'CCC', 'NumWorkers', NumberWorkers);
% 
% % GARCH(1,1)-Laplace and Gaussian Copula with CCC matrix
% estimate_copula(EstOut.GARCH_laplace, 'copula_dist', 'norm', ...
%                 'corr_model', 'CCC', 'NumWorkers', NumberWorkers);
% 
% % GARCH(1,1)-Empirical and Gaussian Copula with CCC matrix
% estimate_copula(EstOut.GARCH_norm, 'copula_dist', 'norm', ...
%                 'corr_model', 'CCC', 'empirical_pits', true, ...
%                 'NumWorkers', NumberWorkers);
% 
% % GJRGARCH(1,1)-Normal and Gaussian Copula with CCC matrix
% estimate_copula(EstOut.GJRGARCH_norm, 'copula_dist', 'norm', ...
%                 'corr_model', 'CCC', 'NumWorkers', NumberWorkers);



%% Simulation of H-step ahead forecast distribution 
clearvars -except R assets MarginalModels weightMat NumberWorkers

% Specify models depending on copula specifications estimated previously
CopulaModels = {'CCC_norm', ...
                %'CCC_norm_empirical', ...
                %'CCC_t', ...
                %'DCC_norm', ...
                %'DCC_t'
                }; % Add Copula models here 

% Simulation set-up
Hsim = 10;
Msim = 200;


% Load all combinations of univariate variance models, assets, and copulas
EstOut = read_copula_est_results(MarginalModels,CopulaModels,assets);


simulate_return(EstOut.GARCH_norm_CCC_norm, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);
simulate_return(EstOut.GARCH_t_CCC_norm, R, 'H', Hsim, 'M', Msim, ...
                'NumWorkers', NumberWorkers);


% simulate_return(EstOut.GARCH_skewt_CCC_norm,R,'H',Hsim,'M',Msim,'NumWorkers', NumberWorkers)
% simulate_return(EstOut.GARCH_norm_CCC_t,R,'H',Hsim,'M',Msim,'NumWorkers', NumberWorkers)
% simulate_return(EstOut.GARCH_norm_CCC_norm_empirical,R,'H',Hsim,'M',Msim,'NumWorkers', NumberWorkers)
% simulate_return(EstOut.GJRGARCH_norm_CCC_norm,R,'H',Hsim,'M',Msim,'NumWorkers', NumberWorkers)
% simulate_return(EstOut.GARCH_norm_DCC_norm,R,'H',Hsim,'M',Msim,'NumWorkers', NumberWorkers)
% simulate_return(EstOut.GARCH_norm_DCC_t,R,'H',Hsim,'M',Msim,'NumWorkers', NumberWorkers)



% Shut down parallel pool
if NumberWorkers > 1
    pool = gcp('nocreate');
    delete(pool);
end



%% Compute Portfolio VaR and ES
clearvars -except R assets MarginalModels weightMat NumberWorkers

% Specify models depending on copula specifications estimated previously
CopulaModels = {'CCC_norm'}; % Add models here 

% Load simulation results of all variance models, assets, and copulas
SimOut = read_sim_results(MarginalModels, CopulaModels, assets);

% Specify alpha-quantile
alpha = 0.05;

VaRandES = compute_var_es(SimOut, weightMat, alpha);



%% Forecast evaluation

score_fz(VaRandES, R, 'HEval', 10, 'DateStart', 20060201, ...
                                   'DateEnd', 20221230)



