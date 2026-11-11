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
K = 50;

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
    RV = RVData.RCOV_CTC_vech_30;
elseif K == 50
    RV = RVData.RCOV_CTC_vech_50;
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
MarginalModels = {'HEAVY_norm', 'HEAVY_t', ...
                  }; % Add marginal models here

% Specify models depending on copula specifications estimated previously
CopulaModels = {'CCC_norm', 'CCC_t', ...
                % 'DCC_norm_empirical', ...
                % 'DCC_t'
                }; % Add Copula models here 



%% Rolling-window estimation of marginals and copula

%%%% GARCH(1,1) with various distributions and CCC Copula

%%% HEAVY-Normal 
EstOut = estimate_heavy(R, RV, 'dist', 'norm', 'WindLength', WindLength, ...
                        'ReestFreq', reest_freq, 'assets', assets, ...
                        'NumWorkers', NumberWorkers, 'dates', dates);

% Gaussian-CCC
estimate_copula(EstOut, 'copula_dist', 'norm',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);

% t-CCC 
estimate_copula(EstOut, 'copula_dist', 't',  'corr_model', 'CCC', ...
                'NumWorkers', NumberWorkers);


%%% HEAVY-t
EstOut = estimate_heavy(R, RV, 'dist', 't', 'WindLength', WindLength, ...
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
                               'OutputName', 'EquallyWeighted_HEAVY');


