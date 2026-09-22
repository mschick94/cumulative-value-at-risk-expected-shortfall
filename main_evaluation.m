%% Estimation of portfolio h-step-ahead VaR and ES - Forecast Evaluation

% Change directory
cd 'C:/Users/Schick/Documents/Forschung/8_Portfolio_Var_ES_hstep'

close all
clc

% Directories
addpath(genpath('Code'));
addpath(genpath('Output'));


%% MC Simulation results

% Number of replications
B = 1000;
clc
% Report simulation results
BT_estpars  = BackTestSimTable(B);
BT_truepars = BackTestSimTable(B, 'EstPars', false);

% % Construct combined output for LaTeX
% Tab2 = BT_truepars.LaTeX;
% Tab2.Properties.VariableNames = strcat(Tab2.Properties.VariableNames,'_2');
% disp([BT_estpars.LaTeX(:,1:end-1) Tab2(:,2:end)])


%%


clearvars -except R assets

% Read in VaR and ES forecasts per model
clearvars -except R assets
load(sprintf('Output/VaRandES/VaRandES_EquallyWeighted_%s.mat', ...
             strjoin(assets, '_')));

% Prepare Result strucutre to be merged with other models and for FZ loss
VaRandES.VaR = squeeze(VaRandES.VaR); % Depends on what Anne does :)
VaRandES.ES  = squeeze(VaRandES.ES);


%%%% Fissler-Ziegel loss function for model comparison
ModelEval = [1, 5, 6, 9, 3, 21, 22, 23, 24, 25, 10];
ModelEval = [1, 5, 6, 9, 3, 21, 22, 23, 24, 25];

% Fissler-Ziegel (FZ) loss and Model Confidence Set of 1% VaR and ES
MCSTable_0010 = score_fz(VaRandES, R, 'HEval', 10, ...
                         'alpha_level', 0.01, 'Models_id', [], ...
                         'DateStart', 20060201, 'DateEnd', 20191230, ...
                         'Decimals', 3, 'PrintTable', false, ...
                         'MCSLevel', 0.10);
MCSTable_0010.LaTeX

% Fissler-Ziegel (FZ) loss and Model Confidence Set of 2.5% VaR and ES
MCSTable_0025 = score_fz(VaRandES, R, 'HEval', 10, ...
                         'alpha_level', 0.025, 'Models_id', [], ...
                         'DateStart', 20060201, 'DateEnd', 20191230, ...
                         'Decimals', 3, 'PrintTable', false, ...
                         'MCSLevel', 0.10);
MCSTable_0025.LaTeX














% %%%% Backtesting of VaR and ES
% var_uc_test(VaRandES_0010, R)
% 
% var_uc_test(VaRandES_0025, R)