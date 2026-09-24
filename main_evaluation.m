%% Estimation of portfolio h-step-ahead VaR and ES - Forecast Evaluation

% Change directory
cd 'C:/Users/Schick/Documents/Forschung/8_Portfolio_Var_ES_hstep'

close all
clc

% Directories
addpath('Data');
addpath(genpath('Code'));
addpath(genpath('Output'));

% Load return data
clear
ReturnData = readtable('CTC_RET.xlsx','VariableNamingRule','preserve');


%% MC Simulation results

% Number of replications
B = 1000;
clc

% GARCH-Normal and GARCH-t
BT_estpars_n = BackTestSimTable(B);
BT_estpars_t = BackTestSimTable(B, 'dist', 't');

% Construct combined output for LaTeX (ready for copy and paste)
Tab2 = BT_estpars_t.LaTeX;
Tab2.Properties.VariableNames = strcat(Tab2.Properties.VariableNames,'_2');
% disp([BT_estpars_n.LaTeX(:,1:end-1) Tab2(:,2:end)])



%% Load model forecasts for K

% Numbe of assets K = 10, 30, or 50
K = 10;

% Read in Returns as well as VaR and ES forecasts per model
VarNames = ReturnData.Properties.VariableNames;
VarNames = VarNames(2:K+1);
assets   = cellfun(@(x) x(2:end-1), VarNames, 'UniformOutput', false);
assets_q = strcat("'", assets, "'");
R        = table2array(ReturnData(:, assets_q));
load(sprintf('Output/VaRandES/VaRandES_EquallyWeighted_%s.mat', ...
             strjoin(assets, '_')));

% Load Annes forecasts 

% Prepare result strucutre to be merged with other models and for FZ loss
VaRandES.VaR = squeeze(VaRandES.VaR); % Depends on what Anne does :)
VaRandES.ES  = squeeze(VaRandES.ES);


% Dates excluded according to maximum loss differentials among all models
EvalDates_alpha_0010_q_0010 = score_fz(VaRandES, R, 'HEval', 10, ...
    'alpha_level', 0.01, 'Models_id', [], 'ExcludeQuant', 0.01, ...
    'DateStart', 20060201, 'DateEnd', 20231130, 'MCSTest', false);

EvalDates_alpha_0010_q_0005 = score_fz(VaRandES, R, 'HEval', 10, ...
    'alpha_level', 0.01, 'Models_id', [], 'ExcludeQuant', 0.005, ...
    'DateStart', 20060201, 'DateEnd', 20231130, 'MCSTest', false);

EvalDates_alpha_0025_q_0010 = score_fz(VaRandES, R, 'HEval', 10, ...
    'alpha_level', 0.025, 'Models_id', [], 'ExcludeQuant', 0.01, ...
    'DateStart', 20060201, 'DateEnd', 20231130, 'MCSTest', false);

EvalDates_alpha_0025_q_0005 = score_fz(VaRandES, R, 'HEval', 10, ...
    'alpha_level', 0.025, 'Models_id', [], 'ExcludeQuant', 0.005, ...
    'DateStart', 20060201, 'DateEnd', 20231130, 'MCSTest', false);



%% Fissler-Ziegel (FZ) loss and Model Confidence

% Specify models for forecast evaluation
ModelEval = [ ];

% MCS significance level
SigLevel = 0.10;


%%% All observations
% Fissler-Ziegel (FZ) loss and Model Confidence Set of 1% VaR and ES
Table_0010_all = score_fz(VaRandES, R, 'HEval', 10, ...
    'alpha_level', 0.01, 'Models_id', ModelEval, 'PrintTable', false, ...
    'DateStart', 20060201, 'DateEnd', 20231130, 'ExcludeDates', [], ...
    'EvalDates', [], 'Decimals', 3, 'MCSLevel', SigLevel);

% Fissler-Ziegel (FZ) loss and Model Confidence Set of 2.5% VaR and ES
Table_0025_all = score_fz(VaRandES, R, 'HEval', 10, ...
    'alpha_level', 0.025, 'Models_id', ModelEval, 'PrintTable', false, ...
    'DateStart', 20060201, 'DateEnd', 20231130, 'ExcludeDates', [], ...
    'EvalDates', [], 'Decimals', 3, 'MCSLevel', SigLevel);


%%% Filtering 0.01% most extreme quantile losses
% Fissler-Ziegel (FZ) loss and Model Confidence Set of 1% VaR and ES
Table_0010_q0010 = score_fz(VaRandES, R, 'HEval', 10, ...
    'alpha_level', 0.01, 'Models_id', ModelEval, 'PrintTable', false, ...
    'DateStart', 20060201, 'DateEnd', 20231130, 'ExcludeDates', [], ...
    'EvalDates', EvalDates_alpha_0010_q_0010.dates_eval, 'Decimals', 3, ...
    'MCSLevel', SigLevel);

% Fissler-Ziegel (FZ) loss and Model Confidence Set of 2.5% VaR and ES
Table_0025_q0010 = score_fz(VaRandES, R, 'HEval', 10, ...
    'alpha_level', 0.025, 'Models_id', ModelEval, 'PrintTable', false, ...
    'DateStart', 20060201, 'DateEnd', 20231130, 'ExcludeDates', [], ...
    'EvalDates', EvalDates_alpha_0025_q_0010.dates_eval, 'Decimals', 3, ...
    'MCSLevel', SigLevel);


%%% Filtering 0.005% most extreme quantile losses
% Fissler-Ziegel (FZ) loss and Model Confidence Set of 1% VaR and ES
Table_0010_q0005 = score_fz(VaRandES, R, 'HEval', 10, ...
    'alpha_level', 0.01, 'Models_id', ModelEval, 'PrintTable', false, ...
    'DateStart', 20060201, 'DateEnd', 20231130, 'ExcludeDates', [], ...
    'EvalDates', EvalDates_alpha_0010_q_0005.dates_eval, 'Decimals', 3, ...
    'MCSLevel', SigLevel);

% Fissler-Ziegel (FZ) loss and Model Confidence Set of 2.5% VaR and ES
Table_0025_q0005 = score_fz(VaRandES, R, 'HEval', 10, ...
    'alpha_level', 0.025, 'Models_id', ModelEval, 'PrintTable', false, ...
    'DateStart', 20060201, 'DateEnd', 20231130, 'ExcludeDates', [], ...
    'EvalDates', EvalDates_alpha_0025_q_0005.dates_eval, 'Decimals', 3, ...
    'MCSLevel', SigLevel);


% Construct combined output for LaTeX (ready for copy and paste)
Tab2 = Table_0025_all.LaTeX;
Tab2.Properties.VariableNames = strcat(Tab2.Properties.VariableNames,'_2');
Tab3 = Table_0010_q0005.LaTeX;
Tab3.Properties.VariableNames = strcat(Tab3.Properties.VariableNames,'_3');
Tab4 = Table_0025_q0005.LaTeX;
Tab4.Properties.VariableNames = strcat(Tab4.Properties.VariableNames,'_4');
Tab5 = Table_0010_q0010.LaTeX;
Tab5.Properties.VariableNames = strcat(Tab5.Properties.VariableNames,'_5');
Tab6 = Table_0025_q0010.LaTeX;
Tab6.Properties.VariableNames = strcat(Tab6.Properties.VariableNames,'_6');
% disp([Table_0010_all.LaTeX(:,1:end-1) Tab2(:,2:end-1) ...
%       Tab3(:,2:end-1) Tab4(:,2:end-1) Tab5(:,2:end-1) Tab6(:,2:end)])



%% UC and ES backtest


BacktestTable_0010 = VaRESBacktest(VaRandES, R, 'alpha_level', 0.01, ...
                                   'Models_id', []);
BacktestTable_0025 = VaRESBacktest(VaRandES, R, 'alpha_level', 0.025, ...
                                   'Models_id', []);

BacktestTable_0010.LaTeX
BacktestTable_0025.LaTeX



