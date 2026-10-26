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
BT_truepars_n = BackTestSimTable(B, 'dist', 'norm', 'EstPars', false);
BT_estpars_n  = BackTestSimTable(B, 'dist', 'norm', 'EstPars', true);
BT_truepars_t = BackTestSimTable(B, 'dist', 't', 'EstPars', false);
BT_estpars_t  = BackTestSimTable(B, 'dist', 't', 'EstPars', true);

% % Construct combined output for LaTeX (ready for copy and paste)
% TabTrue = BT_truepars_n.LaTeX;
% TabEst  = BT_estpars_n.LaTeX;
% TabEst.Properties.VariableNames = strcat(TabEst.Properties.VariableNames,'_2');
% disp([TabTrue([1,3],1:5) TabEst([1,3],2:5) TabTrue([1,3],6:end-1) TabEst([1,3],6:end); ...
%       TabTrue([2,4],1:5) TabEst([2,4],2:5) TabTrue([2,4],6:end-1) TabEst([2,4],6:end)])
% 
% TabTrue = BT_truepars_t.LaTeX;
% TabEst  = BT_estpars_t.LaTeX;
% TabEst.Properties.VariableNames = strcat(TabEst.Properties.VariableNames,'_2');
% disp([TabTrue([1,3],1:5) TabEst([1,3],2:5) TabTrue([1,3],6:end-1) TabEst([1,3],6:end); ...
%       TabTrue([2,4],1:5) TabEst([2,4],2:5) TabTrue([2,4],6:end-1) TabEst([2,4],6:end)])



%% Load model forecasts for K
clearvars -except ReturnData

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

% Prepare result structure to be merged with other models
VaRandES.VaR = squeeze(VaRandES.VaR);
VaRandES.ES  = squeeze(VaRandES.ES);

% Load univariate HEAVY multivariate RiskMetrics and HEAVY-GAS
VaRandES = merge_mv_models(VaRandES);

% % Output for Anne
% VaRandES.VaR     = VaRandES.VaR(1001:end-9, :, :);
% VaRandES.ES      = VaRandES.ES(1001:end-9, :, :);
% VaRandES.EmpPITs = VaRandES.EmpPITs(1001:end-9, :);
% VaRandES.dates   = VaRandES.dates(1001:end-9);


% Out-of-sample evaluation

% VaR and ES level
% alpha = 0.01;
alpha = 0.025;

% MCS significance level
SigLevel = 0.10;

% Models
ModelEval = [12, 14, 16, 18, 20, 10, 32, 35, 33, 36, 34, 26, 30, 27, 28, 29]; 

%%% Tick Loss (Quantile Score)

% Dates excluded based on median loss differentials
EvalDates = score_fz(VaRandES, R, 'HEval', 10, 'alpha_level', alpha, ...
    'Models_id', ModelEval, 'ExcludeQuant', 0.005, ...
    'DateStart', 20060201, 'DateEnd', 20231130, 'MCSTest', false, ...
    'Loss', 'QS');

% Quantile score
QS = score_fz(VaRandES, R, 'HEval', 10, 'alpha_level', alpha, ...
    'Models_id', ModelEval, 'PrintTable', false, 'DateStart', 20060201, ...
    'DateEnd', 20231130, 'ExcludeDates', [], ...
    'Decimals', 3, 'EvalDates', EvalDates.dates_eval, ...
    'MCSLevel', SigLevel, 'Loss', 'QS');


%%% FZ Loss

% Dates excluded based on median loss differentials
EvalDates = score_fz(VaRandES, R, 'HEval', 10, 'alpha_level', alpha, ...
    'Models_id', ModelEval, 'ExcludeQuant', 0.005, ...
    'DateStart', 20060201, 'DateEnd', 20231130, 'MCSTest', false, ...
    'Loss', 'FZ');

% Quantile score
FZ = score_fz(VaRandES, R, 'HEval', 10, 'alpha_level', alpha, ...
    'Models_id', ModelEval, 'PrintTable', false, 'DateStart', 20060201, ...
    'DateEnd', 20231130, 'ExcludeDates', [], ...
    'Decimals', 3, 'EvalDates', EvalDates.dates_eval, ...
    'MCSLevel', SigLevel, 'Loss', 'FZ');

% Backtests
BT = VaRESBacktest(VaRandES, R, 'alpha_level', alpha, ...
    'Models_id', ModelEval);


% Table for LaTeX
FZTab = FZ.LaTeX;
FZTab.Properties.VariableNames = strcat(FZTab.Properties.VariableNames,'_2');
disp([QS.LaTeX(:,1:3), FZTab(:,2:3), BT.LaTeX(:,2:end)])

