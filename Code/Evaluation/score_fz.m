function FZEvalTable = score_fz(VaRES, R, varargin)
%SCORE_FZ Compute Fissler-Ziegel (FZ) loss scores for VaR and ES forecasts
% and perform Model Confidence Set (MCS) evaluation across models.
%
%   FZEvalTable = SCORE_FZ(VaRES, R) evaluates VaR and ES forecasts using
%   the jointly consistent FZ scoring function from Patton, Ziegel & Chen
%   (2019) and applies the MCS procedure of Hansen, Lunde & Nason (2011) to 
%   identify the set of superior models.
%
%   The FZ loss at time t is:
%       S_t = -1/(alpha*ES_t) * 1(r_t <= VaR_t) * (VaR_t - r_t)
%             + VaR_t/ES_t + log(-ES_t) - 1
%   where VaR and ES are reported as negative numbers (left tail).
%   Lower average loss indicates better forecast performance.
%
%   INPUTS (required):
%       VaRES : Struct, output from compute_var_es. Must contain fields:
%               .VaR        - (T x J x H) VaR forecasts, negative values
%               .ES         - (T x J x H) ES forecasts, negative values
%               .Models     - (J x 1) cell array of model names
%               .alpha      - significance level e.g. 0.025
%               .PFweights  - (1 x K) portfolio weights
%               .WindLength - estimation window length
%               .dates      - (T x 1) date vector
%               .H          - actual simulation horizon
%       R     : (T x K) matrix of observed asset returns
%
%   INPUTS (optional name-value):
%       'HEval'       : Vector of forecast horizons to evaluate
%                       (default: all available horizons)
%       'DateStart'   : Scalar, start date of evaluation sample
%                       (default: WindLength + 1)
%       'DateEnd'     : Scalar, end date of evaluation sample
%                       (default: T - H_max + 1)
%       'BlockLength' : Scalar, fixed block length for MCS bootstrap
%                       (default: data-driven via ACF)
%       'NBootstrap'  : Scalar, number of bootstrap replications
%                       (default: 25000)
%       'Decimals'    : Scalar, decimal places in LaTeX table (default: 2)
%       'MCSLevel'    : Scalar, confidence level for MCS (default: 0.9)
%       'PrintTable'  : Logical, display table in command window
%                       (default: true)
%
%   OUTPUT:
%       FZEvalTable : Struct containing evaluation results:
%                     .LaTeX      - MATLAB table with LaTeX-formatted loss
%                                   strings, bold best model, grey MCS
%                                   survivors (ready for copy-pasting)
%                     .Losses     - MATLAB table with numeric mean losses,
%                                   models as rows, horizons as columns
%                     .alpha      - significance level used
%                     .PFweights  - portfolio weights used
%                     .WindLength - estimation window length
%                     .T_eval     - number of evaluation observations
%                     .dates_eval - dates of evaluation sample
%                     .assets     - asset names
%                     .H          - simulation horizon
%
%   NOTES:
%       - Block length for MCS bootstrap is selected as the maximum
%         significant ACF lag across all models, capped at
%         max(floor(T^(1/3)), 2*H_max) to avoid spurious large blocks
%       - Realized cumulative PF returns are constructed as
%         r_t^PF(h) = sum(R(t:t+h-1,:) * w') aligned at forecast origin t
%       - Requires MCS implementation from Kevin Sheppard's MFE Toolbox:
%         https://www.mathworks.com/matlabcentral/fileexchange/170381
%       - Requires \usepackage{colortbl} in LaTeX preamble for cellcolor
%         (e.g. \definecolor{gray}{rgb}{0.90, 0.90, 0.90})
%
%   REFERENCES:
%       Patton, A., Ziegel, J., Chen, R. (2019). Dynamic semiparametric
%       models for expected shortfall (and Value-at-Risk). Journal of
%       Econometrics, 211(2), 388-413.
%
%       Hansen, P., Lunde, A., Nason, J. (2011). The model confidence set.
%       Econometrica, 79(2), 453-497.


% Name-value inputs
p = inputParser;
addParameter(p, 'HEval',       []); % default is all horizons provided
addParameter(p, 'DateStart',   []);
addParameter(p, 'DateEnd',     []);
addParameter(p, 'BlockLength', []);
addParameter(p, 'NBootstrap',  25000);
addParameter(p, 'Decimals',    2);
addParameter(p, 'MCSLevel',    0.9);
addParameter(p, 'PrintTable',  true);
parse(p, varargin{:});

HEval       = p.Results.HEval;
DateStart   = p.Results.DateStart;
DateEnd     = p.Results.DateEnd;
BlockLength = p.Results.BlockLength;
NBootstrap  = p.Results.NBootstrap;
Decimals    = p.Results.Decimals;
MCSLevel    = p.Results.MCSLevel;
PrintTable  = p.Results.PrintTable;

% Read out all model names considered and general set up
ModelNames = VaRES.Models;
J          = max(size(ModelNames));
alpha      = VaRES.alpha;
PFweights  = VaRES.PFweights;
WindLength = VaRES.WindLength;
dates      = VaRES.dates;
T          = size(dates,1);

% Compute actual PF returns
ActualPFRet = R * PFweights';

% Forecast horizons to evaluate (default is 1, ..., H)
n_h = size(VaRES.VaR, 3);   % 1 or H depending on what was stored

if isempty(HEval)
    if n_h == 1
        HEval = VaRES.H;    % only h=H stored, evaluate at h=H
    else
        HEval = 1:n_h;      % full term structure stored, evaluate all
    end                     % otherwise, horizons provided in vector HEval 
end

if any(HEval > VaRES.H)
    warning(['score_fz: HEval contains horizons exceeding H=%d ' ...
             '— clipping to H'], VaRES.H);
    HEval = HEval(HEval <= VaRES.H);
end
Hlength = max(size(HEval));

% Cumulative PF returns up to max horizon needed
H_max          = max(HEval);
CumActualPFRet = NaN(T, H_max);
CumSum         = cumsum(ActualPFRet);
for h = 1:H_max
    CumActualPFRet(1:T-h+1, h) = CumSum(h:T) - [0; CumSum(1:T-h)];
end

% Read out model forecasts of VaR and ES
VaRmat = VaRES.VaR;
ESmat  = VaRES.ES;

% Compute FZ losses from Patton et al. (2019) for specified horizons h
LossMat = NaN(T, J, Hlength);
for i = 1:Hlength

    h = HEval(i);         % actual forecast horizon
    h_idx = min(h, n_h);  % maps actual horizon to array index (if n_h=1)

    hStepVaRmat = VaRmat(:, :, h_idx);  % (T x J)
    hStepESmat  = ESmat(:, :, h_idx);   % (T x J)

    % Realized cumulative PF returns at horizon h
    ActualCumPFRet = CumActualPFRet(:, h);

    % FZ Loss computed as two parts, i.e., FirstSum + SecondSum
    VaRNegidx      = ActualCumPFRet <= hStepVaRmat;
    IndVaRMinusRet = VaRNegidx .* (hStepVaRmat - ActualCumPFRet);

    FirstSum  = - 1/alpha * IndVaRMinusRet ./ hStepESmat;
    SecondSum = hStepVaRmat ./ hStepESmat + log(-hStepESmat) - 1;

    % Overall loss
    LossMat(:, :, i) = FirstSum + SecondSum;
end

% Determine evaluation sample
if ~isempty(DateStart)
    t_start_eval = find(dates == DateStart, 1);
else
    t_start_eval = WindLength + 1;
end

if ~isempty(DateEnd)
    t_end_eval = find(dates == DateEnd, 1);
else
    t_end_eval = T - H_max + 1;
end

if ~isempty(DateStart) && (t_start_eval < WindLength + 1)
    warning(['score_fz: DateStart is within burn-in period — results ' ...
             'may contain NaN']);
end
if ~isempty(DateEnd) && (t_end_eval > T - H_max + 1)
    warning(['score_fz: DateEnd too close to end of sample — results ' ...
             'may contain NaN']);
end
if isempty(t_start_eval) || isempty(t_end_eval)
    error('score_fz: DateStart or DateEnd not found in dates vector');
end

% Only keep losses over the evaluation sample
LossMat    = LossMat(t_start_eval:t_end_eval, :, :);
dates_eval = dates(t_start_eval:t_end_eval);
T_eval     = size(dates_eval,1);

% Safety check: LossMat should not contain NaNs
if any(isnan(LossMat(:)))
    warning(['score_fz: NaN values found in LossMat after evaluation ' ...
             'sample restriction; check t_start_eval and t_end_eval']);
end


% Forecast evaluation for specified horizons h
includedMCS = cell(Hlength, 1);
for i = 1:Hlength
    hLossMat = LossMat(:, :, i);

    % Block length for bootstrap
    lags    = 1;
    max_lag = max(floor(T_eval^(1/3)), 2*H_max);
    if isempty(BlockLength)
        conf = 1.96 / sqrt(T_eval);
        for j = 1:J
            acf_j    = autocorr(hLossMat(:,j), 'NumLags', max_lag);
            sig_lags = find(abs(acf_j(2:end)) > conf);
            if ~isempty(sig_lags)
                lags = max(lags, max(sig_lags));
            end
        end
    else
        lags = BlockLength;
    end

    % MCS
    includedMCS{i} = mcs(hLossMat, MCSLevel, NBootstrap, lags, 'BLOCK');

end

% Compute average losses per horizon h and construct LaTeX formatted table
meanLosses = reshape(mean(LossMat, 1), J, Hlength);  
FZEvalTable.LaTeX = print_latex_losses(meanLosses, includedMCS, ...
    ModelNames, HEval, Decimals, PrintTable);

% Numeric summary table
h_col_names        = arrayfun(@(h) sprintf('h%d', h), HEval, ...
                                           'UniformOutput', false);
NumericTab         = array2table(meanLosses, 'RowNames', ModelNames, ...
                                 'VariableNames', h_col_names);
FZEvalTable.Losses = NumericTab;

% Pack additional information
FZEvalTable.alpha      = alpha;
FZEvalTable.PFweights  = PFweights;
FZEvalTable.WindLength = WindLength;
FZEvalTable.T_eval     = T_eval;
FZEvalTable.dates_eval = dates_eval;
FZEvalTable.assets     = VaRES.assets;
FZEvalTable.H          = HEval;

end



function Tab = print_latex_losses(mean_losses, includedModels, ...
    ModelNames, HEval, decimals, PrintTable)
%PRINT_LATEX_LOSSES Print LaTeX table body for VaR/ES forecast evaluation
% with MCS highlighting and bold letters for best model.
%
%   INPUTS:
%       mean_losses    : (J x Hlength) mean FZ losses per model and horizon
%       includedModels : (Hlength x 1) cell array of MCS survivor indices
%       ModelNames     : (J x 1) cell array of model names
%       HEval          : (1 x Hlength) vector of evaluated horizons
%       decimals       : scalar, number of decimal places
%       PrintTable     : if true, print LaTeX table to output window
%
%   OUTPUT:
%       Tab : MATLAB table with LaTeX-formatted loss strings, MCS shading
%             and bold letters for best model per horizon. Each row is a
%             valid LaTeX table row ready for copy-pasting. Columns contain
%             model names, ampersand separators, formatted loss strings,
%             and row terminators (\\). Tab is also displayed via disp().
%
%   NOTES:
%       - Lower loss is better — bold indicates best model per horizon
%       - Grey shading via \cellcolor{gray} indicates MCS inclusion
%       - Column names are display-only and not part of LaTeX output
%       - Requires \usepackage{colortbl} in LaTeX preamble for cellcolor:
%         \usepackage{colortbl}
%         \definecolor{gray}{rgb}{0.90, 0.90, 0.90}

J          = size(mean_losses, 1);
Hlength    = size(mean_losses, 2);
format_str = sprintf('%%.%df', decimals);

% Find best model per horizon
[~, bold_ind] = min(mean_losses, [], 1);   % (1 x Hlength)

% Build string table
% Structure: model_name | & | cell_str | & | cell_str | ... | cell_str | \\
n_cols    = 2 * Hlength + 2;
table_str = strings(J, n_cols);

% Model names, first & and final \\
table_str(:, 1)   = string(ModelNames);
table_str(:, 2)   = '&';
table_str(:, end) = '\\';

% Fill cell strings and & separators — col=3 since cols 1,2 are name and &
col = 3;
for i = 1:Hlength
    for j = 1:J
        val     = mean_losses(j, i);
        num_str = sprintf(format_str, val);
        is_bold = (j == bold_ind(i));
        is_grey = ismember(j, includedModels{i});

        if is_bold && is_grey
            cell_str = ['\cellcolor{gray}\textbf{' num_str '}'];
        elseif is_bold
            cell_str = ['\textbf{' num_str '}'];
        elseif is_grey
            cell_str = ['\cellcolor{gray}' num_str];
        else
            cell_str = num_str;
        end
        table_str(j, col) = cell_str;
    end
    if i < Hlength
        table_str(:, col+1) = '&';
        col = col + 2;
    end
end

% Convert to table with categorical columns to remove quotes in display
Tab = array2table(table_str);
for i = 1:n_cols
    Tab.(i) = categorical(Tab.(i));
end

% Add column names
col_names      = cell(1, n_cols);
col_names{1}   = 'Model';
col_names{end} = 'newline';
col_names{2}   = 'amp1';
for i = 1:Hlength
    col_names{2*i+1} = sprintf('h%d', HEval(i));
    if i < Hlength
        col_names{2*i+2} = sprintf('amp%d', i+1);
    end
end

Tab.Properties.VariableNames = col_names;

% Print LaTeX table
if PrintTable
    disp(Tab)
end

end