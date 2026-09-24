function BackTestTable = VaRESBacktest(VaRES, R, varargin)


% Name-value inputs
p = inputParser;
addParameter(p, 'HEval',        []); % default is all horizons provided
addParameter(p, 'alpha_level',  []);
addParameter(p, 'Models_id',    []);
addParameter(p, 'DateStart',    []);
addParameter(p, 'DateEnd',      []);
addParameter(p, 'ExcludeDates', []);
addParameter(p, 'Decimals',     3);
addParameter(p, 'LaTeX',        true);
addParameter(p, 'PrintTable',   true);
parse(p, varargin{:});

HEval        = p.Results.HEval;
alpha_level  = p.Results.alpha_level;
Models_id    = p.Results.Models_id;
DateStart    = p.Results.DateStart;
DateEnd      = p.Results.DateEnd;
ExcludeDates = p.Results.ExcludeDates;
Decimals     = p.Results.Decimals;
LaTeX        = p.Results.LaTeX;
PrintTable   = p.Results.PrintTable;

% Read out all model names considered and general set up
ModelNames = VaRES.Models;
J          = max(size(ModelNames));
alpha      = VaRES.alpha;
PFweights  = VaRES.PFweights;
WindLength = VaRES.WindLength;
dates      = VaRES.dates;
T          = size(dates,1);

% Compute actual PF returns
ActualPFRet = R * PFweights;

% ONLY h = H used in this project so far; can be made flexible again later!
if isempty(HEval)
    HEval = VaRES.H;
end

% Cumulative PF returns up to max horizon needed
H_max          = max(HEval);
CumActualPFRet = NaN(T, H_max);
CumSum         = cumsum(ActualPFRet);
for h = 1:H_max
    CumActualPFRet(1:T-h+1, h) = CumSum(h:T) - [0; CumSum(1:T-h)];
end
CumActualRet = CumActualPFRet(:, H_max);

% Read out VaR and ES level
if isempty(alpha_level)
    alpha_test = alpha(1);
    alpha_ind  = 1;
else
    alpha_ind  = find(alpha_level == alpha);
    alpha_test = alpha(alpha_ind);
end

if isempty(alpha_ind)
    error('VaRESBacktest: alpha level not found in estimation results');
end

% Read out model forecasts of VaR and ES transform at alpha_test level
if isempty(Models_id)
    Models_id = 1:J;
else
    J = length(Models_id);
end
VaRmat  = VaRES.VaR(:, Models_id, alpha_ind);
EmpPITs = VaRES.EmpPITs(:, Models_id);

% (Demeaned) Hit sequence
hits = CumActualRet <= VaRmat;  % logical
hitsdiff = hits - alpha_test;   % numeric
HitRate  = mean(hits);


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
    warning(['VaRESBacktest: DateStart is within burn-in period — ' ...
             'results may contain NaN']);
end
if ~isempty(DateEnd) && (t_end_eval > T - H_max + 1)
    warning(['VaRESBacktest: DateEnd too close to end of sample — ' ...
             'results may contain NaN']);
end
if isempty(t_start_eval) || isempty(t_end_eval)
    error('VaRESBacktest: DateStart or DateEnd not found in dates vector');
end

% Only keep losses over the evaluation sample
hitsdiff    = hitsdiff(t_start_eval:t_end_eval, :, :);
EmpPITsEval = EmpPITs(t_start_eval:t_end_eval, :, :);
dates_eval  = dates(t_start_eval:t_end_eval);
T_eval      = size(dates_eval,1);


% Exclude specified date ranges
if ~isempty(ExcludeDates)
    excl_ind = false(T_eval, 1);
    for i = 1:size(ExcludeDates, 1)
        % Last date equal or smaller than start
        t_excl_start = find(dates_eval <= ExcludeDates(i,1), 1, 'last');
        % First date equal or after end
        t_excl_end   = find(dates_eval >= ExcludeDates(i,2), 1, 'first');
        if isempty(t_excl_start) || isempty(t_excl_end)
            warning(['VaRESBacktest: ExcludeDates row %d outside ' ...
                     'evaluation sample — skipping'], i);
            continue
        end
        excl_ind(t_excl_start:t_excl_end) = true;
    end
    hitsdiff    = hitsdiff(~excl_ind, :, :);
    EmpPITsEval = EmpPITsEval(~excl_ind, :);
    dates_eval = dates_eval(~excl_ind);
    T_eval     = size(dates_eval, 1);
    % fprintf('score_fz: excluded %d observations\n', sum(excl_ind));
end

% Safety check: LossMat should not contain NaNs
if any(isnan(hitsdiff(:)))
    warning(['VaRESBacktest: NaN values found in LossMat after ' ...
             'evaluation sample restriction; check t_start_eval and ' ...
             't_end_eval']);
end


% Unconditional coverage test from Patton et al. (2019) with HAC
UC_xbar = (HitRate - alpha_test)';
UC_se      = NaN(J,1);
for j = 1:J
    [~, UC_se(j)] = hac(ones(T_eval, 1), hitsdiff(:,j), 'type', 'HAC', ...
                 'Intercept', false, 'bandwidth', H_max, 'display', 'off');
end
UC_t_HAC = UC_xbar ./ UC_se;


% PIT-based ES test (Du & Escanciano via empirical PITs) with HAC
E_h     = alpha_test / 2;
H_mat   = (1./alpha_test) .* (alpha_test - EmpPITsEval) ...
           .* (EmpPITsEval <= alpha_test);
ES_xbar = mean(H_mat)';
ES_se   = NaN(J,1);
for j = 1:J
    [~, ES_se(j)] = hac(ones(T_eval, 1), H_mat(:,j), 'type', 'HAC', ...
                 'Intercept', false, 'bandwidth', H_max, 'display', 'off');
end
ES_t_HAC = (ES_xbar - E_h) ./ ES_se;


% Display results table 
if PrintTable
    fprintf('\nBacktest results: alpha = %.3f, T_eval = %d\n', alpha_test, T_eval);
    fprintf('\n%-40s %10s %10s %10s %10s\n', ...
            'Model', 'Hit Rate', 'UC t-stat', 'ES xbar', 'ES t-stat');
    fprintf('%s\n', repmat('-', 1, 85));
    for j = Models_id
        fprintf('%-40s %10.4f %10.3f %10.4f %10.3f\n', ...
                ModelNames{j}, ...
                UC_xbar(j) + alpha_test, ...
                UC_t_HAC(j), ...
                ES_xbar(j), ...
                ES_t_HAC(j));
    end
    fprintf('%s\n', repmat('-', 1, 85));
end


% LaTeX table
if LaTeX
    fmt_str = sprintf('%%.%df', Decimals);
    get_stars = @(t) repmat('*', 1, (abs(t) > 1.645) + (abs(t) > 1.960) ...
                             + (abs(t) > 2.576));
    n_cols    = 8;   % name & hitrate & UC & ES & \\
    table_str = strings(length(Models_id), n_cols);

    for idx = 1:length(Models_id)
        j = Models_id(idx);
        hr_str = sprintf(fmt_str, UC_xbar(j) + alpha_test);
        uc_str = [sprintf(fmt_str, UC_t_HAC(j)) get_stars(UC_t_HAC(j))];
        es_str = [sprintf(fmt_str, ES_t_HAC(j)) get_stars(ES_t_HAC(j))];
        table_str(idx,:) = {ModelNames{j}, '&', hr_str, '&', uc_str, ...
                            '&', es_str, '\\'};
    end

    % T row at bottom
    T_row = strings(1, n_cols);
    T_row(1)   = 'T';
    T_row(2)   = '&';
    T_row(3)   = sprintf('%d', T_eval);
    T_row(4)   = '&';
    T_row(5)   = sprintf('%d', T_eval);
    T_row(6)   = '&';
    T_row(7)   = sprintf('%d', T_eval);
    T_row(end) = '\\';
    table_str  = [table_str; T_row];

    Tab = array2table(table_str);
    for i = 1:n_cols
        Tab.(i) = categorical(Tab.(i));
    end
    BackTestTable.LaTeX = Tab;
end



end