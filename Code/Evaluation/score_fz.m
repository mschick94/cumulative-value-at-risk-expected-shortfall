function EvalOut = score_fz(VaRES, R, varargin)

% NOTE: HStepCumReturnSim must contain either all h=1,...,H or only h=H
% Storing arbitrary subsets of horizons is not supported

% Add dependencies to header



% Name-value inputs
p = inputParser;
addParameter(p, 'HEval',     []); % default is all horizons provided
addParameter(p, 'DateStart', []);
addParameter(p, 'DateEnd',   []);
parse(p, varargin{:});

HEval     = p.Results.HEval;
DateStart = p.Results.DateStart;
DateEnd   = p.Results.DateEnd;

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

    h = HEval(i);        % actual forecast horizon
    h_idx = min(h, n_h); % maps actual horizon to array index (if n_h=1)

    hStepVaRmat = VaRmat(:, :, h_idx);   % (T x J)
    hStepESmat  = ESmat(:, :, h_idx);    % (T x J)

    % Realized cumulative PF returns at horizon h
    ActualCumPFRet = CumActualPFRet(:, h);

    % FZ Loss computed as two parts, i.e. FirstSum + SecondSum
    VaRNegidx      = ActualCumPFRet <= hStepVaRmat;
    IndVaRMinusRet = VaRNegidx .* (hStepVaRmat - ActualCumPFRet);

    FirstSum = - 1/alpha * IndVaRMinusRet ./ hStepESmat;
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
    warning('score_fz: DateStart is within burn-in period — results may contain NaN');
end
if ~isempty(DateEnd) && (t_end_eval > T - H_max + 1)
    warning('score_fz: DateEnd too close to end of sample — results may contain NaN');
end
if isempty(t_start_eval) || isempty(t_end_eval)
    error('score_fz: DateStart or DateEnd not found in dates vector');
end


LossMat    = LossMat(t_start_eval:t_end_eval, :, :);
dates_eval = dates(t_start_eval:t_end_eval);



end