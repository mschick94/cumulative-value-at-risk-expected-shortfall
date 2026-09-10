function EstOut = estimate_heavy(R, RV, varargin)
%ESTIMATE_HEAVY Rolling-window estimation of a univariate HEAVY model
% for K assets using realized variance as an additional input.
%
%   EstOut = ESTIMATE_HEAVY(R, RV) estimates a HEAVY model with normal
%   innovations using default settings.
%
%   INPUTS (required):
%       R  : (TxK) matrix of returns
%       RV : (Tx(K*(K+1)/2)) matrix of realized variances in vech format
%
%   INPUTS (optional name-value):
%       'dist'       : String, marginal distribution
%                      'norm' - standard normal (default)
%                      't', 'skewt', 'laplace'
%       'ReestFreq'  : Scalar, re-estimation freq. in days (default: 21)
%       'WindLength' : Scalar, estimation window length (default: 1000)
%       'NumWorkers' : Scalar, parallel workers (default: 1)
%       'assets'     : Cell array of asset names (default: [])
%       'dates'      : Vector of dates (default: [])
%       'SaveDisk'   : Logical, store results to disk (default: true)
%
%   OUTPUT:
%       EstOut : Struct containing all estimation output
%                EstOut.model         - model identifier string
%                EstOut.dist          - marginal distribution
%                EstOut.assets        - asset names
%                EstOut.dates         - dates vector
%                EstOut.WindLength    - estimation window length
%                EstOut.ReestFreq     - re-estimation frequency
%                EstOut.HEAVY_r_pars  - (T x 3 x K) return equation pars
%                EstOut.HEAVY_RV_pars - (T x 4 x K) RM equation pars
%                EstOut.mu            - (T x K) mean estimates
%                EstOut.H_last        - (T x K) last filtered variance
%                EstOut.Tau_last      - (T x K) last filtered RV measure
%                EstOut.NegLL_r       - (T x K) return equation neg LL
%                EstOut.NegLL_RV      - (T x K) RM equation neg LL
%                EstOut.std_res       - (W x K x T) rolling std. resid.
%                EstOut.nu            - (T x K) degrees of freedom (t, skewt)
%                EstOut.lambda        - (T x K) skewness parameter (skewt)
%

% Name-value inputs
p = inputParser;
addParameter(p, 'dist',        'norm');
addParameter(p, 'ReestFreq',   21);
addParameter(p, 'WindLength',  1000);
addParameter(p, 'NumWorkers',  1);
addParameter(p, 'assets',      []);
addParameter(p, 'dates',       []);
addParameter(p, 'SaveDisk',    true);
parse(p, varargin{:});

dist        = p.Results.dist;
reest_freq  = p.Results.ReestFreq;
W           = p.Results.WindLength;
num_workers = p.Results.NumWorkers;
assets      = p.Results.assets;
dates       = p.Results.dates;
SaveDisk    = p.Results.SaveDisk;

% Dimensions
[T, K]  = size(R);
t_start = W + 1;

% Set up parallel pool if requested
if num_workers > 1
    pool = gcp('nocreate');
    if isempty(pool)
        parpool('local', min(num_workers, 6));
    elseif pool.NumWorkers < num_workers
        delete(pool);
        parpool('local', min(num_workers, 6));
    end
end

% fmincon options
options = optimoptions(@fmincon, 'Algorithm', 'sqp', 'Display', 'off');

% Bounds on constraints
epsi = 1e-10;


% Distribution-specific setup for return distribution (Return equation)
switch dist
    case 'norm'
        A          = [     0,      1,      1];
        b          =  1-epsi;
        lb         = [     0;      0;      0];
        ub         = [   Inf; 1-epsi; 1-epsi];
        start      = [   0.2,    0.1,    0.8];
        ll_fun     = @(pars, r, RV) ...
            VarianceModels.univ_heavy_r(pars, r, RV);
        get_nu     = false;
        get_lambda = false;
        idx_nu     = NaN;
        idx_lambda = NaN;        

    case 't'
        A          = [     0,      1,      1,    0];
        b          =  1-epsi;
        lb         = [     0;      0;      0;  2.2];
        ub         = [   Inf; 1-epsi; 1-epsi; 1000];
        start      = [   0.2,    0.1,    0.8,   10];
        ll_fun     = @(pars, r, RV) ...
            VarianceModels.univ_heavy_r_t(pars, r, RV);
        get_nu     = true;
        get_lambda = false;
        idx_nu     = 4;    % [omega, alpha, beta, nu]
        idx_lambda = NaN;        

    case 'skewt'
        A          = [     0,      1,      1,       0,    0];
        b          =  1-epsi;
        lb         = [     0;      0;      0; -1+0.05;  2.2];
        ub         = [   Inf; 1-epsi; 1-epsi;  1-0.05; 1000];
        start      = [   0.2,    0.1,    0.8,       0,   10];
        ll_fun     = @(pars, r, RV) ...
            VarianceModels.univ_heavy_r_skew_t(pars, r, RV);
        get_nu     = true;
        get_lambda = true;
        idx_nu     = 5;    % [omega, alpha, beta, lambda, nu]
        idx_lambda = 4;

    case 'laplace'
        A          = [     0,      1,      1];
        b          =  1-epsi;
        lb         = [     0;      0;      0];
        ub         = [   Inf; 1-epsi; 1-epsi];
        start      = [   0.2,    0.1,    0.8];
        ll_fun     = @(pars, r, RV) ...
            VarianceModels.univ_heavy_r_laplace(pars, r, RV);
        get_nu     = false;
        get_lambda = false;
        idx_nu     = NaN;
        idx_lambda = NaN;        

    otherwise
        error(['estimate_garch: unknown dist ''%s''. ' ...
               'Expected ''norm'', ''t'', ''skewt'', or ''laplace''.'], ...
               dist);
end

% RV set up shared by return equations (Realized Measure equation)
Arv          = [     0,      1,      1,    0];
brv          =  1-epsi;
lbrv         = [     0;      0;      0; epsi];
ubrv         = [   Inf; 1-epsi; 1-epsi;  Inf];
startrv      = [   0.2,    0.1,    0.8,    1];
ll_fun_RM = @(pars, rv) VarianceModels.univ_heavy_RM(pars, rv);

% Pre-allocate
mu            = NaN(T, K);
HEAVY_r_pars  = NaN(T, 3, K);
HEAVY_RV_pars = NaN(T, 4, K);
H_last        = NaN(T, K);
Tau_last      = NaN(T, K);
NegLL_r       = NaN(T, K);
NegLL_RV      = NaN(T, K);
std_res       = NaN(W, K, T);
nu            = NaN(T, K);
lambda        = NaN(T, K);

% Pre-compute re-estimation dates
reest_dates = t_start:reest_freq:T;
n_reest     = length(reest_dates);

% Temporary storage for parfor (index by i not t) of HEAVY-r and HEAVY-RM
mu_reest           = NaN(n_reest, K);
HEAVY_r_pars_reest = NaN(n_reest, 3, K);
H_last_reest       = NaN(n_reest, K);
Tau_last_reest     = NaN(n_reest, K);
NegLL_reest        = NaN(n_reest, K);
std_res_reest      = NaN(W, K, n_reest);
nu_reest           = NaN(n_reest, K);
lambda_reest       = NaN(n_reest, K);

HEAVY_RV_pars_reest = NaN(n_reest, 4, K);
NegLL_RV_reest      = NaN(n_reest, K);

% Read out realized variances
diag_idx = 1 + cumsum([0, K:-1:2]); % indices of diagonal in vech
RVs      = RV(:, diag_idx);         % (T x K) realized variances

% Pre-slice R and RV windows before parfor to avoid overhead
R_windows  = cell(n_reest, K);
RV_windows = cell(n_reest, K);
for i = 1:n_reest
    t = reest_dates(i);
    for k = 1:K
        R_windows{i,k}  = R(t-W:t-1, k);
        RV_windows{i,k} = RVs(t-W:t-1, k);
    end
end

% Loop 1: parfor/for over re-estimation dates
if num_workers > 1
    parfor i = 1:n_reest
        ll_fun_local    = ll_fun;
        ll_fun_RM_local = ll_fun_RM;
        for k = 1:K
            r  = R_windows{i,k};
            rv = RV_windows{i,k};

            % Estimate RM equation
            Heavy_RV_pars = fmincon(@(pars) ll_fun_RM_local(pars, rv), ...
                startrv, Arv, brv, [], [], lbrv, ubrv, [], options); 

            [NegLLRVk, Tau_t]          = ll_fun_RM_local(Heavy_RV_pars,rv);
            Tau_last_reest(i,k)        = Tau_t(end); 
            HEAVY_RV_pars_reest(i,:,k) = Heavy_RV_pars;
            NegLL_RV_reest(i,k)        = NegLLRVk;

            % Estimate return equation
            Heavy_r_pars = fmincon(@(pars) ll_fun_local(pars, r, rv), ...
                start, A, b, [], [], lb, ub, [], options);             
                   
            [NegLLk, H_t, mu_est]     = ll_fun_local(Heavy_r_pars, r, rv); 
            HEAVY_r_pars_reest(i,:,k) = Heavy_r_pars(1:3);
            mu_reest(i,k)             = mu_est;
            H_last_reest(i,k)         = H_t(end);
            NegLL_reest(i,k)          = NegLLk;
            std_res_reest(:,k,i)      = (r - mu_est) ./ sqrt(H_t);
            if get_nu
                nu_reest(i,k) = Heavy_r_pars(idx_nu);
            end
            if get_lambda
                lambda_reest(i,k) = Heavy_r_pars(idx_lambda);
            end
        end
    end
else
    for i = 1:n_reest
        for k = 1:K
            r  = R_windows{i,k};
            rv = RV_windows{i,k};

            % Estimate RM equation
            Heavy_RV_pars = fmincon(@(pars) ll_fun_RM(pars, rv), ...
                startrv, Arv, brv, [], [], lbrv, ubrv, [], options); 

            [NegLLRVk, Tau_t]          = ll_fun_RM(Heavy_RV_pars, rv);
            HEAVY_RV_pars_reest(i,:,k) = Heavy_RV_pars;
            Tau_last_reest(i,k)        = Tau_t(end);            
            NegLL_RV_reest(i,k)        = NegLLRVk;

            % Estimate return equation
            Heavy_r_pars = fmincon(@(pars) ll_fun(pars, r, rv), ...
                start, A, b, [], [], lb, ub, [], options);             
                   
            [NegLLk, H_t, mu_est]     = ll_fun(Heavy_r_pars, r, rv); 
            HEAVY_r_pars_reest(i,:,k) = Heavy_r_pars(1:3);
            mu_reest(i,k)             = mu_est;
            H_last_reest(i,k)         = H_t(end);
            NegLL_reest(i,k)          = NegLLk;
            std_res_reest(:,k,i)      = (r - mu_est) ./ sqrt(H_t);
            if get_nu
                nu_reest(i,k) = Heavy_r_pars(idx_nu);
            end
            if get_lambda
                lambda_reest(i,k) = Heavy_r_pars(idx_lambda);
            end
        end
    end
end

% Map back to full arrays
for i = 1:n_reest
    t                    = reest_dates(i);
    HEAVY_r_pars(t,:,:)  = HEAVY_r_pars_reest(i,:,:);
    HEAVY_RV_pars(t,:,:) = HEAVY_RV_pars_reest(i,:,:);
    mu(t,:)              = mu_reest(i,:);
    H_last(t,:)          = H_last_reest(i,:);
    Tau_last(t,:)        = Tau_last_reest(i,:);
    NegLL_r(t,:)         = NegLL_reest(i,:);
    NegLL_RV(t,:)        = NegLL_RV_reest(i,:);
    std_res(:,:,t)       = std_res_reest(:,:,i);
    nu(t,:)              = nu_reest(i,:);
    lambda(t,:)          = lambda_reest(i,:);
end

% Loop 2: sequential carry-forward between re-estimation dates
% If reest_freq = 1 this loop body never executes
for t = t_start:T
    if mod(t - t_start, reest_freq) ~= 0
        for k = 1:K
            HEAVY_r_pars(t,:,k)  = HEAVY_r_pars(t-1,:,k);
            HEAVY_RV_pars(t,:,k) = HEAVY_RV_pars(t-1,:,k);
            mu(t,k)          = mu(t-1,k);
            omega            = HEAVY_r_pars(t,1,k);
            alpha            = HEAVY_r_pars(t,2,k);
            beta             = HEAVY_r_pars(t,3,k);
            H_last(t,k)      = omega + alpha*RVs(t-1,k) ...
                               + beta*H_last(t-1,k);
            omega_rv         = HEAVY_RV_pars(t,1,k);
            alpha_rv         = HEAVY_RV_pars(t,2,k);
            beta_rv          = HEAVY_RV_pars(t,3,k);   
            Tau_last(t,k)    = omega_rv + alpha_rv*RVs(t-1,k) ...
                               + beta_rv*Tau_last(t-1,k);           
            new_res          = (R(t-1,k) - mu(t,k)) / sqrt(H_last(t,k));
            std_res(:,k,t)   = [std_res(2:end,k,t-1); new_res];
            nu(t,k)          = nu(t-1,k);
            lambda(t,k)      = lambda(t-1,k);
        end
    end
end


% Pack EstOut
EstOut.model         = sprintf('HEAVY_%s', dist);
EstOut.dist          = dist;
EstOut.assets        = assets;
EstOut.dates         = dates;
EstOut.WindLength    = W;
EstOut.ReestFreq     = reest_freq;
EstOut.HEAVY_r_pars  = HEAVY_r_pars;     % (T x 3 x K) return equation pars
EstOut.HEAVY_RV_pars = HEAVY_RV_pars;    % (T x 4 x K) RM equation pars
EstOut.mu            = mu;               % (T x K)
EstOut.H_last        = H_last;           % (T x K)
EstOut.Tau_last      = Tau_last;         % (T x K)
EstOut.NegLL_r       = NegLL_r;          % (T x K) return LL
EstOut.NegLL_RV      = NegLL_RV;         % (T x K) RM LL
EstOut.std_res       = std_res;          % (W x K x T)

% Distribution-specific fields
switch dist
    case {'norm', 'laplace'}
        % No extra fields
    case 't'
        EstOut.nu = nu;
    case 'skewt'
        EstOut.nu     = nu;
        EstOut.lambda = lambda;
end


% Save EstOut to disk
if isempty(assets)
    filename = sprintf('Output/Estimation/Marginals/EstOut_%s.mat', ...
                       EstOut.model);
else
    filename = sprintf('Output/Estimation/Marginals/EstOut_%s_%s.mat', ...
                       EstOut.model, strjoin(assets, '_'));
end

if SaveDisk
    save(filename, 'EstOut');
end


end

