function EstOut = estimate_gjr_garch(R, varargin)
%ESTIMATE_GJR_GARCH Rolling-window estimation of a univariate 
% GJR-GARCH(1,1) model for K assets.
%
%   EstOut = ESTIMATE_GJR_GARCH(R) estimates a GJR-GARCH(1,1) model with 
%   normal innovations using default settings.
%
%   INPUTS (required):
%       R      : (TxK) matrix of returns
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
%
%   OUTPUT:
%       EstOut : Struct containing all estimation output
%                EstOut.model         - model identifier string
%                EstOut.dist          - marginal distribution
%                EstOut.assets        - asset names
%                EstOut.dates         - dates vector
%                EstOut.WindLength    - estimation window length
%                EstOut.ReestFreq     - re-estimation frequency
%                EstOut.GJRGARCHpars  - (T x 4 x K) parameter estimates
%                EstOut.mu            - (T x K) mean estimates
%                EstOut.H_last        - (T x K) last filtered variance
%                EstOut.NegLL         - (T x K) negative log-likelihood
%                EstOut.std_res       - (W x K x T) rolling standard. res.
%                EstOut.nu            - (T x K) DoF (t, skewt)
%                EstOut.lambda        - (T x K) skewness parameter (skewt)

% Name-value inputs
p = inputParser;
addParameter(p, 'dist',       'norm');
addParameter(p, 'ReestFreq',  21);
addParameter(p, 'WindLength', 1000);
addParameter(p, 'NumWorkers', 1);
addParameter(p, 'assets',     []);
addParameter(p, 'dates',      []);
parse(p, varargin{:});

dist        = p.Results.dist;
reest_freq  = p.Results.ReestFreq;
W           = p.Results.WindLength;
num_workers = p.Results.NumWorkers;
assets      = p.Results.assets;
dates       = p.Results.dates;

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

% Distribution-specific setup
switch dist
    case 'norm'
        A          = [    0,     1,     1,   0.5;
                          0,    -1,     0,    -1];
        b          = [1-epsi;    0];
        lb         = [    0;     0;     0;  -Inf];
        ub         = [  Inf; 1-epsi; 1-epsi;   Inf];
        start      = [  0.2,   0.1,   0.8,   0.2];
        ll_fun     = @(pars, r) VarianceModels.univ_gjr_garch(pars, r);
        get_nu     = false;
        get_lambda = false;
        idx_nu     = NaN;
        idx_lambda = NaN;    

    case 't'
        A          = [     0,      1,      1,   0.5,    0;
                           0,     -1,      0,    -1,    0];
        b          = [1-epsi;     0];
        lb         = [     0;      0;      0;  -Inf;    2];
        ub         = [   Inf; 1-epsi; 1-epsi;   Inf; 1000];
        start      = [   0.2,    0.1,    0.8,   0.2,   10];
        ll_fun     = @(pars, r) VarianceModels.univ_gjr_garch_t(pars, r);
        get_nu     = true;
        get_lambda = false;
        idx_nu     = 5;    % [omega, alpha, beta, nu]
        idx_lambda = NaN; 

    case 'skewt'
        A          = [     0,      1,      1,   0.5,       0,    0;
                           0,     -1,      0,    -1,       0,    0];
        b          = [1-epsi;     0];
        lb         = [     0;      0;      0;  -Inf; -1+0.05;    2];
        ub         = [   Inf; 1-epsi; 1-epsi;   Inf;  1-0.05; 1000];
        start      = [   0.2,    0.1,    0.8,   0.2,       0,   10];
        ll_fun     = @(pars, r) VarianceModels.univ_gjr_garch_skew_t(pars, r);
        get_nu     = true;
        get_lambda = true;
        idx_nu     = 6;    % [omega, alpha, beta, lambda, nu]
        idx_lambda = 5;

    case 'laplace'
        A          = [     0,      1,      1,   0.5;
                           0,     -1,      0,    -1];
        b          = [1-epsi;     0];
        lb         = [     0;      0;      0;  -Inf];
        ub         = [   Inf; 1-epsi; 1-epsi;   Inf];
        start      = [   0.2,    0.1,    0.8,   0.2];
        ll_fun    = @(pars, r) VarianceModels.univ_gjr_garch_laplace(pars, r);
        get_nu     = false;
        get_lambda = false;
        idx_nu     = NaN;
        idx_lambda = NaN; 

    otherwise
        error(['estimate_gjr_garch: unknown dist ''%s''. ' ...
               'Expected ''norm'', ''t'', ''skewt'', or ''laplace''.'], dist);
end

% Pre-allocate
mu        = NaN(T, K);
GARCHpars = NaN(T, 4, K);
H_last    = NaN(T, K);
NegLL     = NaN(T, K);
std_res   = NaN(W, K, T);
nu        = NaN(T, K);
lambda    = NaN(T, K);

% Pre-compute re-estimation dates
reest_dates = t_start:reest_freq:T;
n_reest     = length(reest_dates);

% Temporary storage for parfor (index by i not t)
mu_reest        = NaN(n_reest, K);
GARCHpars_reest = NaN(n_reest, 4, K);
H_last_reest    = NaN(n_reest, K);
NegLL_reest     = NaN(n_reest, K);
std_res_reest   = NaN(W, K, n_reest);
nu_reest        = NaN(n_reest, K);
lambda_reest    = NaN(n_reest, K);


% Pre-slice R windows before parfor to avoid overhead
R_windows = cell(n_reest, K);
for i = 1:n_reest
    t = reest_dates(i);
    for k = 1:K
        R_windows{i,k} = R(t-W:t-1, k);
    end
end

% Loop 1: parfor/for over re-estimation dates
if num_workers > 1
    parfor i = 1:n_reest
        ll_fun_local = ll_fun;
        for k = 1:K
            r                        = R_windows{i,k};
            GARCH_pars               = fmincon(@(pars) ll_fun_local(pars, r), ...
                                        start, A, b, [], [], lb, ub, [], options);
            [NegLLk, H_t, mu_est]    = ll_fun_local(GARCH_pars, r);
            GARCHpars_reest(i,:,k)   = GARCH_pars(1:4);
            mu_reest(i,k)            = mu_est;
            H_last_reest(i,k)        = H_t(end);
            NegLL_reest(i,k)         = NegLLk;
            std_res_reest(:,k,i)     = (r - mu_est) ./ sqrt(H_t);
            if get_nu
                nu_reest(i,k) = GARCH_pars(idx_nu);
            end
            if get_lambda
                lambda_reest(i,k) = GARCH_pars(idx_lambda);
            end
        end
    end
else
    for i = 1:n_reest
        for k = 1:K
            r                      = R_windows{i,k};
            GARCH_pars             = fmincon(@(pars) ll_fun(pars, r), ...
                                     start, A, b, [], [], lb, ub, [], options);
            [NegLLk, H_t, mu_est]  = ll_fun(GARCH_pars, r);
            GARCHpars_reest(i,:,k) = GARCH_pars(1:4);
            mu_reest(i,k)          = mu_est;
            H_last_reest(i,k)      = H_t(end);
            NegLL_reest(i,k)       = NegLLk;
            std_res_reest(:,k,i)   = (r - mu_est) ./ sqrt(H_t);
            if get_nu
                nu_reest(i,k) = GARCH_pars(idx_nu); 
            end
            if get_lambda
                lambda_reest(i,k) = GARCH_pars(idx_lambda); 
            end
        end
    end
end

% Map back to full arrays
for i = 1:n_reest
    t                = reest_dates(i);
    GARCHpars(t,:,:) = GARCHpars_reest(i,:,:);
    mu(t,:)          = mu_reest(i,:);
    H_last(t,:)      = H_last_reest(i,:);
    NegLL(t,:)       = NegLL_reest(i,:);
    std_res(:,:,t)   = std_res_reest(:,:,i);
    nu(t,:)          = nu_reest(i,:);
    lambda(t,:)      = lambda_reest(i,:);
end

% Loop 2: sequential carry-forward between re-estimation dates
% If reest_freq = 1 this loop body never executes
for t = t_start:T
    if mod(t - t_start, reest_freq) ~= 0
        for k = 1:K
            GARCHpars(t,:,k) = GARCHpars(t-1,:,k);
            mu(t,k)          = mu(t-1,k);
            omega            = GARCHpars(t,1,k);
            alpha            = GARCHpars(t,2,k);
            beta             = GARCHpars(t,3,k);
            gamma            = GARCHpars(t,4,k); 
            epsi             = R(t-1,k) - mu(t,k);
            I                = epsi < 0;
            H_last(t,k)      = omega + (alpha + I*gamma)*epsi^2 ...
                               + beta*H_last(t-1,k); 
            new_res          = epsi / sqrt(H_last(t,k));
            std_res(:,k,t)   = [std_res(2:end,k,t-1); new_res];
            nu(t,k)          = nu(t-1,k);
            lambda(t,k)      = lambda(t-1,k);
        end
    end
end

% Pack EstOut
EstOut.model      = sprintf('GJRGARCH_%s', dist);
EstOut.dist       = dist;
EstOut.assets     = assets;
EstOut.dates      = dates;
EstOut.WindLength = W;
EstOut.ReestFreq  = reest_freq;
EstOut.GARCHpars  = GARCHpars;
EstOut.mu         = mu;
EstOut.H_last     = H_last;
EstOut.NegLL      = NegLL;
EstOut.std_res    = std_res;

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
save(filename, 'EstOut');


end

