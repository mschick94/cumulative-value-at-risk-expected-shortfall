function EstOut = estimate_copula(MarginalEst, varargin)
%ESTIMATE_COPULA Rolling-window estimation of a copula model for K assets
% using pre-estimated GARCH marginals.
%
%   EstOut = ESTIMATE_COPULA(MarginalEst) estimates a Gaussian CCC copula
%   using normal marginal PITs with default settings.
%
%   EstOut = ESTIMATE_COPULA(MarginalEst, 'copula_dist', 't', ...
%                            'corr_model', 'DCC') estimates a Student-t
%   DCC copula.
%
%   INPUTS (required):
%       MarginalEst : Struct, output from estimate_garch or
%                     estimate_gjr_garch. Must contain fields:
%                     model, dist, assets, dates, WindLength, ReestFreq,
%                     std_res, GARCHpars, mu, H_last.
%                     Also nu (t, skewt) and lambda (skewt) if applicable.
%
%   INPUTS (optional name-value):
%       'copula_dist'    : String, copula innovation distribution
%                          'norm' - Gaussian copula (default)
%                          't'    - Student-t copula
%       'corr_model'     : String, correlation model
%                          'CCC' - Const. Conditional Correlation (default)
%                          'DCC' - Dynamic Conditional Correlation
%       'empirical_pits' : Logical, use empirical PITs instead of
%                          parametric (default: false). Always uses
%                          Gaussian copula. Stores std_res in EstOut.
%       'NumWorkers'     : Scalar, number of parallel workers (default: 1)
%       'SaveDisk'       : Logical, store results to disk
%                          (default: true)
%
%   OUTPUT:
%       EstOut : Struct containing all estimation output, saved to disk.
%                Fields always present:
%                  .model         - model identifier string
%                  .MargDist      - marginal distribution
%                  .EmpiricalPits - logical, whether empirical PITs used
%                  .CopulaDist    - copula distribution
%                  .CorrModel     - correlation model
%                  .assets        - asset names
%                  .dates         - dates vector
%                  .WindLength    - estimation window length
%                  .ReestFreq     - re-estimation frequency
%                  .NegLLCop      - (Tx1) copula negative log-likelihood
%                  .Rmat          - (KxKxT) correlation matrix
%                  .GARCHpars     - (Tx3xK) GARCH parameter estimates
%                  .mu            - (TxK) mean estimates
%                  .H_last        - (TxK) last filtered variance
%                  .margDist      - marginal distribution string
%                Fields present for DCC:
%                  .Rbar          - (KxKxT) unconditional correlation
%                  .Qmat          - (KxKxT) DCC Q matrix
%                  .DCCpars       - (Tx2) DCC alpha and beta estimates
%                Fields present for t-copula:
%                  .CopulaNu      - (Tx1) copula degrees of freedom
%                Fields present for t and skewt marginals:
%                  .nu            - (TxK) degrees of freedom
%                Fields present for skewt marginals:
%                  .lambda        - (TxK) skewness parameters
%                Fields present for empirical PITs:
%                  .std_res       - (WxKxT) rolling standardized residuals
%
%   NOTES:
%       - Copula is re-estimated every ReestFreq days using the in-sample
%         window of standardized residuals from MarginalEst
%       - Between re-estimation dates, CCC carries forward Rmat and DCC
%         filters Qmat and Rmat forward using daily standardized residuals
%       - Output is saved to Output/Estimation/EstOut_<model>_<assets>.mat
%       - Supply NumWorkers > 1 to enable parallel estimation across
%         re-estimation dates via parfor


% Name-value inputs
p = inputParser;
addParameter(p, 'copula_dist',    'norm');
addParameter(p, 'corr_model',     'CCC');
addParameter(p, 'empirical_pits', false);
addParameter(p, 'NumWorkers',     1);
addParameter(p, 'SaveDisk',       true);
parse(p, varargin{:});

copula_dist    = p.Results.copula_dist;
CorrModel      = p.Results.corr_model;
empirical_pits = p.Results.empirical_pits;
num_workers    = p.Results.NumWorkers;
SaveDisk       = p.Results.SaveDisk;

% Read out setting
margModel = MarginalEst.model;
margDist  = MarginalEst.dist; 
assets    = MarginalEst.assets; 
dates     = MarginalEst.dates;
W         = MarginalEst.WindLength;
ReestFreq = MarginalEst.ReestFreq;

% Read out history of standardized residuals
StdRes = MarginalEst.std_res;

% Dimensions
[~, K, T]  = size(StdRes(:,:,:));
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

% Distribution-specific setup: Marginals
switch margDist
    case 'norm'
        margNu     = NaN(T, K);
        margLambda = NaN(T, K);

    case 't'
        margNu     = MarginalEst.nu;
        margLambda = NaN(T, K);

    case 'skewt'
        margNu     = MarginalEst.nu;
        margLambda = MarginalEst.lambda;

    case 'laplace'
        margNu     = NaN(T, K);
        margLambda = NaN(T, K);

    otherwise
        error(['estimate_copula: unknown marginal dist ''%s''. ' ...
               'Expected ''norm'', ''t'', ''skewt'', or ''laplace''.'], margDist);
end

% Distribution-specific setup: Copula
switch copula_dist
    % Gaussian Copula
    case 'norm'     
        switch CorrModel
            case 'CCC'
                % No likelihood estimation; R computed directly from u_mat
                A_cop      = []; 
                b_cop      = [];
                lb_cop     = [];
                ub_cop     = [];
                start_cop  = [];
                ll_fun     = [];
                get_nu_cop = false;

            case 'DCC'
                A_cop      = [     1,      1]; 
                b_cop      =  1-epsi;
                lb_cop     = [     0;      0];
                ub_cop     = [1-epsi; 1-epsi];
                start_cop  = [   0.1,    0.8];
                ll_fun     = @(pars, u) CopulaModel.CopulaDCC(pars, u);
                get_nu_cop = false;

            otherwise
                error(['estimate_copula: unknown correlation model ''%s''. ' ...
                       'Expected ''CCC'' or ''DCC''.'], CorrModel);
        end

    % Student-tCopula
    case 't'
        switch CorrModel
            case 'CCC'
                % Estimate nu only
                A_cop      = [];
                b_cop      = [];
                lb_cop     = 2.1;
                ub_cop     = 1e8;
                start_cop  = 10;
                ll_fun     = @(pars, u) CopulaModel.CopulaCCC(pars, u, ...
                                                              'dist', 't');
                get_nu_cop = true;

            case 'DCC'
                % Estimate alpha, beta, nu
                A_cop      = [     1,       1,      0];
                b_cop      =  1-epsi;
                lb_cop     = [     0;      0;    2.1];
                ub_cop     = [1-epsi; 1-epsi;    1e8];
                start_cop  = [   0.1,    0.8,     10];
                ll_fun     = @(pars, u) CopulaModel.CopulaDCC(pars, u, ...
                                                              'dist', 't');
                get_nu_cop = true;

            otherwise
                error(['estimate_copula: unknown correlation model ''%s''. ' ...
                       'Expected ''CCC'' or ''DCC''.'], CorrModel);
        end
        
    otherwise
        error(['estimate_copula: unknown copula dist ''%s''. ' ...
               'Expected ''norm'' or ''t''.'], copula_dist);
end


% Pre-allocate
Rmat      = NaN(K, K, T);
Qmat      = NaN(K, K, T); % DCC
Rbar      = NaN(K, K, T); % DCC
cop_nu    = NaN(T, 1);
dcc_pars  = NaN(T, 2);
cop_NegLL = NaN(T, 1);


% Pre-compute re-estimation dates
reest_dates      = t_start:ReestFreq:T;
n_reest          = length(reest_dates);
margNu_reest     = margNu(reest_dates, :);
margLambda_reest = margLambda(reest_dates, :);

% Temporary storage for parfor (index by i not t)
Rmat_reest      = NaN(K,K,n_reest);
Qmat_reest      = NaN(K,K,n_reest);
Rbar_reest      = NaN(K,K,n_reest);
cop_nu_reest    = NaN(n_reest, 1);
dcc_pars_reest  = NaN(n_reest, 2);
cop_NegLL_reest = NaN(n_reest, 1);


% Pre-slice StdRes windows before parfor to avoid overhead
StdRes_windows = cell(n_reest, 1);
for i = 1:n_reest
    t = reest_dates(i);
    StdRes_windows{i} = MarginalEst.std_res(:,:,t);   % (W x K)
end


% Loop 1: parfor/for over re-estimation dates
results = cell(n_reest,1);
if num_workers > 1
    parfor i = 1:n_reest
        results{i} = estimate_copula_window(...
                        StdRes_windows{i}, ll_fun, margDist, ...
                        margNu_reest(i,:), margLambda_reest(i,:), ...
                        empirical_pits, CorrModel, isempty(ll_fun), ...
                        start_cop, A_cop, b_cop, lb_cop, ub_cop, ...
                        get_nu_cop, options);
    end
else
    for i = 1:n_reest
        results{i} = estimate_copula_window(...
                        StdRes_windows{i}, ll_fun, margDist, ...
                        margNu_reest(i,:), margLambda_reest(i,:), ...
                        empirical_pits, CorrModel, isempty(ll_fun), ...
                        start_cop, A_cop, b_cop, lb_cop, ub_cop, ...
                        get_nu_cop, options);
    end
end

% Unpack results
for i = 1:n_reest
    Rmat_reest(:,:,i)  = results{i}.Rmat;
    cop_nu_reest(i)    = results{i}.cop_nu;
    cop_NegLL_reest(i) = results{i}.NegLL;
    if ~isscalar(results{i}.Qmat)
        Qmat_reest(:,:,i) = results{i}.Qmat;
    end
    if ~isscalar(results{i}.Rbar)
        Rbar_reest(:,:,i) = results{i}.Rbar;
    end
    if ~isnan(results{i}.dcc_pars)
        dcc_pars_reest(i,:) = results{i}.dcc_pars;
    end
end

% Map back to full arrays
for i = 1:n_reest
    t             = reest_dates(i);
    Rmat(:,:,t)   = Rmat_reest(:,:,i);
    Qmat(:,:,t)   = Qmat_reest(:,:,i);
    Rbar(:,:,t)   = Rbar_reest(:,:,i);
    cop_nu(t)     = cop_nu_reest(i);    
    dcc_pars(t,:) = dcc_pars_reest(i,:);    
    cop_NegLL(t)  = cop_NegLL_reest(i);
end

% Loop 2: sequential carry-forward between re-estimation dates
% If reest_freq = 1 this loop body never executes
for t = t_start:T
    if mod(t - t_start, ReestFreq) ~= 0
        cop_nu(t) = cop_nu(t-1);  
        switch CorrModel
            case 'CCC'
                Rmat(:,:,t) = Rmat(:,:,t-1);

            case 'DCC'
                dcc_pars(t,:) = dcc_pars(t-1,:); 

                % Step 1: Transform raw residuals to uniform via marginals
                std_res_t = StdRes(end,:,t);   
                switch margDist
                    case 'norm'
                        u_t = CopulaModel.pits(std_res_t, 'dist', 'norm');
                    case 't'
                        u_t = CopulaModel.pits(std_res_t, 'dist', 't', ...
                                               'nu', margNu(t,:));
                    case 'skewt'
                        u_t = CopulaModel.pits(std_res_t, ...
                                               'dist', 'skewt', ...
                                               'nu', margNu(t,:), ...
                                               'lambda', margLambda(t,:));
                    case 'laplace'
                        u_t = CopulaModel.pits(std_res_t, ...
                                               'dist', 'laplace');
                end
                
                % Step 2: Transform uniforms to standardized residuals
                switch copula_dist
                    case 'norm'
                        z_t = norminv(u_t);
                    case 't'
                        nu_cop = cop_nu(t);
                        z_t    = sqrt((nu_cop-2)/nu_cop) * tinv(u_t, nu_cop);
                end

                % DCC forward iteration
                Rbar(:,:,t) = Rbar(:,:,t-1);
                alpha       = dcc_pars(t,1);
                beta        = dcc_pars(t,2);
                Qmat(:,:,t) = Rbar(:,:,t-1) * (1 - alpha - beta) ...
                              + alpha * (z_t' * z_t) ...
                              + beta * Qmat(:,:,t-1);
                d           = diag(Qmat(:,:,t)).^(-0.5);
                Rmat(:,:,t) = (d*d') .* Qmat(:,:,t);
        end  
    end
end

% Pack EstOut
if empirical_pits
    model_name = sprintf('%s_%s_%s_empirical', margModel, CorrModel, copula_dist);
else
    model_name = sprintf('%s_%s_%s', margModel, CorrModel, copula_dist);
end

EstOut.model         = model_name;
EstOut.MargDist      = margDist;
EstOut.EmpiricalPits = empirical_pits;
EstOut.CopulaDist    = copula_dist;
EstOut.CorrModel     = CorrModel;
EstOut.assets        = assets;
EstOut.dates         = dates;
EstOut.WindLength    = W;
EstOut.ReestFreq     = ReestFreq;
EstOut.NegLLCop      = cop_NegLL;
EstOut.Rmat          = Rmat;


% Distribution- and model-specific fields
if strcmp('DCC', CorrModel)
    EstOut.Rbar    = Rbar;
    EstOut.Qmat    = Qmat;
    EstOut.DCCpars = dcc_pars;
end
if get_nu_cop
    EstOut.CopulaNu = cop_nu;
end

% Storing relevant objects for the marginal distribution
EstOut.GARCHpars = MarginalEst.GARCHpars;
EstOut.mu        = MarginalEst.mu;
EstOut.H_last    = MarginalEst.H_last;
EstOut.margDist  = MarginalEst.dist;
if ismember(MarginalEst.dist, {'t', 'skewt'})
    EstOut.nu = MarginalEst.nu;
end
if strcmp(MarginalEst.dist, 'skewt')
    EstOut.lambda = MarginalEst.lambda;
end

% Only store std_res if empirical
if empirical_pits
    EstOut.std_res = MarginalEst.std_res;   % (WxKxT) large but necessary
end

if isempty(assets)
    filename = sprintf('Output/Estimation/Copula/EstOut_%s.mat', ...
                       model_name);
else
    filename = sprintf('Output/Estimation/Copula/EstOut_%s_%s.mat', ...
                       model_name, strjoin(assets, '_'));
end

if SaveDisk
    save(filename, 'EstOut');
end


end


% Wrapper function to handle for versus parfor loop over t
function result = estimate_copula_window(std_res, ll_fun, margDist, ...
                                         margNu_i, margLambda_i, ...
                                         empirical_pits, CorrModel, ...
                                         isempty_ll, start_cop, A_cop, ...
                                         b_cop, lb_cop, ub_cop, ...
                                         get_nu_cop, options)
    % PITs
    if empirical_pits
        u_mat = CopulaModel.pits(std_res, 'dist', 'empirical');
    else
        switch margDist
            case 'norm'
                u_mat = CopulaModel.pits(std_res, 'dist', 'norm');
            case 't'
                u_mat = CopulaModel.pits(std_res, 'dist', 't', ...
                                         'nu', margNu_i);
            case 'skewt'
                u_mat = CopulaModel.pits(std_res, 'dist', 'skewt', ...
                                         'nu', margNu_i, ...
                                         'lambda', margLambda_i);
            case 'laplace'
                u_mat = CopulaModel.pits(std_res, 'dist', 'laplace');
        end
    end

    % Estimate copula
    if isempty_ll
        [result.NegLL, ~, result.Rmat] = CopulaModel.CopulaCCC([], u_mat);
        result.Qmat     = NaN;
        result.Rbar     = NaN;
        result.cop_nu   = NaN;
        result.dcc_pars = NaN;
    else
        cop_pars = fmincon(@(pars) ll_fun(pars, u_mat), start_cop, ...
                           A_cop, b_cop, [], [], lb_cop, ub_cop, [], ...
                           options);
        switch CorrModel
            case 'CCC'
                [result.NegLL, ~, result.Rmat] = ll_fun(cop_pars, u_mat);
                result.cop_nu   = cop_pars;
                result.Qmat     = NaN;
                result.Rbar     = NaN;
                result.dcc_pars = NaN;
            case 'DCC'
                [result.NegLL, ~, R_last, Q_last, R_bar] = ...
                    ll_fun(cop_pars, u_mat);
                result.Rmat     = R_last;
                result.Qmat     = Q_last;
                result.Rbar     = R_bar;
                result.dcc_pars = cop_pars(1:2);
                if get_nu_cop
                    result.cop_nu = cop_pars(end);
                else
                    result.cop_nu = NaN;
                end
        end
    end
end
