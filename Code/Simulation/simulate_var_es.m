function VaRESOut = simulate_var_es(CopulaEst, Returns, PFweights, ...
                                    alpha, varargin)
%SIMULATE_VAR_ES Rolling-window simulation of H-step-ahead cumulative
% portfolio VaR and ES using a GARCH/HEAVY-copula model.
%
%   VaRESOut = SIMULATE_VAR_ES(CopulaEst, Returns, PFweights, alpha)
%   simulates H-step-ahead cumulative returns and computes portfolio VaR
%   and ES on the fly without storing the full simulation array.
%
%   INPUTS (required):
%       CopulaEst  : Struct, output from estimate_copula. Must contain
%                    all marginal and copula estimation output including
%                    GARCHpars, mu, H_last, Rmat, and model identifiers.
%       Returns    : (TxK) matrix of observed returns
%       PFweights  : (Kx1) or (1xK) vector of portfolio weights
%       alpha      : p-vector, significance levels e.g. 0.025 for 97.5% VaR
%
%   INPUTS (optional name-value):
%       'H'          : Scalar, simulation horizon in days (default: 10)
%       'M'          : Scalar, number of simulation paths (default: 1000)
%       'NumWorkers' : Scalar, number of parallel workers (default: 1)
%       'SetSeed'    : Logical, set seed for reproducibility 
%                      (default: true)
%
%   OUTPUT:
%       VaRESOut : Struct containing:
%                  .VaR        - (T x H x 1 x P) VaR forecasts, negative values
%                  .ES         - (T x H x 1 x P) ES forecasts, negative values
%                  .alpha      - significance level
%                  .H          - simulation horizon
%                  .M          - number of simulation paths
%                  .model      - model identifier string
%                  .MargDist   - marginal distribution
%                  .CopulaDist - copula distribution
%                  .CorrModel  - correlation model
%                  .GARCHspec  - 'garch' or 'gjr'
%                  .EmpiricalPits - logical, whether empirical PITs used
%                  .assets     - asset names
%                  .dates      - dates vector
%                  .PFweights  - (1xK) portfolio weights used
%                  .WindLength - estimation window length
%                  .ReestFreq  - re-estimation frequency
%
%   NOTES:
%       - VaR and ES are reported as negative numbers (left tail)
%       - Standard normal draws are fixed across t for simulation
%         consistency; only model parameters vary over time
%       - For t-copula, invGamma redrawn at each t with rng(1)
%       - Portfolio VaR/ES aggregated using PFweights at each t
%       - Supply NumWorkers > 1 to enable parallel simulation over t

% Name-value inputs
p = inputParser;
p.KeepUnmatched = true;
addParameter(p, 'H',          10);
addParameter(p, 'M',          1000);
addParameter(p, 'NumWorkers', 1);
addParameter(p, 'SetSeed',    true);
addParameter(p, 'RV',         []);
parse(p, varargin{:});

H           = p.Results.H;
M           = p.Results.M;
RV          = p.Results.RV;
num_workers = p.Results.NumWorkers;

% Transform weight vector to row vector
if size(PFweights, 1) > size(PFweights, 2)
    PFweights = PFweights';
end

% Read out Copula setting
assets     = CopulaEst.assets;
dates      = CopulaEst.dates;
W          = CopulaEst.WindLength;
ReestFreq  = CopulaEst.ReestFreq;
CopulaDist = CopulaEst.CopulaDist;
CorrModel  = CopulaEst.CorrModel;

% Read out Marginal setting
Model = CopulaEst.model;
if contains(Model, 'HEAVY')
    GARCHspec = 'heavy';
    GARCHpars.HEAVY_r_pars  = CopulaEst.HEAVY_r_pars;
    GARCHpars.HEAVY_RV_pars = CopulaEst.HEAVY_RV_pars;
    GARCHpars.Tau_last      = CopulaEst.Tau_last;

    % Read out RV measure
    K        = size(Returns, 2);
    diag_idx = 1 + cumsum([0, K:-1:2]); 
    RV       = RV(:, diag_idx); 

elseif contains(Model, 'GJR')
    GARCHspec = 'gjr';
    GARCHpars = CopulaEst.GARCHpars;
else
    GARCHspec = 'garch';
    GARCHpars = CopulaEst.GARCHpars;
end
margDist      = CopulaEst.MargDist;
empiricalPITs = CopulaEst.EmpiricalPits;
mu            = CopulaEst.mu;
H_last        = CopulaEst.H_last;

% Conditional marginal objects
if empiricalPITs
    StdRes_t = CopulaEst.std_res;
else
    StdRes_t = [];
end

switch margDist
    case 't'
        margNu     = CopulaEst.nu;
        margLambda = [];
    case 'skewt'
        margNu     = CopulaEst.nu;
        margLambda = CopulaEst.lambda;
    otherwise
        margNu     = [];
        margLambda = [];
end

% Read out correlation matrix parameters
Rmat = CopulaEst.Rmat;

% Dimensions
[~, K, T] = size(Rmat);
t_start   = W + 1;

% Save caller's random stream state
stream_state = rng;

% Draw standard normal random numbers — shared across all t
if p.Results.SetSeed
    rng(1)
end
StdN = RiskSim.drawZ(K, H, M);

% Conditional copula objects
if strcmp(CopulaDist, 't')
    CopNu = CopulaEst.CopulaNu;
else
    CopNu = [];
end

if contains(CorrModel, 'DCC')
    DCCpars = CopulaEst.DCCpars;
    Rbar    = CopulaEst.Rbar;
    Qmat    = CopulaEst.Qmat;
else
    DCCpars = [];
    Rbar    = [];
    Qmat    = [];
end

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

% Pre-allocate VaR and ES only
P = length(alpha);
VaR = NaN(T, 1, P); % NaN(T, H, P); if all h=1:H needed 
ES  = NaN(T, 1, P); % NaN(T, 1, P); Computationally more costly!

if num_workers > 1
    parfor t = t_start:T
        % Simulate H-step ahead returns at time t
        Rsim_t = simulate_one_t(t, H, M, K, CorrModel, CopulaDist, ...
            Rmat, DCCpars, Rbar, Qmat, CopNu, StdN, empiricalPITs, ...
            StdRes_t, margDist, margNu, margLambda, GARCHpars, Returns, ...
            H_last, mu, GARCHspec, RV);

        % % Aggregate to portfolio (H x M)  and compute VaR and ES
        % PFRet_t = reshape(sum(Rsim_t.*reshape(PFweights,1,1,K), 3), H, M);
        % sorted   = sort(PFRet_t, 2);
        % alpha_local = alpha;
        % for p = 1:P
        %     idx         = max(1, floor(alpha_local(p) * M));
        %     VaR(t,:,p)  = sorted(:, idx);
        %     ES(t,:,p)   = mean(sorted(:, 1:idx), 2);
        % end

        % Only compute portfolio return at h=H
        PFRet_H  = squeeze(sum(Rsim_t(H,:,:).*reshape(PFweights,1,1,K),3));
        
        % Sort and compute VaR/ES per horizon
        sorted_H = sort(PFRet_H);
        alpha_local = alpha;
        for p = 1:P
            idx        = max(1, floor(alpha_local(p) * M));
            VaR(t,1,p) = sorted_H(idx);
            ES(t,1,p)  = mean(sorted_H(1:idx));
        end

    end
else
    for t = t_start:T
        Rsim_t   = simulate_one_t(t, H, M, K, CorrModel, CopulaDist, ...
            Rmat, DCCpars, Rbar, Qmat, CopNu, StdN, empiricalPITs, ...
            StdRes_t, margDist, margNu, margLambda, GARCHpars, Returns, ...
            H_last, mu, GARCHspec, RV);

        % % Aggregate to portfolio (H x M) and compute VaR and ES
        % PFRet_t  = reshape(sum(Rsim_t.*reshape(PFweights,1,1,K), 3), H, M);
        % sorted = sort(PFRet_t, 2);
        % for p = 1:P
        %     idx         = max(1, floor(alpha(p) * M));
        %     VaR(t,:,p)  = sorted(:, idx);
        %     ES(t,:,p)   = mean(sorted(:, 1:idx), 2);
        % end

        % Only compute portfolio return at h=H
        PFRet_H  = squeeze(sum(Rsim_t(H,:,:).*reshape(PFweights,1,1,K),3));
        
        % Sort and compute VaR/ES per horizon
        sorted_H = sort(PFRet_H);
        for p = 1:P
            idx        = max(1, floor(alpha(p) * M));
            VaR(t,1,p) = sorted_H(idx);
            ES(t,1,p)  = mean(sorted_H(1:idx));
        end

    end
end

% Pack VaRESOut
VaRESOut.VaR           = reshape(VaR, T, 1, 1, P); %reshape(VaR, T, 1, H, P);
VaRESOut.ES            = reshape(ES, T, 1, 1, P); 
VaRESOut.alpha         = alpha;
VaRESOut.H             = H;
VaRESOut.M             = M;
VaRESOut.model         = Model;
VaRESOut.MargDist      = margDist;
VaRESOut.CopulaDist    = CopulaDist;
VaRESOut.CorrModel     = CorrModel;
VaRESOut.GARCHspec     = GARCHspec;
VaRESOut.EmpiricalPits = empiricalPITs;
VaRESOut.assets        = assets;
VaRESOut.dates         = dates;
VaRESOut.PFweights     = PFweights;
VaRESOut.WindLength    = W;
VaRESOut.ReestFreq     = ReestFreq;


% Restore caller's random stream state
rng(stream_state);

end


% Wrapper function to handle for versus parfor loop over t
function Rsim_t = simulate_one_t(t, H, M, K, CorrModel, CopulaDist, ...
    Rmat, DCCpars, Rbar, Qmat, CopNu, StdN, empiricalPITs, StdRes_t, ...
    margDist, margNu, margLambda, GARCHpars, Returns, H_last, mu, ...
    GARCHspec, RV)

% For simulation with K = 1, HEAVY-Model not implemented so far
if K == 1 && strcmp(GARCHspec, 'heavy')
    error(['simulate_one_t: HEAVY model requires K>1. Use ' ...
           'hStepSimHeavy directly for K=1']);
end

% Simulate uniforms from copula
if K == 1 && strcmp(margDist, 'norm')
    z_sim = reshape(squeeze(StdN), H, M, 1);
else
    switch CorrModel
        case 'CCC'
            switch CopulaDist
                case 'norm'
                    u_sim = RiskSim.simulateCopulaCCC(Rmat(:,:,t), H, ...
                        M, K, 'StdNormDraws', StdN);
                case 't'
                    rng(1)
                    nuCopula = CopNu(t,1);
                    invGamma = sqrt(nuCopula ./ ...
                                    (2.*randg(nuCopula./2, H, M, 1)));
                    u_sim    = RiskSim.simulateCopulaCCC(Rmat(:,:,t), ...
                        H, M, K, 'StdNormDraws', StdN, ...
                        'invGamma', invGamma);
            end
        case 'DCC'
            parsDCC    = DCCpars(t,:);
            R_dcc_bar  = Rbar(:,:,t);
            R_dcc_last = Rmat(:,:,t);
            Q_dcc_last = Qmat(:,:,t);
            switch CopulaDist
                case 'norm'
                    u_sim = RiskSim.simulateCopulaDCC(parsDCC, ...
                        R_dcc_bar, R_dcc_last, Q_dcc_last, H, M, K, ...
                        'dist', 'norm', 'StdNormDraws', StdN);
                case 't'
                    rng(1)
                    nuCopula = CopNu(t,1);
                    invGamma = sqrt(nuCopula ./ ...
                                    (2.*randg(nuCopula./2, H, M, 1)));
                    u_sim    = RiskSim.simulateCopulaDCC(parsDCC, ...
                        R_dcc_bar, R_dcc_last, Q_dcc_last, H, M, K, ...
                        'dist', 't', 'nu', nuCopula, ...
                        'StdNormDraws', StdN, 'invGamma', invGamma);
            end
    end
    
    % Map uniform draws to marginals
    if empiricalPITs
        z_sim = CopulaModel.quantileTransform(u_sim, ...
            'dist', 'empirical', 'std_res', StdRes_t(:,:,t));
    else
        switch margDist
            case 'norm'
                z_sim = CopulaModel.quantileTransform(u_sim);
            case 't'
                z_sim = CopulaModel.quantileTransform(u_sim, ...
                    'dist', 't', 'nu', margNu(t,:));
            case 'skewt'
                z_sim = CopulaModel.quantileTransform(u_sim, ...
                    'dist', 'skewt', 'nu', margNu(t,:), ...
                    'lambda', margLambda(t,:));
            case 'laplace'
                z_sim = CopulaModel.quantileTransform(u_sim, ...
                    'dist', 'laplace');
        end
    end
end

% Simulate cumulative h-step ahead returns for each asset
Rsim_t = NaN(H, M, K);

if strcmp(GARCHspec, 'heavy')
    % HEAVY simulation
    for k = 1:K
        pars_r   = reshape(GARCHpars.HEAVY_r_pars(t,:,k),  1, []);
        pars_rv  = reshape(GARCHpars.HEAVY_RV_pars(t,:,k), 1, []);
        Tau_last = GARCHpars.Tau_last;
        Rsim_t(:,:,k) = RiskSim.hStepSimHeavy(pars_r, pars_rv, H, M, ...
                                            RV(t-1,k), H_last(t,k), ...
                                            Tau_last(t,k), mu(t,k), ...
                                            'z', z_sim(:,:,k));
    end
else
    % GARCH/GJR simulation
    n_pars      = size(GARCHpars, 2);
    garch_param = reshape(GARCHpars(t,:,:), n_pars, K)';
    for k = 1:K
        pars          = garch_param(k,:);
        Rsim_t(:,:,k) = RiskSim.hStepSimGarch(pars, H, M, ...
                                              Returns(t-1,k), ...
                                              H_last(t,k), mu(t,k), ...
                                              'z', z_sim(:,:,k), ...
                                              'model', GARCHspec);
    end
end

end