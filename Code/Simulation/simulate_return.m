function SimOut = simulate_return(CopulaEst, Returns, varargin)
%SIMULATE_RETURN Rolling-window simulation of H-step-ahead cumulative
% returns for K assets using a GARCH-copula model.
%
%   SimOut = SIMULATE_RETURN(CopulaEst, Returns) simulates H-step-ahead
%   cumulative returns using the model specification and parameters stored
%   in CopulaEst, conditioning on observed returns in Returns.
%
%   INPUTS (required):
%       CopulaEst : Struct, output from estimate_copula. Must contain
%                   all marginal and copula estimation output including
%                   GARCHpars, mu, H_last, Rmat, and model identifiers.
%       Returns   : (TxK) matrix of observed returns
%
%   INPUTS (optional name-value):
%       'H'          : Scalar, simulation horizon in days (default: 10)
%       'M'          : Scalar, number of simulation paths (default: 1000)
%       'NumWorkers' : Scalar, number of parallel workers (default: 1)
%       'SaveDisk'   : Logical, store results to disk
%                      (default: true)
%
%   OUTPUT:
%       SimOut : Struct containing simulation output, saved to disk.
%                Fields always present:
%                  .HStepCumReturnSim - (HxMxKxT) simulated cumulative
%                                       returns per asset, horizon, path,
%                                       and time
%                  .model             - marginal model identifier string
%                  .MargDist          - marginal distribution
%                  .CopulaDist        - copula distribution
%                  .CorrModel         - correlation model
%                  .GARCHspec         - 'garch' or 'gjr'
%                  .EmpiricalPits     - logical, whether empirical PITs used
%                  .assets            - asset names
%                  .dates             - dates vector
%                  .H                 - simulation horizon
%                  .M                 - number of simulation paths
%                  .WindLength        - estimation window length
%                  .ReestFreq         - re-estimation frequency
%
%   NOTES:
%       - At each t, H-step-ahead cumulative returns are simulated by
%         drawing from the copula, mapping to marginal innovations via
%         quantileTransform, then simulating GARCH paths per asset
%       - Standard normal draws (StdN) are fixed across all t for
%         simulation consistency; only model parameters vary over time
%       - For t-copula, invGamma is redrawn at each t with rng(1) for
%         reproducibility
%       - Portfolio VaR/ES can be computed post-hoc by weight-aggregating
%         HStepCumReturnSim over the K dimension
%       - Output is saved to Output/Simulation/SimOut_<model>_<assets>.mat
%       - Supply NumWorkers > 1 to enable parallel simulation over t
%         via parfor; loop body is fully independent across t


% Name-value inputs
p = inputParser;
addParameter(p, 'H',          10);
addParameter(p, 'M',          1000);
addParameter(p, 'NumWorkers', 1);
addParameter(p, 'SaveDisk',   true);
parse(p, varargin{:});

H           = p.Results.H;
M           = p.Results.M;
num_workers = p.Results.NumWorkers;
SaveDisk    = p.Results.SaveDisk;

% Read out Copula setting
assets     = CopulaEst.assets;
dates      = CopulaEst.dates;
W          = CopulaEst.WindLength;
ReestFreq  = CopulaEst.ReestFreq;
CopulaDist = CopulaEst.CopulaDist;
CorrModel  = CopulaEst.CorrModel;

% Read out Marginal setting
Model         = CopulaEst.model;
margDist      = CopulaEst.MargDist;
empiricalPITs = CopulaEst.EmpiricalPits;
GARCHpars     = CopulaEst.GARCHpars;
mu            = CopulaEst.mu;
H_last        = CopulaEst.H_last;

% Conditional marginal objects — pass empty if not needed
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

if contains(Model, 'GJR')
    GARCHspec = 'gjr';
else
    GARCHspec = 'garch';
end

% Read out correlation matrix parameters
Rmat = CopulaEst.Rmat;

% Dimensions
[~, K, T] = size(Rmat);
t_start   = W + 1;

% Draw standard normal random numbers; shared across all t
rng(1)
StdN = RiskSim.drawZ(K, H, M);

% Conditional copula objects; pass empty if not needed
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

% Simulate H-step ahead return distribution
HStepCumReturnSim = NaN(H, M, K, T);

if num_workers > 1
    parfor t = t_start:T
        HStepCumReturnSim(:,:,:,t) = simulate_one_t(t, H, M, K, ...
            CorrModel, CopulaDist, Rmat, DCCpars, Rbar, Qmat, ...
            CopNu, StdN, empiricalPITs, StdRes_t, ...
            margDist, margNu, margLambda, GARCHpars, ...
            Returns, H_last, mu, GARCHspec);
    end
else
    for t = t_start:T
        HStepCumReturnSim(:,:,:,t) = simulate_one_t(t, H, M, K, ...
            CorrModel, CopulaDist, Rmat, DCCpars, Rbar, Qmat, ...
            CopNu, StdN, empiricalPITs, StdRes_t, ...
            margDist, margNu, margLambda, GARCHpars, ...
            Returns, H_last, mu, GARCHspec);
    end
end

% Pack SimOut
SimOut.HStepCumReturnSim = HStepCumReturnSim;
SimOut.assets            = assets;
SimOut.dates             = dates;
SimOut.H                 = H;
SimOut.M                 = M;
SimOut.WindLength        = W;
SimOut.ReestFreq         = ReestFreq;
SimOut.CorrModel         = CorrModel;
SimOut.CopulaDist        = CopulaDist;
SimOut.MargDist          = margDist;
SimOut.model             = Model;
SimOut.EmpiricalPits     = empiricalPITs;
SimOut.GARCHspec         = GARCHspec;

% Save to disk
sim_name = Model;
if empiricalPITs
    sim_name = [sim_name, '_empirical'];
end
assets_str = strjoin(assets, '_');

if SaveDisk
    save(sprintf('Output/Simulation/SimOut_%s_%s.mat', sim_name, ...
         assets_str), 'SimOut');
end


end


% Wrapper function to handle for versus parfor loop over t
function Rsim_t = simulate_one_t(t, H, M, K, ...
    CorrModel, CopulaDist, Rmat, DCCpars, Rbar, Qmat, ...
    CopNu, StdN, empiricalPITs, StdRes_t, ...
    margDist, margNu, margLambda, GARCHpars, ...
    Returns, H_last, mu, GARCHspec)

    % Simulate uniforms from copula
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
    
    % Simulate cumulative h-step ahead returns for each asset
    n_pars      = size(GARCHpars,2);
    garch_param = reshape(GARCHpars(t,:,:), n_pars, K)'; 
    Rsim_t      = NaN(H, M, K);
    for k = 1:K
        pars          = garch_param(k,:);
        Rsim_t(:,:,k) = RiskSim.hStepSimGarch(pars, H, M, ...
                                              Returns(t-1,k), ...
                                              H_last(t,k), mu(t,k), ...
                                              'z', z_sim(:,:,k), ...
                                              'model', GARCHspec);
    end

end