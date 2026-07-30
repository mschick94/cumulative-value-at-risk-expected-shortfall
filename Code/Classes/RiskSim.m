classdef RiskSim
    %RISKSIM A container class for simulating forward-looking return
    % distributions and computing risk measures.
    %
    %   Contains static methods for h-step ahead Monte Carlo simulation
    %   of return paths under a fitted GARCH(1,1) model, and computation
    %   of Value-at-Risk (VaR) and Expected Shortfall (ES) from the
    %   simulated distributions.
    %
    %   Supports normal, Student-t, and Hansen skew-t innovation
    %   distributions via an optional name-value interface. Innovation
    %   draws can be supplied externally for reproducibility and
    %   efficiency across multiple evaluation days.
    %
    %   Usage:
    %       [VaR, ES, r_sim, h_sim] = RiskSim.hStepSimGarch(pars, H, M, ...
    %                                     r_last, h_last, mu)
    %       z = RiskSim.drawZ(H, M, 'dist', 't', 'nu', 8)

    methods(Static)


%==========================================================================
%       Simulation of h-step-ahead return distribution
%==========================================================================

        function [R_cum, r_sim, h_sim, VaR, ES] = hStepSimGarch(pars, ...
                                        H, M, r_last, h_last, mu, varargin)
        %HSTEPSIMGARCH Simulate h-step ahead return paths from a GARCH(1,1)
        % model and compute cumulative VaR and ES at each horizon.
        %
        %   [R_cum, r_sim, h_sim, VaR, ES] = HSTEPSIMGARCH(pars, H, M, ...
        %                               r_last, h_last, conf, mu, varargin)
        %   simulates M paths of length H forward from the last observed
        %   return and variance, iterating the GARCH(1,1) variance equation
        %   using the simulated shocks. VaR and ES are computed on
        %   cumulative returns.
        %
        %   INPUTS (required):
        %       pars   : (3x1) parameter vector [omega; alpha; beta] OR:
        %                (4x1) parameter vector [omega; alpha; beta; gamma]
        %       H      : Scalar, forecast horizon (number of steps ahead)
        %       M      : Scalar, number of simulation paths
        %       r_last : Scalar, last observed return r_T
        %       h_last : Scalar, last filtered conditional variance h_T
        %
        %   INPUTS (optional positional):
        %       mu     : Scalar, mean of return series (default: 0)
        %
        %   INPUTS (optional name-value):
        %       'dist'   : String, innovation distribution. Options:
        %                  'norm'      - standard normal (default)
        %                  't'         - standardized Student-t (zero mean,
        %                                unit variance)
        %                  'skewt'     - standardized Hansen Student-t 
        %                                (zero mean, uni variance)
        %                  'laplace'   - Laplace distribution        
        %                  'empirical' - empirical inverse CDF via
        %                                order statistics (bootstrap)        
        %       'model'  : String, GARCH specification. Options:
        %                  'garch' - standard GARCH(1,1) model (default)
        %                  'gjr'   - GJR-GARCH model
        %       'nu'     : Scalar, degrees of freedom for Student-t
        %                  (required if dist = 't' ot 'skewt', ignored 
        %                  otherwise)
        %       'lambda' : Scalar, skewness parameter for Hansen's skew-t
        %                  (required if dist = 'skewt', ignored otherwise)        
        %       'z'    : (HxM) matrix of pre-drawn standardized innovations
        %                (default: drawn internally based on dist)
        %                Supply externally to reuse draws across days.
        %                Must match the specified distribution.
        %       'conf' : Scalar, quantile level for VaR and ES
        %                (default: 0.01)        
        %
        %   OUTPUTS:
        %       R_cum  : (HxM) matrix of simulated cumulative returns per 
        %                step per path
        %       r_sim  : (HxM) matrix of simulated returns per step per
        %                path
        %       h_sim  : (HxM) matrix of simulated conditional variances
        %                per step per path        
        %       VaR    : (Hx1) vector of VaR estimates at each horizon
        %                h = 1,...,H at quantile level conf. Negative
        %                number (left tail). Multiply by -1 for reporting
        %                as a positive loss.
        %       ES     : (Hx1) vector of ES estimates at each horizon
        %                h = 1,...,H. Mean of simulated cumulative returns
        %                below VaR(conf). Negative number. Multiply by -1
        %                for reporting.
        %
            
            % Optional positional inputs
            if nargin < 6
                mu   = 0; % Mean of return series
            end   
            
            % Name-value inputs
            p = inputParser;
            addParameter(p, 'conf',   0.01);
            addParameter(p, 'dist',   'norm');
            addParameter(p, 'model',  'garch');
            addParameter(p, 'nu',     NaN);
            addParameter(p, 'lambda', NaN);
            addParameter(p, 'z',      []);
            parse(p, varargin{:});
            
            conf   = p.Results.conf;
            dist   = p.Results.dist;
            model  = p.Results.model;
            nu     = p.Results.nu;
            lambda = p.Results.lambda;
            z      = p.Results.z;
            K      = 1;


            % Simulated innovations — draw internally if z not supplied
            if isempty(z)
                rng(1); % only set seed when drawing internally
                z = RiskSim.drawZ(H, M, K, dist, nu, lambda);
            else
                if ~isequal(size(z), [H, M])
                    error(['hStepSimGarch: z must be (%d x %d) but got ' ...
                           '(%d x %d)'], H, M, size(z,1), size(z,2));
                end
            end

            % GARCH(1,1) parameters
            omega = pars(1);
            alpha = pars(2);
            beta  = pars(3);

            % GJR-GARCH
            if strcmp(model, 'gjr')
                gamma = pars(4);
            end

            expected = 3 + strcmp(model, 'gjr');
            if length(pars) ~= expected
                warning('hStepSimGarch: model ''%s'' expects %d parameters but received %d', ...
                        model, expected, length(pars));
            end

            % Pre-specify matrices
            r_sim = NaN(H, M);  % simulated returns
            h_sim = NaN(H, M);  % simulated variances
            
            % Iteration over GARCH model
            if strcmp(model, 'garch')
                % Initialize h_{t+1}
                h_1 = omega + alpha*(r_last-mu)^2 + beta*h_last;
                h_sim(1, :) = h_1;
                
                % Simulate M Paths of r_{t+j} and h_{t+j} for j = 1, ..., H
                for j = 1:H
                    r_sim(j,:) = mu + sqrt(h_sim(j,:)) .* z(j,:);
                    if j < H
                        eps2 = (r_sim(j,:) - mu).^2;
                        h_sim(j+1,:) = omega + alpha*eps2 + beta*h_sim(j,:);
                    end
                end

            % Iteration over GJR-GARCH model    
            elseif strcmp(model, 'gjr')
                % Initialize h_{t+1}
                eps = r_last-mu;
                I = eps < 0;
                h_1 = omega + (alpha + I*gamma)*eps^2 + beta*h_last;
                h_sim(1, :) = h_1;
                
                % Simulate M Paths of r_{t+j} and h_{t+j} for j = 1, ..., H
                for j = 1:H
                    r_sim(j,:) = mu + sqrt(h_sim(j,:)) .* z(j,:);
                    if j < H
                        eps = r_sim(j,:) - mu;
                        I = eps < 0;
                        h_sim(j+1,:) = omega + (alpha + I*gamma).*eps.^2 ...
                                       + beta*h_sim(j,:);
                    end
                end
            end

            % Cumulative returns
            R_cum = cumsum(r_sim, 1);   % row h = sum of steps 1..h
            
            % VaR and ES at each horizon h at 'conf' level
            VaR   = NaN(H, 1);
            ES    = NaN(H, 1);
            
            for h = 1:H
                sorted = sort(R_cum(h, :));        % sort ascending
                idx    = max(1, floor(conf * M));  % left tail index
                VaR(h) = sorted(idx);              % VaR(0.01)
                ES(h)  = mean(sorted(1:idx));      % ES
            end
   
        end



%==========================================================================
%       Simulation from Gaussian and Student's t-copula with constant or
%       dynamic conditional correlation matrix
%==========================================================================
        function u = simulateCopulaCCC(R, H, M, K, varargin)
        %SIMULATECOPULACCC Simulate uniform draws from a Constant
        % Conditional Correlation (CCC) copula.
        %
        %   u = SIMULATECOPULACCC(R, H, M, K) simulates from a Gaussian
        %   CCC copula with correlation matrix R, returning uniform draws
        %   in (0,1) for use in marginal inverse CDF transforms.
        %
        %   INPUTS (required):
        %       R  : (KxK) constant correlation matrix
        %       H  : Scalar, forecast horizon (number of steps ahead)
        %       M  : Scalar, number of simulation paths
        %       K  : Scalar, number of assets
        %
        %   INPUTS (optional name-value):
        %       'dist'        : String, copula distribution
        %                       'norm' - Gaussian copula (default)
        %                       't'    - Student-t copula
        %       'nu'          : Scalar, degrees of freedom
        %                       (required if dist = 't')
        %       'StdNormDraws': (KxHxM) matrix of pre-drawn standard
        %                       normal innovations. Supply externally
        %                       for reproducibility across days.
        %       'invGamma'    : (HxMx1) matrix of pre-drawn inverse
        %                       Gamma scaling factors for t-copula.
        %                       Supply externally for reproducibility.
        %
        %   OUTPUT:
        %       u  : (HxMxK) array of uniform draws in (0,1)
        %
        %   NOTES:
        %       - For the t-copula, u recovers the Gaussian copula in
        %         the limit nu -> inf
        %       - Supply StdNormDraws and inv_gamma externally to reuse
        %         draws across evaluation days within a month

            p = inputParser;
            addParameter(p, 'dist',         'norm');
            addParameter(p, 'nu',           NaN);
            addParameter(p, 'StdNormDraws', []);
            addParameter(p, 'invGamma',     []);
            parse(p, varargin{:});
        
            dist         = p.Results.dist;
            nu           = p.Results.nu;
            StdNormDraws = p.Results.StdNormDraws;
            inv_gamma    = p.Results.invGamma;


            % Draw standard normals internally if not supplied
            if isempty(StdNormDraws)
                StdNormDraws = RiskSim.drawZ(K, H, M);
            else
                if ~isequal(size(StdNormDraws), [K, H, M])
                    warning(['simulateCopulaCCC: StdNormDraws expected ' ...
                             'to be (%d x %d x %d) but got (%d x %d x %d). ' ...
                             'Results may be incorrect.'], ...
                             K, H, M, size(StdNormDraws,1), ...
                             size(StdNormDraws,2), size(StdNormDraws,3));
                end
            end

            % Draw inverse Gamma scaling factors if t-copula and not supplied
            if strcmp(dist, 't')
                if isempty(inv_gamma)
                    inv_gamma = sqrt(nu ./ (2.*randg(nu./2, H, M, 1)));
                else
                    if size(inv_gamma, 1) ~= H || size(inv_gamma, 2) ~= M
                        warning(['simulateCopulaCCC: inv_gamma expected ' ...
                                 'to be (%d x %d x 1) but got (%d x %d x %d). ' ...
                                 'Results may be incorrect.'], ...
                                 H, M, size(inv_gamma,1), ...
                                 size(inv_gamma,2), size(inv_gamma,3));
                    end
                end
            end

            % Cholesky decomposition to induce correlation across assets
            draws_2d        = reshape(StdNormDraws, K, H*M)';       
            corr_draws_2d   = draws_2d * chol(R);
            StdNormDrawsCop = permute(reshape(corr_draws_2d', K, H, M), [2,3,1]);

            % Simulate uniform draws from the specified copula
            switch dist

                % Gaussian copula
                case 'norm'
                    u = normcdf(StdNormDrawsCop);

                % Student's t-copula
                case 't'
                    if isnan(nu)
                        error(['simulateCopulaCCC: nu must be provided ' ...
                               'for dist = ''t''']);
                    end

                    StdtDrawsCop = inv_gamma .* StdNormDrawsCop; 
                    u = tcdf(StdtDrawsCop ./ sqrt((nu-2)/nu), nu);

                otherwise
                    error(['RiskSim.simulateCopulaCCC: unknown dist' ...
                           ' ''%s''. Expected ''norm'' or ''t''.'], dist); 
            end

        end



function u = simulateCopulaDCC(pars, R_bar, R_last, Q_last, H, M, K, varargin)
        %SIMULATECOPULADCC Simulate uniform draws from a Dynamic
        % Conditional Correlation (DCC) copula.
        %
        %   u = SIMULATECOPULADCC(pars, R_bar, R_last, Q_last, H, M, K)
        %   simulates from a Gaussian DCC copula, returning uniform draws
        %   in (0,1) for use in marginal inverse CDF transforms.
        %
        %   INPUTS (required):
        %       pars   : (2x1) parameter vector
        %                pars(1) - alpha (DCC innovation coefficient)
        %                pars(2) - beta  (DCC decay coefficient)
        %       R_bar  : (KxK) unconditional correlation matrix,
        %                computed as u_std'*u_std/T during estimation
        %       R_last : (KxK) last filtered dynamic correlation matrix
        %                R_T from estimation, initializes simulation paths
        %       Q_last : (KxK) last filtered Q_T from estimation,
        %                required since DCC recursion runs on Q not R
        %       H      : Scalar, forecast horizon (number of steps ahead)
        %       M      : Scalar, number of simulation paths
        %       K      : Scalar, number of assets
        %
        %   INPUTS (optional name-value):
        %       'dist'        : String, copula distribution
        %                       'norm' - Gaussian copula (default)
        %                       't'    - Student-t copula
        %       'nu'          : Scalar, degrees of freedom
        %                       (required if dist = 't')
        %       'StdNormDraws': (KxHxM) matrix of pre-drawn standard
        %                       normal innovations. Supply externally
        %                       for reproducibility across days.
        %       'invGamma'    : (HxMx1) matrix of pre-drawn inverse
        %                       Gamma scaling factors for t-copula.
        %                       Supply externally for reproducibility.
        %
        %   OUTPUT:
        %       u      : (HxMxK) array of uniform draws in (0,1)
        %
        %   NOTES:
        %       - Unlike CCC, R_t evolves per path since it depends on
        %         simulated shocks — Q and R are (KxKxM) during simulation
        %       - For the t-copula, u recovers the Gaussian copula in
        %         the limit nu -> inf
        %       - Supply StdNormDraws and invGamma externally to reuse
        %         draws across evaluation days within a month 

            p = inputParser;
            addParameter(p, 'dist',         'norm');
            addParameter(p, 'nu',           NaN);
            addParameter(p, 'StdNormDraws', []);
            addParameter(p, 'invGamma',     []);
            parse(p, varargin{:});
        
            dist         = p.Results.dist;
            nu           = p.Results.nu;
            StdNormDraws = p.Results.StdNormDraws;
            inv_gamma    = p.Results.invGamma;


            % Draw standard normals internally if not supplied
            if isempty(StdNormDraws)
                StdNormDraws = RiskSim.drawZ(K, H, M);
            else
                if ~isequal(size(StdNormDraws), [K, H, M])
                    warning(['simulateCopulaCCC: StdNormDraws expected ' ...
                             'to be (%d x %d x %d) but got (%d x %d x %d). ' ...
                             'Results may be incorrect.'], ...
                             K, H, M, size(StdNormDraws,1), ...
                             size(StdNormDraws,2), size(StdNormDraws,3));
                end
            end

            % Draw inverse Gamma scaling factors if t-copula and not supplied
            if strcmp(dist, 't')
                if isempty(inv_gamma)
                    inv_gamma = sqrt(nu ./ (2.*randg(nu./2, H, M, 1)));
                else
                    if size(inv_gamma, 1) ~= H || size(inv_gamma, 2) ~= M
                        warning(['simulateCopulaCCC: inv_gamma expected ' ...
                                 'to be (%d x %d x 1) but got (%d x %d x %d). ' ...
                                 'Results may be incorrect.'], ...
                                 H, M, size(inv_gamma,1), ...
                                 size(inv_gamma,2), size(inv_gamma,3));
                    end
                end
            end


            % Cholesky decomposition to induce correlation across assets
            a = pars(1);
            b = pars(2);
            
            % Initialize Q and R for all paths from last filtered values
            Q_sim = repmat(Q_last, 1, 1, M);   % (KxKxM)
            R_sim = repmat(R_last, 1, 1, M);   % (KxKxM)
            
            StdNormDrawsCop = NaN(H, M, K);
            
            for h = 1:H
                % Apply Cholesky to each path
                for m = 1:M
                    draws_m                = reshape(StdNormDraws(:,h,m), K, 1)';
                    StdNormDrawsCop(h,m,:) = draws_m * chol(R_sim(:,:,m));

                    % Update Q and R using current correlated draw
                    z_m          = squeeze(StdNormDrawsCop(h,m,:))';
                    Q_sim(:,:,m) = R_bar*(1-a-b) + a*(z_m'*z_m) + b*Q_sim(:,:,m);
                    d            = diag(Q_sim(:,:,m)).^(-0.5);
                    R_sim(:,:,m) = (d*d') .* Q_sim(:,:,m);
                end
            end

            % Simulate uniform draws from the specified copula
            switch dist

                % Gaussian copula
                case 'norm'
                    u = normcdf(StdNormDrawsCop);

                % Student's t-copula
                case 't'
                    if isnan(nu)
                        error(['simulateCopulaDCC: nu must be provided ' ...
                               'for dist = ''t''']);
                    end

                    StdtDrawsCop = inv_gamma .* StdNormDrawsCop; 
                    u = tcdf(StdtDrawsCop ./ sqrt((nu-2)/nu), nu);

                otherwise
                    error(['RiskSim.simulateCopulaDCC: unknown dist' ...
                           ' ''%s''. Expected ''norm'' or ''t''.'], dist);
            end

        end
        

%==========================================================================
%       Draw standardized innovations
%==========================================================================
        function z = drawZ(H, M, K, dist, nu, lambda)
        %DRAWZ Draw an (HxM) matrix of standardized innovations.
        %
        %   z = DRAWZ(H, M) draws from standard normal (default).
        %
        %   INPUTS (required):
        %       d1, d2, d3 : Dimensions of output array. Caller is 
        %                    responsible
        %                    for interpreting the output shape correctly.
        %                    For univariate simulation: 
        %                       call as drawZ(H, M, 1)
        %                    For copula simulation:     
        %                       call as drawZ(K, H, M)
        %
        %   INPUTS (optional):
        %       dist   : String, innovation distribution
        %                'norm'  - standard normal (default)
        %                't'     - standardized Student's t
        %                'skewt' - Hansen's standardized skew-t
        %       nu     : Scalar, degrees of freedom 
        %                (required if dist = 't' or 'skewt')
        %       lambda : Scalar, skewness parameter
        %               (required if dist = 'skewt')        
        %
        %   OUTPUT:
        %       z    : (HxM) matrix of standardized innovations

            if nargin < 4
                dist = 'norm';
            end
            if nargin < 5
                nu   = NaN;
            end
            if nargin < 6
                lambda = NaN;   
            end

            % % Set seed for reproducability 
            % rng(1);

            % Various distributions
            switch dist

                % Normal 
                case 'norm'
                    z = randn(H, M, K);

                % Student's t
                case 't'
                    if isnan(nu)
                        error(['ReturnSim.drawZ: nu must be provided ' ...
                               'for dist = ''t''']);
                    end
                    z = trnd(nu, H, M, K) / sqrt(nu / (nu - 2));
    
                % Hansens skew-t    
                case 'skewt'
                    if isnan(nu) || isnan(lambda)
                        error(['ReturnSim.drawZ: nu and lambda required ' ...
                               'for dist = ''skewt''']);
                    end

                    % Draw standardized t directly (fast)
                    t_draws = trnd(nu, H, M, K) / sqrt(nu / (nu - 2));
                    
                    % Constants
                    log_c = gammaln(0.5*(nu+1)) - 0.5*log(pi*(nu-2)) - ...
                            gammaln(0.5*nu);
                    c     = exp(log_c);
                    a     = 4*lambda*c*(nu-2)/(nu-1);
                    b     = sqrt(1 + 3*lambda^2 - a^2);
                    
                    % Apply piecewise skewing transformation
                    z = NaN(H, M, K);
                    left     = t_draws < 0;
                    right    = t_draws >= 0;
                    z(left)  = (1/b) .* ((1-lambda) .* t_draws(left) - a);
                    z(right) = (1/b) .* ((1+lambda) .* t_draws(right) - a);                    

                otherwise
                    error(['RiskSim.drawZ: unknown dist ''%s''. Expected' ...
                           ' ''norm'', ''t'', or ''skewt''.'], dist);
            end

        end


    end
end






