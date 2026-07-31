classdef CopulaModel
    %CopulaClass A container class for computing probability integral
    % transforms, copula likelihoods, and simulating from copulas.

    methods(Static)


%==========================================================================
%       Copula likelihoods
%========================================================================== 

    function [NegLL, loglike_vec, R] = CopulaCCC(pars, u, varargin)
            %COPULACCC Evaluate the log-likelihood of a Constant 
            % Conditional Correlation (CCC) copula model.
            %
            %   [NegLL, loglike_vec, R] = COPULACCC(pars, u) evaluates the
            %   Gaussian CCC copula log-likelihood. The correlation matrix 
            %   R is estimated directly from u via sample correlation.
            %
            %   [NegLL, loglike_vec, R] = COPULACCC(pars, u, 'dist', 't')
            %   evaluates the Student-t CCC copula log-likelihood with 
            %   degrees of freedom nu = pars.
            %
            %   INPUTS (required):
            %       pars   : Scalar, degrees of freedom nu for the 
            %                t-copula. Pass [] for the Gaussian copula (not 
            %                used).
            %       u      : (TxK) matrix of probability integral 
            %                transforms (PITs) in (0,1), where T is the 
            %                number of observations and K is the number of 
            %                assets
            %
            %   INPUTS (optional name-value):
            %       'dist' : String, copula distribution. Options:
            %                'norm' - Gaussian copula (default)
            %                't'    - Student-t copula
            %
            %   OUTPUTS:
            %       NegLL       : Scalar, negative log-likelihood
            %       loglike_vec : (Tx1) vector of log-likelihood 
            %                     contributions
            %       R           : (KxK) constant correlation matrix, 
            %                     estimated via sample correlation of 
            %                     transformed u
            %
            %   NOTES:
            %       - For the t-copula, nu is capped at 1e8 for numerical
            %         stability. Beyond this threshold the t-copula is
            %         numerically indistinguishable from the Gaussian 
            %         copula.
            %       - For use with fmincon, pass as:
            %         fmincon(@(pars) CopulaCCC(pars, u, 'dist', 't'), ...)
            %

            % Read out dimensions of u
            [T,K] = size(u);

            % Name-value inputs
            p = inputParser;
            addParameter(p, 'dist', 'norm');
            parse(p, varargin{:});
        
            dist = p.Results.dist;
            nu   = pars;
 
            % Log-likelihood contributions
            loglike_vec = NaN(T,1);

            % Negative Copula log-likelihoods
            switch dist

                % Gaussian copula
                case 'norm'

                    % Standardized residuals
                    u_std = norminv(u);

                    % Constant correlation matrix
                    R = corr(u_std);

                    % likelihood contributions
                    log_det_R = log(det(R)); 
                    for t = 1:T
                        u_R_inv_u        = u_std(t,:) * (R \ u_std(t,:)');
                        loglike_vec(t,1) = - 0.5 * u_R_inv_u ...
                                           - 0.5*log_det_R;
                    end
                    loglike_vec = loglike_vec + 0.5*sum(u_std.^2,2);  

                % Student's t-copula
                case 't'
                    if isnan(nu)
                        error(['ReturnSim.drawZ: nu must be provided ' ...
                               'for dist = ''t''']);
                    end
                    
                    % For numerical stability
                    nu = min(nu, 1e8);

                    % Standardized residuals
                    u_std = sqrt((nu-2)/nu) * tinv(u,nu);

                    % Constant correlation matrix
                    R = corr(u_std);

                    % likelihood contributions
                    log_det_R = log(det(R)); 
                    for t = 1:T
                        u_R_inv_u        = u_std(t,:) * (R \ u_std(t,:)');
                        loglike_vec(t,1) = - 0.5 * (nu+K) ...
                                           * log1p(u_R_inv_u/(nu-2)) ...
                                           - 0.5* log_det_R;
                    end
                    loglike_vec = loglike_vec + gammaln(0.5*(nu+K)) ...
                                  + (K-1) * gammaln(nu/2) ...
                                  - K * gammaln(0.5*(nu+1)) ...
                                  + 0.5 * (nu+1) ...
                                  * sum(log1p((1/(nu-2)) * u_std.^2), 2);

                otherwise
                    error('ReturnSim.drawZ: unknown dist ''%s''', dist);
            end

            % Negative log-likelihood
            loglike_vec(1) = []; % for consistency with GARCH and DCC model
            NegLL = -sum(loglike_vec);

    end


    function [NegLL, loglike_vec, R_last, Q_last, R_bar] = ...
            CopulaDCC(pars, u, varargin)
            %COPULADCC Evaluate the log-likelihood of a Dynamic Conditional 
            % Correlation (DCC) copula model.
            %
            %   [NegLL, loglike_vec, R_t] = COPULADCC(pars, u) evaluates 
            %   the Gaussian DCC copula log-likelihood. The correlation  
            %   matrix R_t is estimated directly from u via sample 
            %   correlation.
            %
            %   [NegLL, loglike_vec, R_t] = COPULADCC(pars, u, 'dist', 't')
            %   evaluates the Student-t DCC copula log-likelihood with 
            %   degrees of freedom nu = pars.
            %
            %   INPUTS (required):
            %       pars : (3x1) parameter vector
            %              pars(1) - alpha (DCC coefficient)
            %              pars(2) - beta  (DCC coefficient)
            %              pars(3) - degrees of freedom nu for the t-copula
            %                        (optional if t-copula)         
            %       u      : (TxK) matrix of probability integral 
            %                transforms (PITs) in (0,1), where T is the 
            %                number of observations and K is the number of 
            %                assets
            %
            %   INPUTS (optional name-value):
            %       'dist' : String, copula distribution. Options:
            %                'norm' - Gaussian copula (default)
            %                't'    - Student-t copula
            %
            %   OUTPUTS:
            %       NegLL       : Scalar, negative log-likelihood
            %       loglike_vec : (Tx1) vector of log-likelihood 
            %                     contributions
            %       R_t         : (KxKxT) dynamic correlation matrix, 
            %                     estimated via sample correlation of 
            %                     transformed u
            %
            %   NOTES:
            %       - For the t-copula, nu is capped at 1e8 for numerical
            %         stability. Beyond this threshold the t-copula is
            %         numerically indistinguishable from the Gaussian 
            %         copula.
            %       - For use with fmincon, pass as:
            %         fmincon(@(pars) CopulaDCC(pars, u, 'dist', 't'), ...)
            %

            % Read out dimensions of u
            [T,K] = size(u);

            % Name-value inputs
            p = inputParser;
            addParameter(p, 'dist', 'norm');
            parse(p, varargin{:});
        
            dist = p.Results.dist;
 
            % Log-likelihood contributions
            loglike_vec = NaN(T,1);

            % Negative Copula log-likelihoods
            switch dist

                % Gaussian copula
                case 'norm'

                    % Read out paramters
                    alpha = pars(1);
                    beta  = pars(2);

                    % Standardized residuals
                    u_std = norminv(u);

                    % Iterate on Q_t
                    R_bar = u_std'*u_std/T;
                    Q_old = R_bar;

                    % DCC iteration
                    for t = 2:T
                        % Update Q and R using lag of u_std
                        Q_old = R_bar*(1-alpha-beta) ...
                                + alpha*(u_std(t-1,:)' * u_std(t-1,:)) ...
                                + beta*Q_old;
                        d     = diag(Q_old).^(-0.5);
                        R_old = (d*d') .* Q_old;
                    
                        % likelihood contributions at t using R_t given t-1
                        log_det_R = log(det(R_old));
                        u_R_inv_u = u_std(t,:) * (R_old\u_std(t,:)');
                        loglike_vec(t,1) = -0.5*u_R_inv_u - 0.5*log_det_R;
                    end                    
                    loglike_vec = loglike_vec + 0.5*sum(u_std.^2,2);


                % Student's t-copula
                case 't'
                    if max(size(pars)) < 3
                        error(['CopulaModel.CopulaDCC: pars must be a ' ...
                               '3-dimensional vector for alpha, beta, and nu']);
                    end

                    % Read out paramters
                    alpha = pars(1);
                    beta  = pars(2);

                    % For numerical stability
                    nu = pars(3);
                    nu = min(nu, 1e8);

                    % Standardized residuals
                    u_std = sqrt((nu-2)/nu) * tinv(u,nu);

                    % Iterate on Q_t
                    R_bar = u_std'*u_std/T;
                    Q_old = R_bar;

                    % DCC iteration
                    for t = 2:T
                        % Update Q and R using lag of u_std 
                        Q_old = R_bar*(1-alpha-beta) ...
                                + alpha*(u_std(t-1,:)' * u_std(t-1,:)) ...
                                + beta*Q_old;
                        d     = diag(Q_old).^(-0.5);
                        R_old = (d*d') .* Q_old;
                    
                        % likelihood contributions at t using R_t given t-1
                        log_det_R = log(det(R_old));
                        u_R_inv_u = u_std(t,:) * (R_old\u_std(t,:)');
                        loglike_vec(t,1) = - 0.5 * (nu+K) ...
                                           * log1p(u_R_inv_u/(nu-2)) ...
                                           - 0.5* log_det_R;
                    end                    
                    loglike_vec = loglike_vec + gammaln(0.5*(nu+K)) ...
                                  + (K-1) * gammaln(nu/2) ...
                                  - K * gammaln(0.5*(nu+1)) ...
                                  + 0.5 * (nu+1) * sum(log1p((1/(nu-2)) ...
                                  * u_std.^2), 2);

                otherwise
                    error('CopulaModel.CopulaDCC: unknown dist ''%s''', ...
                        dist);
            end

            % Negative log-likelihood
            loglike_vec(1) = [];
            NegLL = -sum(loglike_vec);

            % Outout for simulation from the copula with DCC
            R_last = R_old;
            Q_last = Q_old;

    end



%==========================================================================
%       Probability intergral transforms
%==========================================================================

        function u = pits(z, varargin)
        %PITS Compute probability integral transforms (PITs) of 
        % standardized residuals under a specified innovation distribution.
        %
        %   u = PITS(z) computes PITs under the standard normal 
        %   distribution.
        %
        %   u = PITS(z, 'dist', dist, 'nu', nu, 'lambda', lambda) computes
        %   PITs under the specified distribution with parameters nu and 
        %   lambda.
        %
        %   INPUTS (required):
        %       z      : (TxK) matrix of standardized residuals, where T is
        %                the number of observations and K is the number of
        %                assets
        %
        %   INPUTS (optional name-value):
        %       'dist'   : String, innovation distribution. Options:
        %                  'norm'      - standard normal (default)
        %                  't'         - standardized Student-t (zero mean,
        %                                unit variance)
        %                  'skewt'     - Hansen skew-t
        %                  'laplace'   - Laplace distribution        
        %                  'empirical' - empirical distribution
        %       'nu'     : (1xK) vector of degrees of freedom, one per 
        %                  asset (required if dist = 't' or 'skewt')
        %       'lambda' : (1xK) vector of skewness parameters, one per 
        %                  asset (required if dist = 'skewt')
        %
        %   OUTPUT:
        %       u      : (TxK) matrix of PITs in (0,1), clipped to
        %                [1e-12, 1-1e-12] for numerical safety

            % Read out dimensions of z 
            [T,K] = size(z);

            % Name-value inputs
            p = inputParser;
            addParameter(p, 'dist',   'norm');
            addParameter(p, 'nu',     NaN);
            addParameter(p, 'lambda', NaN);
            parse(p, varargin{:});
        
            dist   = p.Results.dist;
            nu     = p.Results.nu;
            lambda = p.Results.lambda;   

            % Various distributions
            switch dist

                % Normal 
                case 'norm'
                    u = normcdf(z);    

                % Student's t
                case 't'
                    if isnan(nu)
                        error(['ReturnSim.drawZ: nu must be provided ' ...
                               'for dist = ''t''']);
                    end
                    
                    u = NaN(T,K);
                    for k = 1:K
                        z_std  = sqrt(nu(k) / (nu(k)-2)) * z(:,k);
                        u(:,k) = tcdf(z_std,nu(k));
                    end

                % Hansens skew-t 
                case 'skewt'
                    if any(isnan(nu)) || any(isnan(lambda))
                        error(['ReturnSim.drawZ: nu and lambda required ' ...
                               'for dist = ''skewt''']);
                    end
                    if numel(nu) ~= K || numel(lambda) ~= K
                        error(['ReturnSim.pits: nu and lambda must be ' ...
                               'vectors of length K']);
                    end
                    u = NaN(T,K);
                    for k = 1:K
                        nuk     = nu(k);
                        lambdak = lambda(k);

                        % Constants
                        log_c = gammaln(0.5*(nuk+1)) ...
                                - 0.5*log(pi*(nuk-2)) - gammaln(0.5*nuk);
                        c     = exp(log_c);
                        a     = 4*lambdak*c*(nuk-2)/(nuk-1);
                        b     = sqrt(1 + 3*lambdak^2 - a^2);

                        % Threshold
                        x_star = -a / b;

                        % Apply piecewise skewing transformation
                        left  = z(:,k) < x_star;
                        right = z(:,k) >= x_star;

                        % Left side
                        arg_left  = (b*z(left,k) + a)/(1 - lambdak);
                        u(left,k) = (1 - lambdak) * ...
                                    tcdf(arg_left * sqrt(nuk/(nuk-2)) , nuk);

                        % Right side
                        arg_right  = (b*z(right,k) + a)/(1 + lambdak);
                        u(right,k) = (1 - lambdak)/2 + (1 + lambdak) * ...
                                     (tcdf(arg_right * sqrt(nuk/(nuk-2)) , nuk) - 0.5);
                    end

                % Laplace
                case 'laplace'
                    left     = z < 0;
                    u        = NaN(T,K);
                    u(left)  = 0.5 * exp(sqrt(2) * z(left));
                    u(~left) = 1 - 0.5 * exp(-sqrt(2) * z(~left));

                % Empirical
                case 'empirical'
                    u = tiedrank(z) / (T + 1);

                otherwise
                    error(['RiskSim.drawZ: unknown dist ''%s''. ' ...
                           'Expected ''norm'', ''t'', ''skewt'',' ...
                           ' ''laplace'', or ''empirical''.'], dist);
            end

            % Numerical safety
            u = max(min(u,1-1e-12),1e-12);

        end

 

%==========================================================================
%       Transform uniform draws to marginals
%==========================================================================         
        function z = quantileTransform(u,varargin)
        %QUANTILETRANSFORM Map simulated uniform draws to standardized
        % innovations via the inverse marginal CDF (quantile transform).
        %
        %   z = QUANTILETRANSFORM(u) transforms uniform draws to standard
        %   normal innovations (default).
        %
        %   z = QUANTILETRANSFORM(u, 'dist', dist, 'nu', nu, 'lambda',
        %   lambda) transforms using the specified distribution.
        %
        %   This function is the inverse of PITS — where PITS maps
        %   standardized residuals to uniform marginals, QUANTILETRANSFORM
        %   maps simulated uniforms back to the residual space.
        %
        %   INPUTS (required):
        %       u      : (HxMxK) array of simulated uniform draws in
        %                (0,1), where H is the forecast horizon, M is
        %                the number of simulation paths, and K is the
        %                number of assets
        %
        %   INPUTS (optional name-value):
        %       'dist'    : String, marginal distribution. Options:
        %                   'norm'      - standard normal (default)
        %                   't'         - standardized Student-t
        %                   'skewt'     - Hansen skew-t
        %                   'laplace'   - Laplace distribution        
        %                   'empirical' - empirical inverse CDF via
        %                                 order statistics (bootstrap)
        %       'nu'      : (1xK) vector of degrees of freedom, one per
        %                   asset (required if dist = 't' or 'skewt')
        %       'lambda'  : (1xK) vector of skewness parameters, one per
        %                   asset (required if dist = 'skewt')
        %       'std_res' : (TxK) matrix of historical standardized
        %                   residuals (required if dist = 'empirical').
        %                   Simulated uniforms are mapped to residuals
        %                   via order statistics of std_res.
        %
        %   OUTPUT:
        %       z      : (HxMxK) array of standardized innovations,
        %                ready to be passed to hStepSimGarch per asset
        %
        %   NOTES:
        %       - For dist = 'empirical', the inverse CDF is a lookup
        %         table into sorted historical residuals — equivalent
        %         to a copula-based bootstrap of the marginals
        %       - Output z(:,:,k) is a (HxM) matrix per asset k,
        %         consistent with the input format of hStepSimGarch        

            % Name-value inputs
            p = inputParser;
            addParameter(p, 'dist',    'norm');
            addParameter(p, 'nu',      NaN);
            addParameter(p, 'lambda',  NaN);
            addParameter(p, 'std_res', []);
            parse(p, varargin{:});
        
            dist    = p.Results.dist;
            nu      = p.Results.nu;
            lambda  = p.Results.lambda; 
            std_res = p.Results.std_res;

            % Various distributions
            switch dist 

                % Normal 
                case 'norm'
                    % Normal marginals
                    z = norminv(u); 

                % Student's t
                case 't'
                    if isnan(nu)
                        error(['ReturnSim.drawZ: nu must be provided ' ...
                               'for dist = ''t''']);
                    end

                    % Student's t marginals
                    [H,M,K] = size(u);
                    z = NaN(H,M,K);
                    for k = 1:K
                        z(:,:,k) = sqrt((nu(k)-2) / nu(k)) * ...
                                                     tinv(u(:,:,k),nu(k));
                    end
                    
                % Hansen's skew-t marginals
                case 'skewt'
                    [H,M,K] = size(u);
                    if any(isnan(nu)) || any(isnan(lambda))
                        error(['ReturnSim.drawZ: nu and lambda ' ...
                               'required for dist = ''skewt''']);
                    end
                    if numel(nu) ~= K || numel(lambda) ~= K
                        error(['ReturnSim.pits: nu and lambda must ' ...
                               'be vectors of length K']);
                    end

                    z = NaN(H,M,K);
                    for k = 1:K
                        % Constants
                        log_c = gammaln(0.5*(nu(k)+1)) ...
                                - 0.5*log(pi*(nu(k)-2)) ...
                                - gammaln(0.5*nu(k));
                        c = exp(log_c);
                    
                        a = 4*lambda(k)*c * (nu(k)-2) / (nu(k)-1);
                        b = sqrt(1 + 3*lambda(k)^2 - a^2);
                           
                        % Threshold
                        u_aux  = u(:,:,k);
                        u_star = (1 - lambda(k)) * 0.5;
                        left   = u_aux < u_star;
                    
                        % Invert the piecewise Hansen skew-t CDF: map 
                        % uniform draws to standard t quantiles
                        t_q        = NaN(H,M);
                        t_q(left)  = tinv(u_aux(left) ...
                                     / (1 - lambda(k)), nu(k));
                        t_q(~left) = tinv(0.5 + (u_aux(~left)  ...
                                               - (1 - lambda(k))*0.5) ... 
                                              ./ (1 + lambda(k)), nu(k));
                    
                        % Standardize t quantiles to unit variance and 
                        % apply Hansen location-scale transformation to 
                        % recover skew-t innovations
                        t_scale        = NaN(H,M);
                        scale          = sqrt((nu(k)-2) ./ nu(k));
                        t_scale(left)  = ( (1-lambda(k)) * t_q(left) ...
                                                         * scale - a) / b;
                        t_scale(~left) = ( (1+lambda(k)) * t_q(~left) ...
                                                         * scale - a) / b;
                    
                        z(:,:,k) = t_scale;
                    
                    end

                % Laplace    
                case 'laplace'
                    [H,M,K]   = size(u);
                    left      = u < 0.5;
                    z         = NaN(H,M,K);
                    z(left)   = log(2*u(left))   / sqrt(2);
                    z(~left)  = -log(2*(1-u(~left))) / sqrt(2);

                % Empirical   
                case 'empirical'
                    if isempty(std_res)
                        error(['CopulaModel.quantileTransform: ' ...
                               'std_res must be provided for dist = ' ...
                               '''empirical''']);
                    end

                    % Dimensions
                    T       = size(std_res,1);
                    [H,M,K] = size(u);
                    
                    z = NaN(H,M,K);
                    for k = 1:K
                        z_sorted = sort(std_res(:,k));
                        idx      = max(1, round(u(:,:,k) * T));
                        z(:,:,k) = z_sorted(idx);
                    end

                otherwise
                    error(['RiskSim.drawZ: unknown dist ''%s''. ' ...
                           'Expected ''norm'', ''t'', ''skewt'',' ...
                           ' ''laplace'', or ''empirical''.'], dist);
            end

        end


    end
end

