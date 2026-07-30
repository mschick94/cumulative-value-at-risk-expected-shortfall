classdef VarianceModels
    %VARIANCEMODELS A container class for variance model estimation.
    %
    %   Contains static methods for negative log-likelihood evaluation
    %   of univariate variance models. Currently implements GARCH(1,1)
    %   with normal, Student-t, and Hansen skew-t innovations.
    %
    %   Designed to be extended with additional variance models
    %   (e.g. GJR-GARCH, EGARCH) as needed.
    %
    %   Usage:
    %       NegLL = VarianceModels.univ_garch(pars, r)

    methods(Static)


%==========================================================================
%       Likelihoods of GARCH-type models
%==========================================================================

        function [NegLL, H_t, SampleMean] = univ_garch(pars, r, EstMean)
        %UNIV_GARCH Estimate a univariate GARCH(1,1) model
        %
        %   [NegLL, H_t] = UNIV_GARCH(pars, epsi) computes the negative
        %   log-likelihood and conditional variances of a univariate
        %   GARCH(1,1) process with normal innovations.
        %
        %   INPUT:
        %       pars : (3x1) parameter vector
        %              pars(1) - omega (constant)
        %              pars(2) - alpha (ARCH coefficient)
        %              pars(3) - beta  (GARCH coefficient)
        %
        %       r       : (Tx1) vector of returns
        %       EstMean : if true, estimate sample mean of r (default)
        %
        %   OUTPUT:
        %       NegLL      : Scalar, negative log-likelihood
        %       H_t        : (1xT) vector of conditional variances
        %       SampleMean : Sample mean of returns
        %
           
            % Optional estimation settings
            if nargin < 3
                EstMean = true;
            end

            % Estimate sample mean of r
            if EstMean
                SampleMean = mean(r);
                r          = r - SampleMean;
            else
                SampleMean = [];
            end

            % Initialize
            K = 1; 
            omega = pars(1);
            alpha = pars(2);
            beta  = pars(3);
            n     = size(r,1);

            % Unconditional variance initialization
            H_bar = r'*r/n;
            H_t   = NaN(n,1);
            H_t(1) = H_bar;
            
            % Recursion for conditional variances
            for i = 1:n-1
                H_t(i+1) = alpha*r(i)^2 + beta*H_t(i) + omega;
            end
            
            % Log-likelihood contributions
            ll = NaN(n,1);
            for i = 2:n
                ll(i) = - K/2 * log(2*pi) - 0.5 * log(H_t(i)) ...
                        - 0.5 * r(i)^2 / H_t(i);
            end
              
            % Negative log-likelihood
            ll(1) = [];
            NegLL = -sum(ll);
                
        end


        function [NegLL, H_t, SampleMean] = univ_gjr_garch(pars, r, ...
                                                                   EstMean)
        %UNIV_GJR_GARCH Estimate a univariate GJR-GARCH(1,1) model
        %
        %   [NegLL, H_t] = UNIV_GJR_GARCH(pars, epsi) computes the negative
        %   log-likelihood and conditional variances of a univariate
        %   GJR-GARCH(1,1) process with normal innovations.
        %
        %   INPUT:
        %       pars : (4x1) parameter vector
        %              pars(1) - omega (constant)
        %              pars(2) - alpha (ARCH coefficient)
        %              pars(3) - beta  (GARCH coefficient)
        %              pars(4) - gamma (Leverage coefficient)
        %
        %       r       : (Tx1) vector of returns
        %       EstMean : if true, estimate sample mean of r (default)
        %
        %   OUTPUT:
        %       NegLL      : Scalar, negative log-likelihood
        %       H_t        : (1xT) vector of conditional variances
        %       SampleMean : Sample mean of returns
        %
           
            % Optional estimation settings
            if nargin < 3
                EstMean = true;
            end

            % Estimate sample mean of r
            if EstMean
                SampleMean = mean(r);
                r          = r - SampleMean;
            else
                SampleMean = [];
            end

            % Initialize
            K = 1; 
            omega = pars(1);
            alpha = pars(2);
            beta  = pars(3);
            gamma = pars(4);
            n     = size(r,1);

            % Unconditional variance initialization
            H_bar = r'*r/n;
            H_t   = NaN(n,1);
            H_t(1) = H_bar;
            
            % Recursion for conditional variances
            I = r < 0;
            for i = 1:n-1
                H_t(i+1) = (alpha + I(i)*gamma)*r(i)^2 + beta*H_t(i) ...
                           + omega;
            end
            
            % Log-likelihood contributions
            ll = NaN(n,1);
            for i = 2:n
                ll(i) = - K/2 * log(2*pi) - 0.5 * log(H_t(i)) ...
                        - 0.5 * r(i)^2 / H_t(i);
            end
              
            % Negative log-likelihood
            ll(1) = [];
            NegLL = -sum(ll);
                
        end


        function [NegLL, H_t, SampleMean] = univ_garch_t(pars, r, EstMean)
        %UNIV_GARCH_T Estimate a univariate GARCH(1,1)-t model
        %
        %   [NegLL, H_t] = UNIV_GARCH_t(pars, epsi) computes the negative
        %   log-likelihood and conditional variances of a univariate
        %   GARCH(1,1) process with Student-t innovations.
        %
        %   INPUT:
        %       pars : (4x1) parameter vector
        %              pars(1) - omega (constant)
        %              pars(2) - alpha (ARCH coefficient)
        %              pars(3) - beta  (GARCH coefficient)
        %              pars(4) - nu    (degrees of freedom, Student-t)
        %
        %       r       : (Tx1) vector of returns
        %       EstMean : if true, estimate sample mean of r (default)
        %
        %   OUTPUT:
        %       NegLL      : Scalar, negative log-likelihood
        %       H_t        : (1xT) vector of conditional variances
        %       SampleMean : Sample mean of returns
        %

            % Optional estimation settings
            if nargin < 3
                EstMean = true;
            end

            % Estimate sample mean of r
            if EstMean
                SampleMean = mean(r);
                r          = r - SampleMean;
            else
                SampleMean = [];
            end

            % Initialize
            K = 1; 
            omega = pars(1);
            alpha = pars(2);
            beta  = pars(3);
            nu    = pars(4);
            n     = size(r,1);
            
            % Unconditional variance initialization
            H_bar = r'*r/n;
            H_t   = NaN(n,1);
            H_t(1) = H_bar;
            
            % Recursion for conditional variances
            for i = 1:n-1
                H_t(i+1) = alpha*r(i)^2 + beta*H_t(i) + omega;
            end
            
            % Log-likelihood contributions
            ll = NaN(n,1);
            for i = 2:n
                ll(i) = gammaln( 0.5*(nu+K) ) - gammaln( 0.5*nu ) ...
                        - (0.5*K) * log((nu-2)*pi) - 0.5 * log(H_t(i)) ...
                        - 0.5*(nu+K) * log( 1 + (r(i)^2/H_t(i)) / (nu-2) );
            end
              
            % Negative log-likelihood
            ll(1) = [];
            NegLL = -sum(ll);
                
        end


        function [NegLL, H_t, SampleMean] = univ_gjr_garch_t(pars, r, EstMean)
        %UNIV_GJR_GARCH_T Estimate a univariate GJR-GARCH(1,1)-t model
        %
        %   [NegLL, H_t] = UNIV_GJR_GARCH_t(pars, epsi) computes the 
        %   negative log-likelihood and conditional variances of a 
        %   univariate GARCH(1,1) process with Student-t innovations.
        %
        %   INPUT:
        %       pars : (4x1) parameter vector
        %              pars(1) - omega (constant)
        %              pars(2) - alpha (ARCH coefficient)
        %              pars(3) - beta  (GARCH coefficient)
        %              pars(4) - gamma (Leverage coefficient)        
        %              pars(5) - nu    (degrees of freedom, Student-t)
        %
        %       r       : (Tx1) vector of returns
        %       EstMean : if true, estimate sample mean of r (default)
        %
        %   OUTPUT:
        %       NegLL      : Scalar, negative log-likelihood
        %       H_t        : (1xT) vector of conditional variances
        %       SampleMean : Sample mean of returns
        %

            % Optional estimation settings
            if nargin < 3
                EstMean = true;
            end

            % Estimate sample mean of r
            if EstMean
                SampleMean = mean(r);
                r          = r - SampleMean;
            else
                SampleMean = [];
            end

            % Initialize
            K = 1; 
            omega = pars(1);
            alpha = pars(2);
            beta  = pars(3);
            gamma = pars(4);
            nu    = pars(5);
            n     = size(r,1);
            
            % Unconditional variance initialization
            H_bar = r'*r/n;
            H_t   = NaN(n,1);
            H_t(1) = H_bar;
            
            % Recursion for conditional variances
            I = r < 0;
            for i = 1:n-1
                H_t(i+1) = (alpha + I(i)*gamma)*r(i)^2 + beta*H_t(i) ...
                           + omega;
            end
            
            % Log-likelihood contributions
            ll = NaN(n,1);
            for i = 2:n
                ll(i) = gammaln( 0.5*(nu+K) ) - gammaln( 0.5*nu ) ...
                        - (0.5*K) * log((nu-2)*pi) - 0.5 * log(H_t(i)) ...
                        - 0.5*(nu+K) * log( 1 + (r(i)^2/H_t(i)) / (nu-2) );
            end
              
            % Negative log-likelihood
            ll(1) = [];
            NegLL = -sum(ll);
                
        end        


        function [NegLL, H_t, SampleMean] = univ_garch_skew_t(pars, r, ...
                                                                   EstMean)
        %UNIV_GARCH_SKEW_T Estimate a univariate GARCH(1,1)-skew-t model
        %
        %   [NegLL, H_t] = univ_garch_skew_t(pars, epsi) computes the 
        %   negative log-likelihood and conditional variances of a 
        %   univariate GARCH(1,1) process with Hansen skew-t innovations.
        %
        %   INPUT:
        %       pars : (5x1) parameter vector
        %              pars(1) - omega  (constant)
        %              pars(2) - alpha  (ARCH coefficient)
        %              pars(3) - beta   (GARCH coefficient)
        %              pars(4) - lambda (skewness, Hansen skew-t)
        %              pars(5) - nu     (degrees of freedom, Hansen skew-t)
        %
        %       r       : (Tx1) vector of returns
        %       EstMean : if true, estimate sample mean of r (default)
        %
        %   OUTPUT:
        %       NegLL      : Scalar, negative log-likelihood
        %       H_t        : (1xT) vector of conditional variances
        %       SampleMean : Sample mean of returns
        %

             % Optional estimation settings
            if nargin < 3
                EstMean = true;
            end

            % Estimate sample mean of r
            if EstMean
                SampleMean = mean(r);
                r          = r - SampleMean;
            else
                SampleMean = [];
            end       

            % Initialize
            omega  = pars(1);
            alpha  = pars(2);
            beta   = pars(3);
            lambda = pars(4);
            nu     = pars(5);
            n      = size(r,1);
            
            % Unconditional variance initialization
            H_bar = r'*r/n;
            H_t   = NaN(n,1);
            H_t(1) = H_bar;
            
            % Recursion for conditional variances
            for i = 1:n-1
                H_t(i+1) = alpha*r(i)^2 + beta*H_t(i) + omega;
            end
            
            % Hansen skew-t
            log_c = gammaln(0.5*(nu+1)) - 0.5*log(pi*(nu-2)) ...
                    - gammaln(0.5*nu);
            c = exp(log_c);

            a = 4*lambda*c * (nu-2) / (nu-1);
            b = sqrt(1 + 3*lambda^2 - a^2);
            log_b = log(b);

            
            % Threshold
            x_star = -a / b;

            % Log-likelihood contributions
            ll = NaN(n,1);
            for i = 2:n
                
                sigma_t = sqrt(H_t(i));
                z = r(i) / sigma_t;
                
                if z < x_star
                    ll(i) = log_b + log_c ...
                        - 0.5*(nu+1) * log(1 + (1/(nu-2)) * ( (b*z + a)/(1-lambda) )^2) ...
                        - log(sigma_t);
                else
                    ll(i) = log_b + log_c ...
                        - 0.5*(nu+1) * log(1 + (1/(nu-2)) * ( (b*z + a)/(1+lambda) )^2) ...
                        - log(sigma_t);
                end
            
            end
              
            % Negative log-likelihood
            ll(1) = [];
            NegLL = -sum(ll);
                
        end


        function [NegLL, H_t, SampleMean] = univ_gjr_garch_skew_t(pars, ...
                                                                r, EstMean)
        %UNIV_GJR_GARCH_SKEW_T Estimate a univariate GJR-GARCH(1,1)-skew-t 
        % model
        %
        %   [NegLL, H_t] = univ_gjr_garch_skew_t(pars, epsi) computes the 
        %   negative log-likelihood and conditional variances of a 
        %   univariate GJR-GARCH(1,1) process with Hansen skew-t 
        %   innovations.
        %
        %   INPUT:
        %       pars : (5x1) parameter vector
        %              pars(1) - omega  (constant)
        %              pars(2) - alpha  (ARCH coefficient)
        %              pars(3) - beta   (GARCH coefficient)
        %              pars(4) - gamma  (Leverage coefficient)
        %              pars(5) - lambda (skewness, Hansen skew-t)
        %              pars(6) - nu     (degrees of freedom, Hansen skew-t)
        %
        %       r       : (Tx1) vector of returns
        %       EstMean : if true, estimate sample mean of r (default)
        %
        %   OUTPUT:
        %       NegLL      : Scalar, negative log-likelihood
        %       H_t        : (1xT) vector of conditional variances
        %       SampleMean : Sample mean of returns
        %

             % Optional estimation settings
            if nargin < 3
                EstMean = true;
            end

            % Estimate sample mean of r
            if EstMean
                SampleMean = mean(r);
                r          = r - SampleMean;
            else
                SampleMean = [];
            end       

            % Initialize
            omega  = pars(1);
            alpha  = pars(2);
            beta   = pars(3);
            gamma  = pars(4);
            lambda = pars(5);
            nu     = pars(6);
            n      = size(r,1);
            
            % Unconditional variance initialization
            H_bar = r'*r/n;
            H_t   = NaN(n,1);
            H_t(1) = H_bar;
            
            % Recursion for conditional variances
            I = r < 0;
            for i = 1:n-1
                H_t(i+1) = (alpha + I(i)*gamma)*r(i)^2 + beta*H_t(i) ...
                           + omega;
            end
            
            % Hansen skew-t
            log_c = gammaln(0.5*(nu+1)) - 0.5*log(pi*(nu-2)) ...
                    - gammaln(0.5*nu);
            c = exp(log_c);

            a = 4*lambda*c * (nu-2) / (nu-1);
            b = sqrt(1 + 3*lambda^2 - a^2);
            log_b = log(b);

            
            % Threshold
            x_star = -a / b;

            % Log-likelihood contributions
            ll = NaN(n,1);
            for i = 2:n
                
                sigma_t = sqrt(H_t(i));
                z = r(i) / sigma_t;
                
                if z < x_star
                    ll(i) = log_b + log_c ...
                        - 0.5*(nu+1) * log(1 + (1/(nu-2)) * ( (b*z + a)/(1-lambda) )^2) ...
                        - log(sigma_t);
                else
                    ll(i) = log_b + log_c ...
                        - 0.5*(nu+1) * log(1 + (1/(nu-2)) * ( (b*z + a)/(1+lambda) )^2) ...
                        - log(sigma_t);
                end
            
            end
              
            % Negative log-likelihood
            ll(1) = [];
            NegLL = -sum(ll);
                
        end


        function [NegLL, H_t, SampleMean] = univ_garch_laplace(pars, r, ...
                                                                   EstMean)
        %UNIV_GARCH_laplace Estimate a univariate GARCH(1,1) model with
        % Laplace innovations.
        %
        %   [NegLL, H_t] = UNIV_GARCH_LAPLACE(pars, epsi) computes the 
        %   negative log-likelihood and conditional variances of a 
        %   univariate GARCH(1,1) process with Laplace innovations.
        %
        %   INPUT:
        %       pars : (3x1) parameter vector
        %              pars(1) - omega (constant)
        %              pars(2) - alpha (ARCH coefficient)
        %              pars(3) - beta  (GARCH coefficient)
        %
        %       r       : (Tx1) vector of returns
        %       EstMean : if true, estimate sample mean of r (default)
        %
        %   OUTPUT:
        %       NegLL      : Scalar, negative log-likelihood
        %       H_t        : (1xT) vector of conditional variances
        %       SampleMean : Sample mean of returns
        %
           
            % Optional estimation settings
            if nargin < 3
                EstMean = true;
            end

            % Estimate sample mean of r
            if EstMean
                SampleMean = mean(r);
                r          = r - SampleMean;
            else
                SampleMean = [];
            end

            % Initialize
            K = 1; 
            omega = pars(1);
            alpha = pars(2);
            beta  = pars(3);
            n     = size(r,1);

            % Unconditional variance initialization
            H_bar = r'*r/n;
            H_t   = NaN(n,1);
            H_t(1) = H_bar;
            
            % Recursion for conditional variances
            for i = 1:n-1
                H_t(i+1) = alpha*r(i)^2 + beta*H_t(i) + omega;
            end
            
            % Log-likelihood contributions
            ll = NaN(n,1);
            for i = 2:n
                ll(i) = - K/2 * log(2) - sqrt(2)*abs(r(i))/sqrt(H_t(i)) ...
                        - 0.5 * log(H_t(i));
            end
              
            % Negative log-likelihood
            ll(1) = [];
            NegLL = -sum(ll);
                
        end


        function [NegLL, H_t, SampleMean] = univ_gjr_garch_laplace(pars, ...
                                                                r, EstMean)
        %UNIV_GJR_GARCH_LAPLACE Estimate a univariate GJR-GARCH(1,1) model
        % with Laplace innovations
        %
        %   [NegLL, H_t] = UNIV_GJR_GARCH_LAPLACE(pars, epsi) computes the 
        %   negative log-likelihood and conditional variances of a 
        %   univariate GJR-GARCH(1,1) process with normal innovations.
        %
        %   INPUT:
        %       pars : (4x1) parameter vector
        %              pars(1) - omega (constant)
        %              pars(2) - alpha (ARCH coefficient)
        %              pars(3) - beta  (GARCH coefficient)
        %              pars(4) - gamma (Leverage coefficient)
        %
        %       r       : (Tx1) vector of returns
        %       EstMean : if true, estimate sample mean of r (default)
        %
        %   OUTPUT:
        %       NegLL      : Scalar, negative log-likelihood
        %       H_t        : (1xT) vector of conditional variances
        %       SampleMean : Sample mean of returns
        %
           
            % Optional estimation settings
            if nargin < 3
                EstMean = true;
            end

            % Estimate sample mean of r
            if EstMean
                SampleMean = mean(r);
                r          = r - SampleMean;
            else
                SampleMean = [];
            end

            % Initialize
            K = 1; 
            omega = pars(1);
            alpha = pars(2);
            beta  = pars(3);
            gamma = pars(4);
            n     = size(r,1);

            % Unconditional variance initialization
            H_bar = r'*r/n;
            H_t   = NaN(n,1);
            H_t(1) = H_bar;
            
            % Recursion for conditional variances
            I = r < 0;
            for i = 1:n-1
                H_t(i+1) = (alpha + I(i)*gamma)*r(i)^2 + beta*H_t(i) ...
                           + omega;
            end
            
            % Log-likelihood contributions
            ll = NaN(n,1);
            for i = 2:n
                ll(i) = - K/2 * log(2) - sqrt(2)*abs(r(i))/sqrt(H_t(i)) ...
                        - 0.5 * log(H_t(i));
            end
              
            % Negative log-likelihood
            ll(1) = [];
            NegLL = -sum(ll);
                
        end


    end
end