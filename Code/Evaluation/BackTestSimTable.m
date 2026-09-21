function BackTestTable = BackTestSimTable(B, varargin)

% Name-value inputs
p = inputParser;
addParameter(p, 'H_sim', 10); 
addParameter(p, 'T_sim', 5000); 
addParameter(p, 'M_sim', 25000); 
addParameter(p, 'alpha', [0.01 0.25]); 
addParameter(p, 'dist',  'norm');
addParameter(p, 'EstPars', true); 
addParameter(p, 'LaTeX', false); 
addParameter(p, 'plot',  false); 
parse(p, varargin{:});

H_sim   = p.Results.H_sim;
T_sim   = p.Results.T_sim;
M_sim   = p.Results.M_sim;
alpha   = p.Results.alpha;
dist    = p.Results.dist;
EstPars = p.Results.EstPars;
LaTeX   = p.Results.LaTeX;
plot    = p.Results.plot; 

if EstPars
    ParEstTrue = 'EstPars';
else
    ParEstTrue = 'TruePars';
end

% Load simulation results
filename = sprintf(['Output/SimBacktests/SimBacktest_%s_GARCH_%s_' ...
                   'B%d_T%d_H%d_M%d.mat'], ParEstTrue, dist, B, T_sim, ...
                    H_sim, M_sim);
SimRes = load(filename);

end