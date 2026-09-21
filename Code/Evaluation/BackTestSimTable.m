function BackTestTable = BackTestSimTable(B, varargin)

% Name-value inputs
p = inputParser;
addParameter(p, 'H_sim',      10); 
addParameter(p, 'T_sim',      5000); 
addParameter(p, 'M_sim',      25000); 
addParameter(p, 'alpha',      [0.01 0.025]); 
addParameter(p, 'SigLevel',   0.05); 
addParameter(p, 'dist',       'norm');
addParameter(p, 'EstPars',    true); 
addParameter(p, 'Decimals',   3); 
addParameter(p, 'LaTeX',      true); 
addParameter(p, 'PrintTable', true); 
addParameter(p, 'PlotFig',    false); 
parse(p, varargin{:});

H_sim      = p.Results.H_sim;
T_sim      = p.Results.T_sim;
M_sim      = p.Results.M_sim;
alpha      = p.Results.alpha;
SigLevel   = p.Results.SigLevel;
dist       = p.Results.dist;
EstPars    = p.Results.EstPars;
Decimals   = p.Results.Decimals;
LaTeX      = p.Results.LaTeX;
PrintTable = p.Results.PrintTable;
PlotFig    = p.Results.PlotFig; 

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

% Average Hit rate
MeanHitRate = mean(SimRes.SimBacktest.HitRate);


% One-sided and two sided tests at significance level 'SigLevel'
alphaNo = length(alpha);
OneSided = NaN(3,alphaNo);
TwoSided = NaN(3,alphaNo);

% Critical values
cv_one = norminv(1 - SigLevel);       % e.g. 1.645 for 5%
cv_two = norminv(1 - SigLevel/2);     % e.g. 1.96 for 5%

% UC-HAC test
OneSided(1,:) = mean(SimRes.SimBacktest.UC_t_HAC > cv_one, 1);
TwoSided(1,:) = mean(abs(SimRes.SimBacktest.UC_t_HAC) > cv_two, 1);

% ES-HAC test
OneSided(2,:) = mean(SimRes.SimBacktest.ES_t_HAC > cv_one, 1);
TwoSided(2,:) = mean(abs(SimRes.SimBacktest.ES_t_HAC) > cv_two, 1);

% ES-bernoulli test
OneSided(3,:) = mean(SimRes.SimBacktest.ES_t_bern > cv_one, 1);
TwoSided(3,:) = mean(abs(SimRes.SimBacktest.ES_t_bern) > cv_two, 1);

% Pack output structure
BackTestTable.MeanHitRate = MeanHitRate;
BackTestTable.OneSided    = OneSided;
BackTestTable.TwoSided    = TwoSided;


% Display results table
if PrintTable
    fprintf('\nGARCH-%s: %s\n', upper(dist), ParEstTrue);
    fprintf('%-15s', '');
    for i = 1:alphaNo
        fprintf('  alpha=%.3f         ', alpha(i));
    end
    fprintf('\n');
    fprintf('%-15s', '');
    for i = 1:alphaNo
        fprintf('  One-sided Two-sided ');
    end
    fprintf('\n%s\n', repmat('-', 1, 15 + alphaNo*22));
    
    test_names = {'UC-HAC', 'ES-HAC', 'ES-Bern'};
    for j = 1:3
        fprintf('%-15s', test_names{j});
        for i = 1:alphaNo
            fprintf('  %7.3f   %7.3f   ', OneSided(j,i), TwoSided(j,i));
        end
        fprintf('\n');
    end
    fprintf('%s\n', repmat('-', 1, 15 + alphaNo*22));
    fprintf('%-15s', 'Mean HitRate');
    for i = 1:alphaNo
        fprintf('  %7.3f             ', MeanHitRate(i));
    end
    fprintf('\n');
end

% Plot simulated test distributions
if PlotFig
    test_names = {'UC-HAC', 'ES-HAC', 'ES-Bernoulli'};
    t_fields   = {'UC_t_HAC', 'ES_t_HAC', 'ES_t_bern'};
    x_range    = linspace(-5, 5, 1000);
    
    for i = 1:length(alpha)
        for j = 1:3
            figure
            histogram(SimRes.SimBacktest.(t_fields{j})(:,i), ...
                      'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7])
            hold on
            plot(x_range, normpdf(x_range), 'r-', 'LineWidth', 2)
            xline(0, 'k--', 'LineWidth', 1.5)
            xlabel('t-statistic')
            ylabel('Density')
            title(sprintf('%d%% %s t-statistics vs Standard Normal', ...
                          alpha(i)*100, test_names{j}))
            legend('Simulated', 'N(0,1)', 'Location', 'northwest')
        end
    end
end


% Construct main body of the Table in LaTeX format
if LaTeX
    format_str = sprintf('%%.%df', Decimals);
    
    % Build string table structure
    n_cols    = 2 * alphaNo * 2 + 2;  
    table_str = strings(3, n_cols);
    table_str(1,1) = "UC-HAC";
    table_str(2,1) = "ES-HAC";
    table_str(3,1) = "ES-Bern";
    table_str(:,2)   = '&';
    table_str(:,end) = '\\';
    
    col = 3;
    for i = 1:alphaNo
        for j = 1:3
            table_str(j, col)   = sprintf(format_str, OneSided(j,i));
            table_str(j, col+1) = '&';
            table_str(j, col+2) = sprintf(format_str, TwoSided(j,i));
            if i < alphaNo
                table_str(j, col+3) = '&';
            end
        end
        col = col + 4;
    end
    
    % Convert to categorical table
    Tab = array2table(table_str);
    for i = 1:n_cols
        Tab.(i) = categorical(Tab.(i));
    end
    BackTestTable.LaTeX = Tab;
end


% % Construct main body of the Table in LaTeX format
% if LaTeX
%     latex_str = '';
%     test_names_latex = {'UC-HAC', 'ES-HAC', 'ES-Bern'};
%     for j = 1:3
%         row = sprintf('%s', test_names_latex{j});
%         for i = 1:alphaNo
%             row = sprintf('%s & %.3f & %.3f', row, OneSided(j,i), TwoSided(j,i));
%         end
%         row = sprintf('%s \\\\\\\\', row);
%         latex_str = sprintf('%s%s\n', latex_str, row);
%     end
%     % Mean hit rate row
%     row = sprintf('Mean Hit Rate');
%     for i = 1:alphaNo
%         row = sprintf('%s & %.3f & ', row, MeanHitRate(i));
%     end
%     row = sprintf('%s \\\\\\\\', row);
%     latex_str = sprintf('%s%s\n', latex_str, row);
% 
%     BackTestTable.LaTeX = latex_str;
% end


end