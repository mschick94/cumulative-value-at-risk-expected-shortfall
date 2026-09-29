function VaRandES = merge_mv_models(VaRandES)


% Read out estimation set-up
K       = length(VaRandES.assets);
WinL    = VaRandES.WindLength;
H       = VaRandES.H;
T       = length(VaRandES.dates);
t_start = WinL + 1;
t_end   = T - H + 1;


%%% Merge HEAVY-GAS
MV_Heavy_filename = sprintf('Output/VaRandES/VaR_ES_MV_HEAVY_GAS_k=%d', K);
MV_HEAVY = load(MV_Heavy_filename);

new_names = cellfun(@(x) strrep(strtrim(x), ' ', '_'), ...
                    MV_HEAVY.cell_model, 'UniformOutput', false);
VaRandES.Models = [VaRandES.Models; new_names(:)];

% Pre-allocate full length with NaN and fill with forecasts
J_new    = length(MV_HEAVY.cell_model);
VaR_new  = NaN(T, J_new, 2);
ES_new   = NaN(T, J_new, 2);
u_ES_new = NaN(T, J_new);

VaR_new(t_start:t_end, :, 1) = MV_HEAVY.VaR_mat_h_step_MV_GARCH_99;
VaR_new(t_start:t_end, :, 2) = MV_HEAVY.VaR_mat_h_step_MV_GARCH_975;
ES_new(t_start:t_end, :, 1) = MV_HEAVY.ES_mat_h_step_MV_GARCH_99;
ES_new(t_start:t_end, :, 2) = MV_HEAVY.ES_mat_h_step_MV_GARCH_975;
u_ES_new(t_start:t_end, J_new) = MV_HEAVY.u_ES_mat_h_step_MV_GARCH;

% Concatenate
VaRandES.VaR     = cat(2, VaRandES.VaR,     VaR_new);
VaRandES.ES      = cat(2, VaRandES.ES,      ES_new);
VaRandES.EmpPITs = cat(2, VaRandES.EmpPITs, u_ES_new);


%%% Merge univariate GARCH
GARCH_filename = sprintf('Output/VaRandES/VaR_ES_UV_GARCH_k=%d', K);
UV_GARCH = load(GARCH_filename);

new_names = cellfun(@(x) strrep(strtrim(x), ' ', '_'), ...
                    UV_GARCH.cell_model, 'UniformOutput', false);
VaRandES.Models = [VaRandES.Models; new_names(:)];

% Pre-allocate full length with NaN and fill with forecasts
J_new    = length(UV_GARCH.cell_model);
VaR_new  = NaN(T, J_new, 2);
ES_new   = NaN(T, J_new, 2);
u_ES_new = NaN(T, J_new);

VaR_new(t_start:t_end, :, 1) = UV_GARCH.VaR_mat_h_step_MV_GARCH_99;
VaR_new(t_start:t_end, :, 2) = UV_GARCH.VaR_mat_h_step_MV_GARCH_975;
ES_new(t_start:t_end, :, 1)  = UV_GARCH.ES_mat_h_step_MV_GARCH_99;
ES_new(t_start:t_end, :, 2)  = UV_GARCH.ES_mat_h_step_MV_GARCH_975;
u_ES_new(t_start:t_end, :)   = UV_GARCH.u_ES_mat_h_step_MV_GARCH;

% Concatenate
VaRandES.VaR     = cat(2, VaRandES.VaR,     VaR_new);
VaRandES.ES      = cat(2, VaRandES.ES,      ES_new);
VaRandES.EmpPITs = cat(2, VaRandES.EmpPITs, u_ES_new);






end