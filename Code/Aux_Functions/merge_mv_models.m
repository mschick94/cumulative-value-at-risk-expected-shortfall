function VaRandES = merge_mv_models(VaRandES)
%MERGE_MV_MODELS Append Anne's model forecasts to VaRandES.
%
%   VaRandES = MERGE_MV_MODELS(VaRandES) loads pre-computed VaR, ES and
%   empirical PITs from Anne's univariate and multivariate models and 
%   appends them to the existing VaRandES struct along the model dimension.
%
%   INPUT:
%       VaRandES : Struct, output from simulate_all_var_es with optional
%                  pf_garch_benchmark models already added. Must contain:
%                  .VaR      - (T x J x P) VaR forecasts
%                  .ES       - (T x J x P) ES forecasts
%                  .EmpPITs  - (T x J) empirical PITs
%                  .Models   - (J x 1) cell array of model names
%                  .assets   - cell array of asset names
%                  .WindLength, .H, .dates
%
%   OUTPUT:
%       VaRandES : Updated struct with co-author's models appended:
%                  .VaR      - (T x J_new x P)
%                  .ES       - (T x J_new x P)
%                  .EmpPITs  - (T x J_new)
%                  .Models   - (J_new x 1) cell array including new models
%                  All other fields unchanged.
%
%   NOTES:
%       - Files loaded from Output/VaRandES/ with suffix _k=K
%       - Model names cleaned via strtrim/strrep to match naming convention
%       - NaN padding applied for burn-in (WinL) and last H-1 observations
%       - Add new models by appending to merge_files cell array


% Read out estimation set-up
K       = length(VaRandES.assets);
WinL    = VaRandES.WindLength;
H       = VaRandES.H;
T       = length(VaRandES.dates);
t_start = WinL + 1;
t_end   = T - H + 1;


% Flexibly reading in and adding Anne's VAR and ES output
merge_files = {
    'Output/VaRandES/VaR_ES_MV_HEAVY_GAS'
    'Output/VaRandES/VaR_ES_MV_RISKMETRICS'
    'Output/VaRandES/VaR_ES_UV_HEAVY'
};

for s = 1:length(merge_files)
    data = load(sprintf('%s_k=%d', merge_files{s}, K));
    new_names = cellfun(@(x) strrep(strtrim(x), ' ', '_'), ...
                        data.cell_model, 'UniformOutput', false);
    VaRandES.Models = [VaRandES.Models; new_names(:)];
    J_new    = length(data.cell_model);
    VaR_new  = NaN(T, J_new, 2);
    ES_new   = NaN(T, J_new, 2);
    u_ES_new = NaN(T, J_new);
    VaR_new(t_start:t_end, :, 1) = data.VaR_mat_h_step_MV_GARCH_99;
    VaR_new(t_start:t_end, :, 2) = data.VaR_mat_h_step_MV_GARCH_975;
    ES_new(t_start:t_end,  :, 1) = data.ES_mat_h_step_MV_GARCH_99;
    ES_new(t_start:t_end,  :, 2) = data.ES_mat_h_step_MV_GARCH_975;
    u_ES_new(t_start:t_end, :)   = data.u_ES_mat_h_step_MV_GARCH;
    VaRandES.VaR     = cat(2, VaRandES.VaR,     VaR_new);
    VaRandES.ES      = cat(2, VaRandES.ES,       ES_new);
    VaRandES.EmpPITs = cat(2, VaRandES.EmpPITs, u_ES_new);
end

end