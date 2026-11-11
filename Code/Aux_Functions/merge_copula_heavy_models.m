function VaRandES = merge_copula_heavy_models(VaRandES)
%MERGE_COPULA_HEAVY_MODELS Append HEAVY-Copula models to VaRandES.
%
%   VaRandES = MERGE_HEAVY_COPULA_MODELS(VaRandES) loads pre-computed VaR, 
%   ES and empirical PITs from HEAVY-copula models and appends them to the 
%   existing VaRandES struct along the model dimension.
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
    'Output/VaRandES/VaRandES_EquallyWeighted_HEAVY'
};
assets_str  = strjoin(VaRandES.assets, '_');

for s = 1:length(merge_files)
    data = load([merge_files{s} '_' assets_str]);
    data = data.VaRESOut;
    new_names = cellfun(@(x) strrep(strtrim(x), ' ', '_'), ...
                        data.Models, 'UniformOutput', false);
    VaRandES.Models = [VaRandES.Models; new_names(:)];
    J_new    = length(data.Models);
    VaR_new  = NaN(T, J_new, 2);
    ES_new   = NaN(T, J_new, 2);
    VaR_new(:, :, :) = data.VaR(:, :, 1, :);
    ES_new(:,  :, :) = data.ES(:, :, 1, :);
    u_ES_new         = data.EmpPITs;
    VaRandES.VaR     = cat(2, VaRandES.VaR,     VaR_new);
    VaRandES.ES      = cat(2, VaRandES.ES,      ES_new);
    VaRandES.EmpPITs = cat(2, VaRandES.EmpPITs, u_ES_new);
end

end