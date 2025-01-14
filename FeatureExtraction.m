%[text] # Feature Extraction
%%
% Load the Data
load("preprocessed.mat");

% Sliding Window Feature Extraction
window_size_ms = 350;
delta_t_ms = 30;

% Extract Features from each Dataset
for id_data = 1:length(dataStruct)
    data = dataStruct(id_data).CleanData;
    sample_time_ms = dataStruct(id_data).SampleTime;
    
    samples_per_window = round(window_size_ms / sample_time_ms);
    samples_per_stride = round(delta_t_ms / sample_time_ms);
    
    num_samples = size(data, 1);
    num_windows = floor((num_samples - samples_per_window) / samples_per_stride) + 1;
    
    featureNames = {'max', 'min', 'mean', 'std', 'rms', 'max_gradient', 'zero_crossings'};
    num_features = length(featureNames);
    num_columns = size(data, 2);
    
    % Create unique variable names for the features - we get max, min, mean,
    % std, rms, max_gradient, zero_crossings for each of the columns and we
    % have 63 columns, 7 features, therefore 63*7=441 feature columns
    variableNames = cell(1, num_features * num_columns);
    for col = 1:num_columns
        for feat = 1:num_features
            variableNames{(col-1)*num_features + feat} = sprintf('%s_col%d', featureNames{feat}, col);
        end
    end
    
    features = array2table(zeros(num_windows, num_features * num_columns), 'VariableNames', variableNames);
    
    for i = 1:num_windows
        start_idx = (i - 1) * samples_per_stride + 1;
        end_idx = start_idx + samples_per_window - 1;
    
        window_data = data(start_idx:end_idx, :);
        window_data_numeric = table2array(window_data);
    
        max_values = max(window_data_numeric);
        min_values = min(window_data_numeric);
        mean_values = mean(window_data_numeric);
        std_values = std(window_data_numeric);
        rms_values = rms(window_data_numeric);
        max_gradient = max(diff(window_data_numeric));
        zero_crossings = zeros(1, size(window_data_numeric, 2));
        for col = 1:size(window_data_numeric, 2)
            zero_crossings(col) = sum(window_data_numeric(1:end-1, col) .* window_data_numeric(2:end, col) < 0);
        end
    
        features(i, :) = num2cell([max_values, min_values, mean_values, std_values, rms_values, max_gradient, zero_crossings]);
    end
    
    % Verify that features is 441 wide
    assert(width(features) == 441);
    % Verify that featurea is not zero high
    assert(height(features) ~= 0);
    
    % Consolidate into data structure: features should be 441 wide and each
    % row is a sliding window sample of the preprocessed data
    dataStruct(id_data).Features = features;
end

save('preprocessed_with_features.mat', 'dataStruct');

%[appendix]
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
