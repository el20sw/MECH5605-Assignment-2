%[text] # Pre\-processing the Data
%[text] ## Programatically importing the data
%[text] Note: requires changing some file and directory names from the downloaded data to make it consistent and able to work with this script
activities = {'level_ground_walking', 'ramp_asc', 'ramp_desc', 'sit_to_stand', 'stand_to_sit'};
subjects = {'j', 'lg', 'rs', 'vr'};
trials = {'01', '02', '03'};

dataStruct = struct();
index = 1;

for i = 1:length(activities)
    activity = activities{i};
    for j = 1:length(subjects)
        person = subjects{j};
        for k = 1:length(trials)
            trial = trials{k};
            file_path = sprintf('IMU Data\\%s\\%s_%s_trial_%s.dat', activity, activity, person, trial);
            if isfile(file_path)
                rawData = readtable(file_path);
                dataStruct(index).Activity = activity;
                dataStruct(index).FilePath = file_path;
                dataStruct(index).Data = rawData;
                index = index + 1;
            end
        end
    end
end
%%
%[text] ## Removing Bad Columns from the Data
%[text] What counts as a bad columns?
%[text] - Zero Variance \- the column is consistently the same value, this suggests that something is wrong with it \
%%
for i = 1:length(dataStruct)
    data = dataStruct(i).Data;
    badColumns = var(data{:,:}) == 0;
    cleanData = data(:, ~badColumns);
    dataStruct(i).CleanData = cleanData;
end
%%
%[text] ## Handling NaN values
%[text] Here, linear interpolation is used.
%%
for i = 1:length(dataStruct)
    rawData = dataStruct(i).CleanData;
    linear_fill = fillmissing(rawData, 'linear');
    mean_fill = fillmissing(rawData, 'movmean', 5);

    dataStruct(i).CleanData = linear_fill;
end
%%
%[text] ### Verifying that there are no NaN values in the files
%%
nan_found = false;
for i = 1:length(dataStruct)
    data = dataStruct(i).CleanData;
    if any(any(ismissing(data)))
        fprintf('NaN values found in dataset %d\n', i);
        nan_found = true;
    end
end

if ~nan_found
    fprintf('No NaN values found in any dataset.\n');
end
%%
%[text] ## Calculating the sample rate of the data
%[text] We have timestamps for each reading which can be used to get the sample rate of the data. The timestamps are on every 10th column
for i = 1:length(dataStruct)
    rawData = dataStruct(i).CleanData;
    timestamps = rawData(:, 1:10:end-10);
    time_diffs = diff(timestamps);
    avg_time_diff = mean(time_diffs, 'all');
    dataStruct(i).SampleTime = avg_time_diff.mean;
end
%%
%[text] We have the sample rate for each dataset, so we can remove the timestamp columns as these are not required for the classification
for i = 1:length(dataStruct)
    rawData = dataStruct(i).CleanData;
    timestampIdx = 1:10:width(rawData);
    dataStruct(i).CleanData(:, timestampIdx) = [];
end
%%
%[text] ## Filtering the Data
%%
%[text] ### Moving average filter
window_size = 5;
for i = 1:length(dataStruct)
    rawData = dataStruct(i).CleanData;
    numericData = rawData{:, :};
    data_filtered = movmean(numericData, window_size);
    dataStruct(i).CleanData = array2table(data_filtered, 'VariableNames', rawData.Properties.VariableNames);
end
%%
%[text] ## Labelling the Data
labelMap = containers.Map(activities, 1:numel(activities));
for i = 1:length(dataStruct)
    activity = dataStruct(i).Activity;
    dataStruct(i).Label = labelMap(activity);
end
%%
%[text] # Saving the Data
save('preprocessed.mat', 'dataStruct');
fprintf('Data saved to preprocessed.mat\n');

%[appendix]
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
