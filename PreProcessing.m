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
%[text] - Zero Variance \- the column is consistently the same value, this suggests that something is wrong with it
%[text] - Those with malformed column names \- we expect columns to be the timestamp (position name, i.e. Pelvis, Thigh\_L, etc.) and then the same but with a 1 to 9 suffix. Anything outside this pattern is to be discarded \
%%
for i = 1:length(dataStruct)
    data = dataStruct(i).Data;
    badColumns = var(data{:,:}) == 0;
    cleanData = data(:, ~badColumns);
    dataStruct(i).CleanData = cleanData;
end
%%
% There should be 63 columns in the data -
% if the column name is greater than 9, then it is not a valid column
% find each column which is just the position and remove it
for i = 1:length(dataStruct)
    data = dataStruct(i).CleanData;
    badColumns = true(1, width(data));
    for col = 1:width(data)
        colName = data.Properties.VariableNames{col};
        % we get a match when the regex is correct - if the result is empty
        % there was no match and the column is invalid, we inverse this to
        % make it go to zero and from there it works
        regexResult = isempty(regexp(colName, '^(Thigh|Shank|Foot|Pelvis)(_)?([L|R])?(_)?([1-9])?$', 'once'));
        if regexResult
            badColumns(col) = false;
        end
    end
    dataStruct(i).CleanData = data(:, badColumns);
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
    data = dataStruct(i).CleanData;
    timestampCols = false(1, width(data));
    for col = 1:width(data)
        colName = data.Properties.VariableNames{col};
        regexResult = ~isempty(regexp(colName, '^(Thigh|Shank|Foot|Pelvis)(_)?([L|R])?$', 'once'));
        if regexResult
            timestampCols(col) = true;
        end
    end
    timestamps = data(:, timestampCols);
    time_diffs = diff(timestamps);
    avg_time_diff = mean(time_diffs, 'all');
    dataStruct(i).SampleTime = avg_time_diff.mean;
    % remove the timestamp columns as these aren't required for
    % classification
    dataStruct(i).CleanData(:, timestampCols) = [];
end
%%
%[text] ## Filtering the Data
%%
%[text] ### Moving average filter
window_size = 5;
for i = 1:length(dataStruct)
    data = dataStruct(i).CleanData;
    numericData = data{:, :};
    data_filtered = movmean(numericData, window_size);
    dataStruct(i).CleanData = array2table(data_filtered, 'VariableNames', data.Properties.VariableNames);
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
