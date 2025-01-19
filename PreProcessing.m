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
                dataStruct(index).Subject = person;
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
%[text] - Columns that fall outside the expected column names \- i.e. aren't thigh, shank, foot, pelvis
%[text] - Extra timestamp columns \- these have been visually identified through examination of the data and are being programatically removed \
% removing columns not referencing one of the IMUs
for i = 1:length(dataStruct)
    data = dataStruct(i).Data;
    validColumns = true(1,width(data));
    for col = 1:width(data)
        colName = data.Properties.VariableNames{col};
        result = isempty(regexp(colName, '^(Thigh|Shank|Foot|Pelvis)*', 'once'));
        if result
            validColumns(col) = false;
        end
    end
    dataStruct(i).CleanData = data(:, validColumns);
end
%%
% handling extra columns - we should only have 1 timestamp column and 9
% data columns
for i = 1:length(dataStruct)
    data = dataStruct(i).CleanData;
    % find any bad columns - these are outside the identified naming structure
    badColumns = false(1,width(data));
    % find the timestamp columns
    timestampColumns = false(1, width(data));
    for col = 1:width(data)
        colName = data.Properties.VariableNames{col};

        isTimestamp = ~isempty(regexp(colName, '^(Thigh|Shank|Foot|Pelvis)(_)?([L|R])?$', 'once'));
        if isTimestamp
            timestampColumns(col) = true;
        end
    end

    timestampIdx = find(timestampColumns);

    % if the difference between each timestamp is greater than 10, then we remove the difference from after that timestamp
    % this is because the timestamp is likely to be a bad column
    if ~isempty(timestampIdx)
        % get the index of the last piece of data and +1 to it for a pseudo-timestamp
        timestampIdx = [timestampIdx (width(data) + 1)];
        timestampIdxDiffs = diff(timestampIdx);
        for j = 1:length(timestampIdxDiffs)
            difference = timestampIdxDiffs(j);
            if difference > 10
                num_cols_to_remove = difference - 10;
                idx = timestampIdx(j);
                badColumns(idx+1:idx+num_cols_to_remove) = true;
            end
        end
    end

    % remove the badColumns from the data
    dataStruct(i).CleanData = data(:, ~badColumns);
end
%%
%[text] ## Handling NaN values
%[text] Here, linear interpolation is used.
for i = 1:length(dataStruct)
    rawData = dataStruct(i).CleanData;
    linear_fill = fillmissing(rawData, 'linear');
    mean_fill = fillmissing(rawData, 'movmean', 5);

    dataStruct(i).CleanData = linear_fill;
end
%%
%[text] ### Verifying that there are no NaN values in the files
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
%[text] We have timestamps for each reading which can be used to get the sample rate of the data.
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
%[text] ## Each column for each body part needs to be 1 to 9
%[text] The columns for each body part should be 1 to 9, if they aren't then we need to rename them
for i = 1:length(dataStruct)
    data = dataStruct(i).CleanData;
    varNames = data.Properties.VariableNames;

    bodyMap = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for c = 1:length(varNames)
        colName = varNames{c};
        % tokens = regexp(colName, '^(Thigh|Shank|Foot|Pelvis)_[L|R]?_(\d+)$', 'tokens', 'once');
        tokens = regexp(colName, ...
            '^(?<bp>(Thigh|Shank|Foot|Pelvis))(?:_?)(?<side>[LR])?_(?<num>\d+)', ...
            'names');
        if ~isempty(tokens)
            if ~isempty(tokens.side)
                location = sprintf('%s_%s', tokens.bp, tokens.side);
            else
                location = tokens.bp;
            end
            if isKey(bodyMap, location)
                bodyMap(location) = [bodyMap(location) str2double(tokens.num)];
            else
                bodyMap(location) = str2double(tokens.num);
            end
        end

        newVarNames = varNames;
        bpKeys = keys(bodyMap);
        for j = 1:length(bpKeys)
            key = bpKeys{j};
            nums = bodyMap(key);
            if min(nums) > 1
                % rename the columns such that they are 1 to 9
                for k = 1:length(nums)
                    oldName = sprintf('%s_%d', key, nums(k));
                    newName = sprintf('%s_%d', key, k);
                    newVarNames = strrep(newVarNames, oldName, newName);
                end
            end
        end

        dataStruct(i).CleanData.Properties.VariableNames = newVarNames;
    end
end

%[text] ## Verifying the column names
for i = 1:length(dataStruct)
    data = dataStruct(i).CleanData;
    varNames = data.Properties.VariableNames;
    for c = 1:length(varNames)
        colName = varNames{c};
        tokens = regexp(colName, ...
            '^(?<bp>(Thigh|Shank|Foot|Pelvis))(?:_?)(?<side>[LR])?_(?<num>\d+)', ...
            'names');
        if ~isempty(tokens)
            num = str2double(tokens.num);
            if num < 1 || num > 9
                fprintf('DataStruct: %d\tInvalid column name: %s\n', i, colName);
            end
        end
    end
end
%%
%[text] ## Filtering the Data
% Since typical gait data for normal walking has frequency components in the range of 0.5 to 3.5 Hz,
% a 10-th order 𝐵𝑢𝑡𝑡𝑒𝑟𝑤𝑜𝑟𝑡ℎ bandpass filter was used to extract the required frequency components from the resultant vectors of the IMUs
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
%[text] ## Verifying the Data
%[text] The data should be 63 wide, and there should be 60 datasets
assert(length(dataStruct) == 60);
for i = 1:length(dataStruct)
    data = dataStruct(i).CleanData;
    assert(width(data) == 63);
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
