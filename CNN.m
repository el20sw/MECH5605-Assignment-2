load preprocessed.mat;

% Sliding Window Extraction
window_size_ms = 350;
delta_t_ms = 30;

for id_data = 1:length(dataStruct)
    data = dataStruct(id_data).CleanData;
    sample_time_ms = dataStruct(id_data).SampleTime;

    samples_per_window = round(window_size_ms / sample_time_ms);
    samples_per_stride = round(delta_t_ms / sample_time_ms);

    num_samples = size(data, 1);
    num_columns = size(data, 2);
    num_windows = floor((num_samples - samples_per_window) / samples_per_stride) + 1;

    % windows = zeros(num_windows, samples_per_window, num_columns);
    windows = zeros(num_windows, samples_per_window, num_columns);

    for i = 1:num_windows
        start_idx = (i - 1) * samples_per_stride + 1;
        end_idx = start_idx + samples_per_window - 1;

        window_data = data(start_idx:end_idx, :);
        window_data_numeric = table2array(window_data);

        windows(i, :, :) = window_data_numeric;
    end

    % Verify that windows is the correct shape
    assert(all(size(windows) == [num_windows, samples_per_window, num_columns]));

    % Insert the windows into the dataStruct
    dataStruct(id_data).Windows = windows;
end

% we also want windows that imitate an image - we need a fixed aspect ratio for the windows (1:1)
% we therefore need 63 samples per window and we can choose samples per stride arbitrarily
% we can then use the windows as input to a CNN

for id_data = 1:length(dataStruct)
    data = dataStruct(id_data).CleanData;
    sample_time_ms = dataStruct(id_data).SampleTime;

    % try for 63 x 84 windows (3:4)
    % samples_per_image = 84;

    samples_per_image = 63;
    samples_per_stride = 2;

    num_samples = size(data, 1);
    num_columns = size(data, 2);
    num_images = floor((num_samples - samples_per_image) / samples_per_stride) + 1;

    images = zeros(num_images, samples_per_image, num_columns);

    for i = 1:num_images
        start_idx = (i - 1) * samples_per_stride + 1;
        end_idx = start_idx + samples_per_image - 1;

        image_data = data(start_idx:end_idx, :);
        image_data_numeric = table2array(image_data);

        images(i, :, :) = image_data_numeric;
    end

    % Verify that windows is the correct shape
    assert(all(size(images) == [num_images, samples_per_image, num_columns]));

    % Insert the windows into the dataStruct
    dataStruct(id_data).Images = images;
end



% These windows are the data that are used as input for the CNN
save('windows.mat', 'dataStruct');
%%
load windows.mat;

uniqueSubjects = grp2idx(unique({dataStruct.Subject}));
uniqueSubjectsNonNumerical = unique({dataStruct.Subject});
subjectMap = containers.Map(uniqueSubjectsNonNumerical, uniqueSubjects);

% Create the labels
labels = [];
subjects = [];
for i = 1:length(dataStruct)
    imgs = dataStruct(i).Images;
    num_images = length(imgs);
    labels = [labels; repmat({dataStruct(i).Activity}, num_images, 1)];

    subject = subjectMap(dataStruct(i).Subject);
    subjects = [subjects; repmat(subject, num_images, 1)];
end

% The images have an aspect ratio of 1 and are the input to the CNN
% Prepare the data for the CNN and label it
images = cat(1, dataStruct.Images);
activity = {dataStruct.Activity};
labels = categorical(labels);

% each data point in images should be between 0 and 1
images = images - min(images(:));
images = images / max(images(:));

% reshape each image to be 63x63x1
images = permute(images, [2, 3, 1]);
images = reshape(images, [63, 63, 1, size(images, 3)]);
images = double(images);
% reshape each image to be 84x63x1
% images = permute(images, [2, 3, 1]);
% images = reshape(images, [84, 63, 1, size(images, 3)]);
% images = double(images);

% Split the data into training and testing sets - we leave out one subject
% for testing
subject = randsample(uniqueSubjects, 1);
testIdx = subjects == subject;
trainIdx = ~testIdx;

trainImages = images(:, :, :, trainIdx);
trainLabels = labels(trainIdx, :);

testImages = images(:, :, :, testIdx);
testLabels = labels(testIdx, :);

% shuffle the training data
idx = randperm(length(trainLabels));
trainImages = trainImages(:, :, :, idx);
trainLabels = trainLabels(idx, :);

% shuffle the test data
idx_test = randperm(length(testLabels));
testImages = testImages(:, :, :, idx_test);
testLabels = testLabels(idx_test, :);

% extract some validation data
valRatio = 0.2;
numTrain = round((1 - valRatio) * length(trainLabels));
valImages = trainImages(:, :, :, numTrain+1:end);
valLabels = trainLabels(numTrain+1:end, :);

trainImages = trainImages(:, :, :, 1:numTrain);
trainLabels = trainLabels(1:numTrain, :);

%%
% Define the CNN
inputSize = [63 63 1];
numClasses = numel(categories(labels));

layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3,8,Padding="same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,Stride=2)

    convolution2dLayer(3,16,Padding="same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,Stride=2)

    convolution2dLayer(3,32,Padding="same")
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numClasses)
    softmaxLayer
];

options = trainingOptions("sgdm", ...
    InitialLearnRate=0.01, ...
    MaxEpochs=20, ...
    ValidationData={valImages, valLabels}, ...
    ValidationFrequency=30, ...
    ValidationPatience=4, ...
    Metrics=["accuracy", "recall"], ...
    ObjectiveMetricName="recall", ...
    Plots="training-progress");

% Train the CNN
net = trainnet(trainImages, trainLabels, layers, "crossentropy", options);

% Test the CNN
classNames = categories(testLabels);

YPred = minibatchpredict(net, testImages);
YPred = onehotdecode(YPred, classNames, 2);
YValidation = testLabels;
accuracy = sum(YPred == YValidation) / numel(YValidation);
disp(accuracy);
confusionchart(YValidation, YPred);
title('Confusion Matrix for CNN');
xlabel('Predicted Activity');
ylabel('True Activity');
