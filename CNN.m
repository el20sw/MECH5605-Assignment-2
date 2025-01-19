%% Miscellaneous Setup
% Clear the workspace
clear;
close all;
clc;

% Set the random seed for reproducibility
rng(42);

% Turn off LaTeX interpretation globally
set(0, 'DefaultTextInterpreter', 'none')
set(0, 'DefaultAxesTickLabelInterpreter', 'none')
set(0, 'DefaultLegendInterpreter', 'none')
set(0, 'DefaultColorbarTickLabelInterpreter', 'none')

%% Extract 'Images' and Windows
load preprocessed.mat;

function dataStruct = extractFeatures(dataStruct)
    % from the assignment brief
    window_size_ms = 350;
    delta_t_ms = 30;

    % normalise the data
    dataStruct = normaliseData(dataStruct);

    dataStruct = extractSlidingWindows(dataStruct, ...
        window_size_ms, delta_t_ms);
    dataStruct = extractDataImages(dataStruct, 63, 2);
    dataStruct = extractDataImages(dataStruct, 64, 2);

    save('windows.mat', 'dataStruct');
end

function dataStruct = normaliseData(dataStruct)
    % normalise data with min-max scaling
    for i = 1:length(dataStruct)
        data = dataStruct(i).CleanData;
        minVal = min(data, [], 1);
        maxVal = max(data, [], 1);
        data = (data - minVal) ./ (maxVal - minVal);
        dataStruct(i).CleanData = data;
    end
end

function dataStruct = extractSlidingWindows(dataStruct, windowSize, strideSize)
    for i = 1:length(dataStruct)
        data = dataStruct(i).CleanData;
        sample_time_ms = dataStruct(i).SampleTime;
        samples_per_window = round(windowSize / sample_time_ms);
        samples_per_stride = round(strideSize / sample_time_ms);

        windows = extractDataWindows(data, samples_per_window, samples_per_stride);
        dataStruct(i).Windows = windows;
    end
end

function dataStruct = extractDataImages(dataStruct, imageSize, strideSize)
    for i = 1:length(dataStruct)
        data = dataStruct(i).CleanData;
        images = extractDataWindows(data, imageSize, strideSize);

        if imageSize == 64
            % If the image size is 64, then we pad the images to be 64x64
            paddedImages = zeros(size(images, 1), 64, 64);
            paddedImages(:, :, 1:63) = images;
            dataStruct(i).PaddedImages = paddedImages;
        else
            dataStruct(i).Images = images;
        end
    end
end

function windows = extractDataWindows(data, windowSize, strideSize)
    numSamples = size(data, 1);
    numColumns = size(data, 2);
    numWindows = floor((numSamples - windowSize) / strideSize) + 1;

    windows = zeros(numWindows, windowSize, numColumns);

    for i = 1:numWindows
        startIdx = (i - 1) * strideSize + 1;
        endIdx = startIdx + windowSize - 1;

        windowData = data(startIdx:endIdx, :);
        windows(i, :, :) = table2array(windowData);
    end

    assert(all(size(windows) == [numWindows, windowSize, numColumns]));
end

dataStruct = extractFeatures(dataStruct);
%% Process the Images
% Load the windows
load windows.mat;

% Extract the images
images = cat(1, dataStruct.Images);
paddedImages = cat(1, dataStruct.PaddedImages);

imageLabels = [];
imageSubjects = [];

for i = 1:length(dataStruct)
    numImages = length(dataStruct(i).Images);
    imageLabels = [imageLabels; repmat({dataStruct(i).Activity}, numImages, 1)];
    imageSubjects = [imageSubjects; repmat({dataStruct(i).Subject}, numImages, 1)];
end

paddedLabels = [];
paddedSubjects = [];

for i = 1:length(dataStruct)
    numImages = length(dataStruct(i).PaddedImages);
    paddedLabels = [paddedLabels; repmat({dataStruct(i).Activity}, numImages, 1)];
    paddedSubjects = [paddedSubjects; repmat({dataStruct(i).Subject}, numImages, 1)];
end

% Convert the labels to categorical
imageLabels = categorical(imageLabels);
paddedLabels = categorical(paddedLabels);

% Convert the subjects to numerical representation
uniqueSubjects = unique(imageSubjects);
subjectMap = containers.Map(uniqueSubjects, 1:length(uniqueSubjects));
imageSubjects = cellfun(@(x) subjectMap(x), imageSubjects);
paddedSubjects = cellfun(@(x) subjectMap(x), paddedSubjects);

% Verify the shapes
assert(size(images, 1) == length(imageLabels));
assert(size(images, 1) == length(imageSubjects));
assert(size(paddedImages, 1) == length(paddedLabels));
assert(size(paddedImages, 1) == length(paddedSubjects));
%% Reshape the Images
% Reshape to be [features, time, channel, samples]
% currently the shape is [samples, time, features]
reshapedImages = reshapeImages(images);
reshapedPaddedImages = reshapeImages(paddedImages);

function reshapedImages = reshapeImages(images)
    reshapedImages = permute(images, [3, 2, 1]);
    reshapedImages = reshape(reshapedImages, [size(images, 2), ...
        size(images, 3), 1, size(images, 1)]);
end
%% Plot some sampled images
numImages = 5;
[sampledImages, idx] = datasample(reshapedImages, numImages, 4);

figure;
for i = 1:numImages
    subplot(1, numImages, i);
    imshow(sampledImages(:, :, 1, i));
    title(imageLabels(idx(i)));
end

% Plot some sampled padded images
[sampledPaddedImages, idx] = datasample(reshapedPaddedImages, numImages, 4);

figure;
for i = 1:numImages
    subplot(1, numImages, i);
    imshow(sampledPaddedImages(:, :, 1, i));
    title(paddedLabels(idx(i)));
end
%% Plot the distribution of the labels
figure;
histogram(imageLabels);
title('Distribution of Labels for Images');
fprintf('Proportion of each label for Images\n');
tabulate(imageLabels);

figure;
histogram(paddedLabels);
title('Distribution of Labels for Padded Images');
fprintf('Proportion of each label for Padded Images\n');
tabulate(paddedLabels);
%% Train/Test/Validation Splits
% Split the data into training, validation and test sets
[trainImages, trainLabels, valImages, valLabels, testImages, testLabels] = ...
    splitData(reshapedPaddedImages, paddedLabels, paddedSubjects, 0.15);

function [trainImages, trainLabels, valImages, valLabels, testImages, testLabels] = ...
    splitData(images, labels, subjects, valSplit)
    uniqueSubjects = unique(subjects);

    % Split the data into training, validation and test sets
    % We leave out one subject for testing
    subject = randsample(uniqueSubjects, 1);
    testIdx = subjects == subject;
    trainIdx = ~testIdx;

    trainImages = images(:, :, :, trainIdx);
    trainLabels = labels(trainIdx, :);

    testImages = images(:, :, :, testIdx);
    testLabels = labels(testIdx, :);

    % Shuffle the training data
    idx = randperm(length(trainLabels));
    trainImages = trainImages(:, :, :, idx);
    trainLabels = trainLabels(idx, :);

    % Extract some validation data
    valIdx = rand(size(trainLabels, 1), 1) < valSplit;

    valImages = trainImages(:, :, :, valIdx);
    valLabels = trainLabels(valIdx, :);

    trainImages = trainImages(:, :, :, ~valIdx);
    trainLabels = trainLabels(~valIdx, :);

    % Verify the shapes
    assert(size(trainImages, 4) == length(trainLabels));
    assert(size(valImages, 4) == length(valLabels));
    assert(size(testImages, 4) == length(testLabels));
end
%% Balance the Training Data
[trainImages, trainLabels] = balanceData(trainImages, trainLabels);

function [balancedImages, balancedLabels] = balanceData(images, labels)
    cats = categories(labels);
    numCats = length(cats);
    numSamplesPerCategory = countcats(labels);

    minSamples = min(numSamplesPerCategory);

    totalSamples = numCats * minSamples;
    balancedImages = zeros([size(images,1), size(images,2), ...
        size(images,3), totalSamples], 'like', images);
    balancedLabels = categorical(repmat(cats(1), totalSamples, 1));

    currentIdx = 1;
    for i = 1:numCats
        categoryIdx = find(labels == cats{i});
        selectedIdx = categoryIdx(randperm(length(categoryIdx), minSamples));
        insertIdx = currentIdx:(currentIdx + minSamples - 1);

        balancedImages(:,:,:,insertIdx) = images(:,:,:,selectedIdx);
        balancedLabels(insertIdx) = labels(selectedIdx);

        currentIdx = currentIdx + minSamples;
    end
end

figure;
histogram(trainLabels);
title('Distribution of Labels for Images');
fprintf('Proportion of each label for Images\n');
tabulate(trainLabels);
%% Define the CNN Architecture
inputSize = [size(trainImages, 1), size(trainImages, 2), size(trainImages, 3)];
outputSize = numel(categories(trainLabels));

layers = [
    imageInputLayer(inputSize, 'Name', 'input')

    convolution2dLayer(3,8,Padding='same',Name='conv_1')
    batchNormalizationLayer(Name='bn_1')
    reluLayer(Name='relu_1')

    maxPooling2dLayer(2,Stride=2,Name='maxpool_1')

    convolution2dLayer(3,16,Padding='same',Name='conv_2')
    batchNormalizationLayer(Name='bn_2')
    reluLayer(Name='relu_2')

    maxPooling2dLayer(2,Stride=2,Name='maxpool_2')

    convolution2dLayer(3,32,Padding='same',Name='conv_3')
    batchNormalizationLayer(Name='bn_3')
    reluLayer(Name='relu_3')

    fullyConnectedLayer(outputSize,Name='fc')
    softmaxLayer(Name='softmax')
];

% Define the options for training
options = trainingOptions('adam', ...
    InitialLearnRate=0.001, ...
    MaxEpochs=10, ...
    MiniBatchSize=32, ...
    ValidationData={valImages, valLabels}, ...
    ValidationFrequency=30, ...
    ValidationPatience=10, ...
    LearnRateSchedule='piecewise', ...
    LearnRateDropFactor=0.1, ...
    LearnRateDropPeriod=20, ...
    Shuffle='every-epoch', ...
    Metrics=["accuracy","fscore","recall"], ...
    ObjectiveMetricName="fscore", ...
    Plots='training-progress', ...
    Verbose=true);

% Train the CNN
net = trainnet(trainImages, trainLabels, layers, 'crossentropy', options);

% Test the CNN
classNames = categories(testLabels);

YPred = minibatchpredict(net, testImages);
YPred = onehotdecode(YPred, classNames, 2);
YTrue = testLabels;
accuracy = sum(YPred == YTrue) / numel(YTrue);
fprintf('Accuracy: %.2f\n', accuracy);

% Plot the confusion matrix
figure
cm = confusionchart(YTrue, YPred);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
title('Confusion Matrix for CNN');
xlabel('Predicted Activity');
ylabel('True Activity');
%% Define an Alternative CNN Architecture
% Following: INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC
% from https://cs231n.github.io/convolutional-networks/#conv
% note: we use the padded input for this architecture due to the 64x64 input size
inputSize = [64, 64, 1];
outputSize = numel(categories(trainLabels));

layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3,16,Padding='same',Name='conv_1')
    reluLayer(Name='relu_1')
    convolution2dLayer(3,16,Padding='same',Name='conv_2')
    reluLayer(Name='relu_2')
    maxPooling2dLayer(2,Stride=2,Name='maxpool_1')

    convolution2dLayer(3,32,Padding='same',Name='conv_3')
    reluLayer(Name='relu_3')
    convolution2dLayer(3,32,Padding='same',Name='conv_4')
    reluLayer(Name='relu_4')
    maxPooling2dLayer(2,Stride=2,Name='maxpool_2')

    convolution2dLayer(3,64,Padding='same',Name='conv_5')
    reluLayer(Name='relu_5')
    convolution2dLayer(3,64,Padding='same',Name='conv_6')
    reluLayer(Name='relu_6')
    maxPooling2dLayer(2,Stride=2,Name='maxpool_3')

    fullyConnectedLayer(128,Name='fc_1')
    reluLayer(Name='relu_7')
    fullyConnectedLayer(64,Name='fc_2')
    reluLayer(Name='relu_8')

    fullyConnectedLayer(outputSize,Name='fc_3')
    softmaxLayer(Name='softmax')
];

% Define the options for training
options = trainingOptions('adam', ...
    InitialLearnRate=0.001, ...
    MaxEpochs=10, ...
    MiniBatchSize=32, ...
    ValidationData={valImages, valLabels}, ...
    ValidationFrequency=30, ...
    ValidationPatience=10, ...
    LearnRateSchedule='piecewise', ...
    LearnRateDropFactor=0.1, ...
    LearnRateDropPeriod=20, ...
    Shuffle='every-epoch', ...
    Metrics=["accuracy","fscore","recall"], ...
    ObjectiveMetricName="fscore", ...
    Plots='training-progress', ...
    Verbose=true);

% Train the CNN
net = trainnet(trainImages, trainLabels, layers, 'crossentropy', options);

% Test the CNN
classNames = categories(testLabels);

YPred = minibatchpredict(net, testImages);
YPred = onehotdecode(YPred, classNames, 2);
YTrue = testLabels;
accuracy = sum(YPred == YTrue) / numel(YTrue);
fprintf('Accuracy: %.2f\n', accuracy);

% Plot the confusion matrix
confusionchart(YTrue, YPred);
title('Confusion Matrix for CNN');
xlabel('Predicted Activity');
ylabel('True Activity');
%% Define a Deeper CNN Architecture
% Following: https://dl.acm.org/doi/pdf/10.1145/3214277
% C (1×3@32)→C (1×3@32)→C (1×3@32)→C (1×3@32)→P (1×2@64)→ C
% (1×3@64)→C (1×3@64)→ C (1×3@64)→C (1×3@64)→P (1×5@64)→F
inputSize = [64, 64, 1];
outputSize = numel(categories(trainLabels));

layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3,32,Padding='same',Name='conv_1')
    convolution2dLayer(3,32,Padding='same',Name='conv_2')
    convolution2dLayer(3,32,Padding='same',Name='conv_3')
    reluLayer(Name='relu_1')

    maxPooling2dLayer(3,Stride=2,Name='maxpool_1')

    convolution2dLayer(2,64,Padding='same',Name='conv_4')
    convolution2dLayer(3,64,Padding='same',Name='conv_5')
    convolution2dLayer(3,64,Padding='same',Name='conv_6')
    convolution2dLayer(3,64,Padding='same',Name='conv_7')
    reluLayer(Name='relu_2')

    maxPooling2dLayer(3,Stride=2,Name='maxpool_2')

    fullyConnectedLayer(outputSize,Name='fc')
    softmaxLayer(Name='softmax')
];

% Define the options for training
options = trainingOptions('adam', ...
    InitialLearnRate=0.001, ...
    MaxEpochs=10, ...
    MiniBatchSize=32, ...
    ValidationData={valImages, valLabels}, ...
    ValidationFrequency=30, ...
    ValidationPatience=10, ...
    LearnRateSchedule='piecewise', ...
    LearnRateDropFactor=0.1, ...
    LearnRateDropPeriod=20, ...
    Shuffle='every-epoch', ...
    Metrics=["accuracy","fscore","recall"], ...
    ObjectiveMetricName="fscore", ...
    Plots='training-progress', ...
    Verbose=true);

% Train the CNN
net = trainnet(trainImages, trainLabels, layers, 'crossentropy', options);

% Test the CNN
classNames = categories(testLabels);

YPred = minibatchpredict(net, testImages);
YPred = onehotdecode(YPred, classNames, 2);
YTrue = testLabels;
accuracy = sum(YPred == YTrue) / numel(YTrue);
fprintf('Accuracy: %.2f\n', accuracy);

% Plot the confusion matrix
confusionchart(YTrue, YPred);
title('Confusion Matrix for CNN');
xlabel('Predicted Activity');
ylabel('True Activity');
