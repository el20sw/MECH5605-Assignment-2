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
%%
load preprocessed_with_features.mat

allFeatures = [];

uniqueSubjects = grp2idx(unique({dataStruct.Subject}));
for i = 1:length(dataStruct)
    dataStruct(i).Features.Activity = repmat({dataStruct(i).Activity}, size(dataStruct(i).Features, 1), 1);
    dataStruct(i).Features.Subject = repmat({dataStruct(i).Subject}, size(dataStruct(i).Features, 1), 1);
end

% Concatenate all the features into a single table
for i = 1:length(dataStruct)
    features = dataStruct(i).Features;
    allFeatures = [allFeatures; features];
end

tabulate(allFeatures.Activity);
%% Train/Test/Validation Split
% split the data into training and testing sets
% using leave one subject out for testing

% first we need to convert the subjects into a numerical representation
allFeatures.Subject = grp2idx(allFeatures.Subject);
subject = randsample(uniqueSubjects, 1);
testIdx = allFeatures.Subject == subject;
trainIdx = ~testIdx;

% extract the targets and remove from the features
allTargets = allFeatures.Activity;
allFeatures.Activity = [];

% get the heading names, sans 'Subject' and 'Activity'
headingNames = allFeatures.Properties.VariableNames;
headingNames = headingNames(~ismember(headingNames, {'Subject', 'Activity'}));

% convert allFeatures into a matrix - trees accept a table: columns are variables, rows are samples
allFeatures = table2array(allFeatures);
allFeatures(:, end) = [];

trainFeatures = allFeatures(trainIdx, 1:end);
trainTargets = allTargets(trainIdx, :);
trainTargets = categorical(trainTargets);

testFeatures = allFeatures(testIdx, 1:end);
testTargets = allTargets(testIdx, :);
testTargets = categorical(testTargets);

data = array2table(allFeatures);
data.Properties.VariableNames = headingNames;
data.Activity = allTargets;
data.Properties.VariableNames = [headingNames, {'Activity'}];

trainFeatures = array2table(trainFeatures);
trainFeatures.Properties.VariableNames = headingNames;
trainFeatures.Activity = trainTargets;
trainFeatures.Properties.VariableNames = [headingNames, {'Activity'}];

% get a random sample for validation
valSplit = 0.15;
valIdx = rand(size(trainFeatures, 1), 1) < valSplit;
valFeatures = trainFeatures(valIdx, :);
valTargets = trainTargets(valIdx, :);
valFeatures.Activity = [];
valTargets = categorical(valTargets);

trainFeatures = trainFeatures(~valIdx, :);
trainTargets = trainTargets(~valIdx, :);

testFeatures = array2table(testFeatures);
testFeatures.Properties.VariableNames = headingNames;
testFeatures.Activity = testTargets;
testFeatures.Properties.VariableNames = [headingNames, {'Activity'}];

classNames = unique(data.Activity);
%% Baseline Model
tTree = templateTree("MaxNumSplits", 5);
model = fitcensemble(trainFeatures,'Activity', ...
    'NumBins',50, ...
    'Method','AdaBoostM2', ...
    'Learners',tTree);
predictions = predict(model, testFeatures);
predictions = categorical(predictions);
accuracy = sum(predictions == testTargets)/numel(testTargets);
fprintf("Accuracy: %.2f\n", accuracy);
figure
cm = confusionchart(testTargets, predictions);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
title('Confusion Matrix for RF');
xlabel('Predicted Activity');
ylabel('True Activity');
%% Hyperparameter Tuning
% Grid search over Ensemble and Tree hyperparameters
% Define the hyperparameters to search over

numTrees = 100:100:500;
maxNumSplits = 10:10:50;
minLeafSize = [1, 10:10:50];
% Grid size
gridSize = length(numTrees) * length(maxNumSplits) * length(minLeafSize);

% Grid search for hyperparameters
disp('Starting grid search for hyperparameters');
disp(['Grid size: ', num2str(gridSize)]);

% create the grid to search over
grid = [];
for i = 1:length(numTrees)
    for j = 1:length(maxNumSplits)
        for k = 1:length(minLeafSize)
            grid = [grid; [numTrees(i), maxNumSplits(j), minLeafSize(k)]];
        end
    end
end
%%
gridSize = size(grid, 1);
hyperparameters = table('Size', [gridSize, 4], ...
    'VariableTypes', {'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'NumTrees', 'MaxNumSplits', 'MinLeafSize', 'Accuracy'});
fprintf('Starting grid search\n');
parfor idx = 1:gridSize
    row = grid(idx, :);
    numTrees = double(row(1));
    maxNumSplits = double(row(2));
    minLeafSize = double(row(3));

    tTree = templateTree('MaxNumSplits', maxNumSplits, ...
        'MinLeafSize', minLeafSize);
    model = fitcensemble(trainFeatures,'Activity', ...
        'Method','AdaBoostM2', ...
        'Learners',tTree, ...
        'NumLearningCycles', numTrees);
    predictions = predict(model, valFeatures);
    predictions = categorical(predictions);
    accuracy = sum(predictions == valTargets)/numel(valTargets);
    params = {numTrees, maxNumSplits, minLeafSize, accuracy};
    hyperparameters(idx, :) = params;

    fprintf('Accuracy: %.2f\n', accuracy);
end

save('rf_tree_hyperparameters.mat', 'hyperparameters');
%% Best Hyperparameters
% Find the best hyperparameters
load rf_tree_hyperparameters.mat

[~, idx] = max(hyperparameters.Accuracy);
bestHyperparameters = hyperparameters(idx, :);
numTrees = bestHyperparameters.NumTrees;
maxNumSplits = bestHyperparameters.MaxNumSplits;
minLeafSize = bestHyperparameters.MinLeafSize;
fprintf('Best Hyperparameters\n');
disp(bestHyperparameters);
%% Train the Model
tTree = templateTree('MaxNumSplits', maxNumSplits, ...
        'MinLeafSize', minLeafSize);
    model = fitcensemble(trainFeatures,'Activity', ...
        'Method','AdaBoostM2', ...
        'Learners',tTree, ...
        'NumLearningCycles', numTrees);
predictions = predict(model, testFeatures);
predictions = categorical(predictions);
accuracy = sum(predictions == testTargets)/numel(testTargets);
fprintf("Accuracy: %.2f\n", accuracy);
figure
cm = confusionchart(testTargets, predictions);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
title('Test Confusion Matrix for RF');
xlabel('Predicted Activity');
ylabel('True Activity');

% Complete Confusion Matrix
figure
subplot(2, 2, 1);
trainPred = predict(model, trainFeatures);
trainCm = confusionchart(trainTargets, trainPred);
% trainCm.ColumnSummary = 'column-normalized';
% trainCm.RowSummary = 'row-normalized';
title('Train Confusion Matrix for RF');
xlabel('Predicted Activity');
ylabel('True Activity');

subplot(2, 2, 2);
valPred = predict(model, valFeatures);
valCm = confusionchart(valTargets, valPred);
% valCm.ColumnSummary = 'column-normalized';
% valCm.RowSummary = 'row-normalized';
title('Validation Confusion Matrix for RF');
xlabel('Predicted Activity');
ylabel('True Activity');

subplot(2, 2, 3);
testCm = confusionchart(testTargets, predictions);
% testCm.ColumnSummary = 'column-normalized';
% testCm.RowSummary = 'row-normalized';
title('Test Confusion Matrix for RF');
xlabel('Predicted Activity');
ylabel('True Activity');

subplot(2, 2, 4);
totalTargets = [trainTargets; valTargets; testTargets];
totalPredictions = [trainPred; valPred; predictions];
totalCm = confusionchart(totalTargets, totalPredictions);
% totalCm.ColumnSummary = 'column-normalized';
% totalCm.RowSummary = 'row-normalized';
title('Total Confusion Matrix for RF');
xlabel('Predicted Activity');
ylabel('True Activity');
%% Classification Metrics
% Display the hyperparameters
fprintf('=== Hyperparameters ===\n');
fprintf('Number of Trees: %d\n', numTrees);
fprintf('Max Number of Splits: %d\n', maxNumSplits);
fprintf('Min Leaf Size: %d\n', minLeafSize);

% Display the classification error
fprintf('\n=== Classification Metrics ===\n');
fprintf('Accuracy: %.4f\n', accuracy);
fprintf('Classification error: %.4f\n', 1/accuracy);
