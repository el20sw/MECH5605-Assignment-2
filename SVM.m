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
%%
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

% convert allFeatures into a matrix - for an SVM, columns are features and rows are samples
allFeatures = table2array(allFeatures);
allFeatures(:, end) = [];

trainFeatures = allFeatures(trainIdx, 1:end);
trainTargets = allTargets(trainIdx, :);
trainTargets = categorical(trainTargets);

extraTrainFeatures = trainFeatures;
extraTrainTargets = trainTargets;

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
valSplit = 0.2;
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
%%
% Grid search for SVM hyperparameter tuning - we care about the box constraint,
% kernel function (and then polynomial order)

% Define the hyperparameters to search over
boxConstraint = linspace(1e-3, 1e3, 11);
kernelFunction = ["linear", "polynomial", "gaussian"];
polynomialOrder = [2, 3];
multiclassCoding = {'onevsone', 'onevsall'};
standardiseData = ["on", "off"];

% make note of the hyperparameters and the accuaracy they produce
hyperparameters = [];
gridSize = length(standardiseData)*length(kernelFunction)* ...
    length(polynomialOrder)*length(multiclassCoding)*length(boxConstraint);
count = 0;

% create the grid to search over
grid = [];
for i = 1:length(standardiseData)
    for j = 1:length(kernelFunction)
        if kernelFunction(j) == "polynomial"
            for k = 1:length(polynomialOrder)
                for l = 1:length(multiclassCoding)
                    for m = 1:length(boxConstraint)
                        grid = [grid; [standardiseData(i), kernelFunction(j), polynomialOrder(k), multiclassCoding(l), boxConstraint(m)]];
                    end
                end
            end
        else
            for l = 1:length(multiclassCoding)
                for m = 1:length(boxConstraint)
                    grid = [grid; [standardiseData(i), kernelFunction(j), NaN, multiclassCoding(l), boxConstraint(m)]];
                end
            end
        end
    end
end
%%
gridSize = size(grid, 1);
hyperparameters = table('Size', [gridSize, 6], ...
    'VariableTypes', {'double', 'string', 'double', 'string', 'string', 'double'}, ...
    'VariableNames', {'BoxConstraint', 'KernelFunction', 'PolynomialOrder', 'Coding', 'StandardiseData', 'Accuracy'});

fprintf('Starting grid search\n');
parfor idx = 1:gridSize
    row = grid(idx, :);
    standardiseData = string(row(1));
    kernelFunction = string(row(2));
    order = double(row(3));
    coding = string(row(4));
    box = double(row(5));

    if kernelFunction == "polynomial"
        model = createTemplateSVM(box, kernelFunction, order, standardiseData);
    else
        model = createTemplateSVM(box, kernelFunction, [], standardiseData);
    end

    svm = fitcecoc(trainFeatures, 'Activity', ...
        'Learners', model, ...
        'Coding', coding, ...
        'ClassNames', classNames);
    predictions = predict(svm, valFeatures);
    predictions = categorical(predictions);
    accuracy = sum(predictions == valTargets)/length(valTargets);
    params = {box, kernelFunction, order, coding, standardiseData, accuracy};
    hyperparameters(idx, :) = params;

    fprintf('Accuracy: %.2f\n', accuracy);
end

function model = createTemplateSVM(box, kernelFunction, order, standardiseData)
    model = templateSVM(...
        'KernelFunction', kernelFunction, ...
        'PolynomialOrder', order, ...
        'KernelScale', 'auto', ...
        'BoxConstraint', box, ...
        'Standardize', standardiseData);
end

save('svm_hyperparameters.mat', 'hyperparameters');
%% Further hyperparameter tuning
opts = struct(...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'ShowPlots', true, ...
    'Verbose', 1, ...
    'UseParallel', true);

% Train SVM with automatic hyperparameter optimization
[model, optimisationResults] = fitcecoc(trainFeatures, trainTargets, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', opts);

%%
% Find the best hyperparameters
load svm_hyperparameters.mat

% get the rows with the highest accuracy
accuracies = hyperparameters.Accuracy;
[~, idx] = max(accuracies);
bestHyperparameters = hyperparameters(idx, :);
box = bestHyperparameters.BoxConstraint;
kernelFunction = bestHyperparameters.KernelFunction;
% order may be NaN, so we need to check
if isnan(double(bestHyperparameters.PolynomialOrder))
    order = [];
else
    order = bestHyperparameters.PolynomialOrder;
end
coding = bestHyperparameters.Coding;
standardiseData = bestHyperparameters.StandardiseData;

model = createTemplateSVM(box, kernelFunction, order, standardiseData);
svm = fitcecoc(trainFeatures, 'Activity', ...
    'Learners', model, ...
    'Coding', coding, ...
    'ClassNames', classNames);
predictions = predict(svm, testFeatures);
predictions = categorical(predictions);
accuracy = sum(predictions == testTargets)/length(testTargets);
fprintf("Accuracy: %.3f\n", accuracy);

figure
cm = confusionchart(testTargets, predictions);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
cm.Title = 'Confusion Matrix for SVM';
xlabel('Predicted Activity');
ylabel('True Activity');

disp(svm.CodingMatrix)

trainPred = predict(svm, trainFeatures);
trainPred = categorical(trainPred);
valPred = predict(svm, valFeatures);
valPred = categorical(valPred);

% Train and Validation Confusion
figure
subplot(1, 2, 1);
trainCM = confusionchart(trainTargets, trainPred);
trainCM.RowSummary = 'row-normalized';
trainCM.Title = 'Training Confusion Matrix for SVM';
xlabel('Predicted Activity');
ylabel('True Activity');

subplot(1, 2, 2);
valCM = confusionchart(valTargets, valPred);
valCM.RowSummary = 'row-normalized';
valCM.Title = 'Validation Confusion Matrix for SVM';
xlabel('Predicted Activity');
ylabel('True Activity');

% Complete Confusion Matrix
figure
% Training Confusion Matrix
subplot(2, 2, 1);
trainAccuracy = sum(trainPred == trainTargets)/length(trainTargets);
trainCM = confusionchart(trainTargets, trainPred);
trainCM.RowSummary = 'row-normalized';
trainCM.ColumnSummary = 'column-normalized';
trainCM.Title = 'Training Confusion Matrix for SVM';
xlabel('Predicted Activity');
ylabel('True Activity');

% Validation Confusion Matrix
subplot(2, 2, 2);
valAccuracy = sum(valPred == valTargets)/length(valTargets);
valCM = confusionchart(valTargets, valPred);
valCM.RowSummary = 'row-normalized';
valCM.ColumnSummary = 'column-normalized';
valCM.Title = 'Validation Confusion Matrix for SVM';
xlabel('Predicted Activity');
ylabel('True Activity');

% Test Confusion Matrix
subplot(2, 2, 3);
cm = confusionchart(testTargets, predictions);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
cm.Title = 'Test Confusion Matrix for SVM';
xlabel('Predicted Activity');
ylabel('True Activity');

% Total Confusion Matrix
subplot(2, 2, 4);
totalTargets = [trainTargets; valTargets; testTargets];
totalPredictions = [trainPred; valPred; predictions];
totalCM = confusionchart(totalTargets, totalPredictions);
totalCM.RowSummary = 'row-normalized';
totalCM.ColumnSummary = 'column-normalized';
totalCM.Title = 'Total Confusion Matrix for SVM';
xlabel('Predicted Activity');
ylabel('True Activity');

% Display the best hyperparameters
fprintf('\n=== Best Hyperparameters ===\n');
fprintf('Box Constraint: %.4f\n', box);
fprintf('Kernel Function: %s\n', kernelFunction);
if ~isempty(order)
    fprintf('Polynomial Order: %d\n', order);
end
fprintf('Coding: %s\n', coding);
fprintf('Standardise Data: %s\n', standardiseData);
fprintf('Accuracy: %.4f\n', accuracy * 100);
fprintf('Best accuracy: %.4f\n', max(hyperparameters.Accuracy) * 100);
fprintf('Grid size: %d\n', gridSize);

% Display the classification error
fprintf('\n=== Classification Metrics ===\n');
fprintf('Classification error: %.4f\n', 1/accuracy);
%% From Further Hyperparameter Tuning
box = 950.68;
kernelFunction = "linear";
kernelScale = 262.9;
coding = "onevsone";
standardiseData = "on";

model = templateSVM(...
        'KernelFunction', kernelFunction, ...
        'PolynomialOrder', [], ...
        'KernelScale', kernelScale, ...
        'BoxConstraint', box, ...
        'Standardize', standardiseData);
svm = fitcecoc(trainFeatures, 'Activity', ...
    'Learners', model, ...
    'Coding', coding, ...
    'ClassNames', classNames);
predictions = predict(svm, testFeatures);
predictions = categorical(predictions);
accuracy = sum(predictions == testTargets)/length(testTargets);
fprintf("Accuracy: %.3f\n", accuracy);
figure
cm = confusionchart(testTargets, predictions);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
cm.Title = 'Confusion Matrix for SVM';
xlabel('Predicted Activity');
ylabel('True Activity');

disp(svm.CodingMatrix)

% Display the best hyperparameters
fprintf('\n=== Best Hyperparameters ===\n');
fprintf('Box Constraint: %.4f\n', box);
fprintf('Kernel Function: %s\n', kernelFunction);
if ~isempty(order)
    fprintf('Polynomial Order: %d\n', order);
end
fprintf('Coding: %s\n', coding);
fprintf('Standardise Data: %s\n', standardiseData);
fprintf('Accuracy: %.4f\n', accuracy * 100);
fprintf('Best accuracy: %.4f\n', max(hyperparameters.Accuracy) * 100);
fprintf('Grid size: %d\n', gridSize);

% Display the classification error
fprintf('\n=== Classification Metrics ===\n');
fprintf('Classification error: %.4f\n', 1/accuracy);
