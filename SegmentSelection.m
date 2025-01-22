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

%% Load the Data
load preprocessed_with_features.mat

% We need to divide the data into body parts: Shank_R, Foot_L, Pelvis, etc.
[segmentStruct, allFeatures] = splitSegments(dataStruct);

function [segmentStruct, allFeatures] = splitSegments(dataStruct)
    columns = extractLocationIdx(dataStruct);
    allFeatures = consolidateFeatures(dataStruct);
    [trainX, trainY, testX, testY] = trainTestSplit(allFeatures);
    segmentStruct = splitFeaturesBySegment(trainX, trainY, testX, ...
        testY, columns);
end

function columns = extractLocationIdx(dataStruct)
    columns = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for i = 1:length(dataStruct)
        data = dataStruct(i).Features;
        varNames = data.Properties.VariableNames;
        for c = 1:length(varNames)
            colName = varNames{c};
            tokens = regexp(colName, ...
                '^(?<bp>(Thigh|Shank|Foot|Pelvis))(?:_?)(?<side>[LR])?_(?<num>\d+)', ...
                'names');
            if ~isempty(tokens)
                if ~isempty(tokens.side)
                    location = sprintf('%s_%s', tokens.bp, tokens.side);
                else
                    location = tokens.bp;
                end
                if isKey(columns, location)
                    columns(location) = [columns(location) c];
                else
                    columns(location) = c;
                end
            end
        end
    end
    % this get's the columns for each location for each entry into
    % dataStruct, this will be 60*63 - we need only the unique elements
    % which if the data is organised correctly, should be 63 wide
    keys = columns.keys;
    for j = 1:length(keys)
        key = keys{j};
        columns(key) = unique(columns(key));
        assert(length(columns(key)) == 63);
    end
end

function allFeatures = consolidateFeatures(dataStruct)
    allFeatures = [];
    for i = 1:length(dataStruct)
        dataStruct(i).Features.Activity = repmat({dataStruct(i).Activity}, size(dataStruct(i).Features, 1), 1);
        dataStruct(i).Features.Subject = repmat({dataStruct(i).Subject}, size(dataStruct(i).Features, 1), 1);
    end

    % Concatenate all the features into a single table
    for i = 1:length(dataStruct)
        features = dataStruct(i).Features;
        allFeatures = [allFeatures; features];
    end
end

function [trainX, trainY, testX, testY] = trainTestSplit(allFeatures, testRatio)
    allFeatures.Subject = grp2idx(allFeatures.Subject);
    uniqueSubjects = unique(allFeatures.Subject);
    subject = randsample(uniqueSubjects, 1);
    testIdx = allFeatures.Subject == subject;
    trainIdx = ~testIdx;

    allFeatures.Subject = [];
    allTargets = allFeatures.Activity;
    allFeatures.Activity = [];

    trainX = allFeatures(trainIdx, :);
    testX = allFeatures(testIdx, :);

    trainY = allTargets(trainIdx, :);
    testY = allTargets(testIdx, :);
end

function segmentStruct = splitFeaturesBySegment(trainX, trainY, testX, testY, columns)
    segmentStruct = struct();
    locations = columns.keys;
    for i = 1:length(locations)
        location = locations{i};
        cols = columns(location);
        idx = min(cols):max(cols);
        locTrainX = trainX(:, idx);
        locTestX = testX(:, idx);
        segmentStruct(i).Location = location;
        segmentStruct(i).trainX = locTrainX;
        segmentStruct(i).trainY = trainY;
        segmentStruct(i).testX = locTestX;
        segmentStruct(i).testY = testY;
    end
end

classNames = unique(allFeatures.Activity)';

save('segmentData.mat', 'dataStruct', 'segmentStruct', ...
    'allFeatures', "classNames");
%% Find Significant Features
% We will train a simple ANN model for each segment
% ANNs accept features as rows and samples as columns so we need to transpose
segmentResults = struct();

nTrials = 100;

for i = 1:length(segmentStruct)
    location = segmentStruct(i).Location;
    trainX = table2array(segmentStruct(i).trainX)';
    trainY = onehotencode(categorical(segmentStruct(i).trainY), 2)';
    testX = table2array(segmentStruct(i).testX)';
    testY = onehotencode(categorical(segmentStruct(i).testY), 2)';

    accuracyResults = zeros(nTrials, 1);
    confusionResults = cell(nTrials, 1);

    parfor j = 1:nTrials
        net = patternnet(10);
        net.divideParam.trainRatio = 85/100;
        net.divideParam.valRatio = 15/100;
        net.divideParam.testRatio = 0/100;
        net.trainParam.showWindow = 0;
        [net, tr] = train(net, trainX, trainY);

        % test the neural network
        predictedTargets = net(testX);
        % view(net);
        [c, cm] = confusion(testY, predictedTargets);
        accuracy = sum(diag(cm))/sum(cm(:));
        accuracyResults(j) = accuracy;
        confusionResults{j} = cm;
    end

    segmentResults(i).Location = location;
    segmentResults(i).Accuracy = accuracyResults;
    segmentResults(i).Confusion = confusionResults;
    segmentResults(i).MeanAccuracy = mean(accuracyResults);
    segmentResults(i).StdAccuracy = std(accuracyResults);
    segmentResults(i).MeanConfusion = mean(cat(3, confusionResults{:}), 3);
end

save('segmentResults.mat', 'segmentResults');

fprintf('Segment Results\n');
fprintf('================\n');
fprintf('Location\t\tMean Accuracy\t\tStd Accuracy\n');
for i = 1:length(segmentResults)
    fprintf('%s\t\t\t%.4f\t\t\t\t%.4f\n', segmentResults(i).Location, ...
        segmentResults(i).MeanAccuracy, segmentResults(i).StdAccuracy);
end
%% Training ANN and SVM
load segmentData.mat
load segmentResults.mat

% Get the segment with the highest mean accuracy
meanAccuracies = [segmentResults.MeanAccuracy];
[~, idx] = max(meanAccuracies);
location = segmentResults(idx).Location;

% In the report we also identify the Pelvis as having a high significance
location_2 = "Pelvis";

% Get the data for the selected segment
for i = 1:length(segmentStruct)
    if strcmp(segmentStruct(i).Location, location)
        trainX = segmentStruct(i).trainX;
        trainY = segmentStruct(i).trainY;
        testX = segmentStruct(i).testX;
        testY = segmentStruct(i).testY;
    end
    if strcmp(segmentStruct(i).Location, location_2)
        trainX_2 = segmentStruct(i).trainX;
        trainY_2 = segmentStruct(i).trainY;
        testX_2 = segmentStruct(i).testX;
        testY_2 = segmentStruct(i).testY;
    end
end
%%
% Hyperparameter Tuning for the ANN
layer1 = 10:10:100;
layer2 = 10:10:100;
grid = [layer1; zeros(length(layer1), 1)'];
grid = [grid, combvec(layer1, layer2)];
gridSize = size(grid, 2);
ann_hyperparameters = struct();

X = table2array(trainX)';
Y = onehotencode(categorical(trainY), 2)';
tX = table2array(testX)';
tY = onehotencode(categorical(testY), 2)';

fprintf('Starting Grid Search\n');
fprintf('Grid size: %d\n', gridSize);
parfor i = 1:gridSize
    layers = grid(:, i)';
    if layers(2) == 0
        layers = layers(1);
    end
    [net, ~] = trainNet(X, Y, layers);
    [accuracy, cm] = testNet(net, tX, tY);
    ann_hyperparameters(i).Layers = layers;
    ann_hyperparameters(i).Accuracy = accuracy;
    ann_hyperparameters(i).Net = net;
    ann_hyperparameters(i).Confusion = cm;
end

function [net, tr] = trainNet(trainFeatures, trainTargets, layers)
    net = patternnet(layers);
    net.divideParam.trainRatio = 85/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 0/100;

    net.trainParam.showWindow = 0;
    [net, tr] = train(net, trainFeatures, trainTargets);
end

function [accuracy, cm] = testNet(net, testFeatures, testTargets)
    predictedTargets = net(testFeatures);
    [~, cm] = confusion(testTargets, predictedTargets);
    accuracy = sum(diag(cm))/sum(cm(:));
    fprintf('Accuracy: %f\n', accuracy);
end

save("ann_segment_hyperparameters.mat", "ann_hyperparameters");
%%
load ann_segment_hyperparameters.mat

% Get the best hyperparameters
accuracies = [ann_hyperparameters.Accuracy];
[~, idx] = max(accuracies);
bestNet = ann_hyperparameters(idx).Net;
bestLayers = ann_hyperparameters(idx).Layers;
fprintf('Best Hyperparameters\n');
fprintf('Layers: %s\n', mat2str(bestLayers));
fprintf('Accuracy: %.4f\n', accuracies(idx));

% Test the best ANN
X = table2array(testX)';
Y = onehotencode(categorical(testY), 2)';
predictedTargets = bestNet(X);
[~, cm] = confusion(Y, predictedTargets);
accuracy = sum(diag(cm))/sum(cm(:));
fprintf('Retrained ANN Accuracy: %.4f\n', accuracy);
figure
cm = confusionchart(cm, classNames');
cm.Title = strcat("ANN Confusion Matrix for ", location);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
xlabel('Predicted Activity');
ylabel('True Activity');
%%
% Hyperparameter Tuning for the SVM
svm_hyperparameters = struct();

boxConstraint = 1e-3:100:1e3;
kernelFunction = ["gaussian", "polynomial"];
polynomialOrder = 1:3;
multiclassCoding = ["onevsone", "onevsall"];
standardiseData = ["on", "off"];

grid = [];
for i = 1:length(boxConstraint)
    for j = 1:length(kernelFunction)
        if strcmp(kernelFunction(j), "polynomial")
            for k = 1:length(polynomialOrder)
                for l = 1:length(multiclassCoding)
                    for m = 1:length(standardiseData)
                        grid = [grid; boxConstraint(i), kernelFunction(j), ...
                            polynomialOrder(k), multiclassCoding(l), standardiseData(m)];
                    end
                end
            end
        else
            for l = 1:length(multiclassCoding)
                for m = 1:length(standardiseData)
                    grid = [grid; boxConstraint(i), kernelFunction(j), ...
                        NaN, multiclassCoding(l), standardiseData(m)];
                end
            end
        end
    end
end
%%
gridSize = size(grid, 1);
svm_hyperparameters = struct();

fprintf('Starting Grid Search\n');
parfor idx = 1:gridSize
    row = grid(idx, :);
    boxConstraint = double(row(1));
    kernelFunction = string(row(2));
    polynomialOrder = double(row(3));
    multiclassCoding = string(row(4));
    standardiseData = string(row(5));

    if isnan(polynomialOrder)
        polynomialOrder = [];
    end

    t = templateSVM('BoxConstraint', boxConstraint, ...
        'KernelFunction', kernelFunction, ...
        'PolynomialOrder', polynomialOrder, ...
        'Standardize', standardiseData, ...
        'KernelScale', 'auto');

    model = fitcecoc(trainX, trainY, 'Learners', t, ...
        'Coding', multiclassCoding, 'ClassNames', classNames);

    predictedTargets = predict(model, testX);
    cm = confusionmat(testY, predictedTargets);
    accuracy = sum(diag(cm))/sum(cm(:));

    svm_hyperparameters(idx).BoxConstraint = boxConstraint;
    svm_hyperparameters(idx).KernelFunction = kernelFunction;
    svm_hyperparameters(idx).PolynomialOrder = polynomialOrder;
    svm_hyperparameters(idx).MulticlassCoding = multiclassCoding;
    svm_hyperparameters(idx).StandardiseData = standardiseData;
    svm_hyperparameters(idx).Model = model;
    svm_hyperparameters(idx).Confusion = cm;
    svm_hyperparameters(idx).Accuracy = accuracy;
end

save("svm_segment_hyperparameters.mat", "svm_hyperparameters");
%%
load svm_segment_hyperparameters.mat

% Get the best hyperparameters
accuracies = [svm_hyperparameters.Accuracy];
[~, idx] = max(accuracies);
bestModel = svm_hyperparameters(idx).Model;
bestBoxConstraint = svm_hyperparameters(idx).BoxConstraint;
bestKernelFunction = svm_hyperparameters(idx).KernelFunction;
bestPolynomialOrder = svm_hyperparameters(idx).PolynomialOrder;
bestMulticlassCoding = svm_hyperparameters(idx).MulticlassCoding;

fprintf('Best Hyperparameters\n');
fprintf('Box Constraint: %.4f\n', bestBoxConstraint);
fprintf('Kernel Function: %s\n', bestKernelFunction);
fprintf('Polynomial Order: %d\n', bestPolynomialOrder);
fprintf('Multiclass Coding: %s\n', bestMulticlassCoding);
fprintf('Accuracy: %.4f\n', accuracies(idx));

% Test the best SVM
predictedTargets = predict(bestModel, testX);
cm = confusionmat(testY, predictedTargets);
accuracy = sum(diag(cm))/sum(cm(:));
fprintf('Retrained SVM Accuracy: %.4f\n', accuracy);
figure
cm = confusionchart(cm, classNames');
cm.Title = strcat("SVM Confusion Matrix for ", location);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
xlabel('Predicted Activity');
ylabel('True Activity');
%% With Shank and Pelvis Segments combined
X = [trainX, trainX_2];
Y = trainY;
tX = [testX, testX_2];
tY = testY;

X = table2array(X)';
Y = onehotencode(categorical(Y), 2)';
tX = table2array(tX)';
tY = onehotencode(categorical(tY), 2)';

% Hyperparameter Tuning for the ANN
layer1 = 10:10:100;
layer2 = 10:10:100;
grid = [layer1; zeros(length(layer1), 1)'];
grid = [grid, combvec(layer1, layer2)];
gridSize = size(grid, 2);
combined_ann_hyperparameters = struct();

fprintf('Starting Grid Search\n');
fprintf('Grid size: %d\n', gridSize);
parfor i = 1:gridSize
    layers = grid(:, i)';
    if layers(2) == 0
        layers = layers(1);
    end
    [net, ~] = trainNet(X, Y, layers);
    [accuracy, cm] = testNet(net, tX, tY);
    combined_ann_hyperparameters(i).Layers = layers;
    combined_ann_hyperparameters(i).Accuracy = accuracy;
    combined_ann_hyperparameters(i).Net = net;
    combined_ann_hyperparameters(i).Confusion = cm;
end

save("combined_ann_segment_hyperparameters.mat", "combined_ann_hyperparameters");
%%
load combined_ann_segment_hyperparameters.mat

% Get the best hyperparameters
accuracies = [combined_ann_hyperparameters.Accuracy];
[~, idx] = max(accuracies);
bestNet = combined_ann_hyperparameters(idx).Net;
bestLayers = combined_ann_hyperparameters(idx).Layers;
fprintf('Best Hyperparameters\n');
fprintf('Layers: %s\n', mat2str(bestLayers));
fprintf('Accuracy: %.4f\n', accuracies(idx));

% Test the best ANN
predictedTargets = bestNet(tX);
[~, cm] = confusion(tY, predictedTargets);
accuracy = sum(diag(cm))/sum(cm(:));
fprintf('Retrained ANN Accuracy: %.4f\n', accuracy);
fprintf('Retrained ANN Classification Error: %.4f\n', 1 - accuracy);
figure
cm = confusionchart(cm, classNames');
cm.Title = "ANN Confusion Matrix for Shank and Pelvis";
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
xlabel('Predicted Activity');
ylabel('True Activity');
%%
% Hyperparameter Tuning for the SVM
boxConstraint = 1e-3:100:1e3;
kernelFunction = ["gaussian", "polynomial"];
polynomialOrder = 1:3;
multiclassCoding = ["onevsone", "onevsall"];
standardiseData = ["on", "off"];

grid = [];

for i = 1:length(boxConstraint)
    for j = 1:length(kernelFunction)
        if strcmp(kernelFunction(j), "polynomial")
            for k = 1:length(polynomialOrder)
                for l = 1:length(multiclassCoding)
                    for m = 1:length(standardiseData)
                        grid = [grid; boxConstraint(i), kernelFunction(j), ...
                            polynomialOrder(k), multiclassCoding(l), standardiseData(m)];
                    end
                end
            end
        else
            for l = 1:length(multiclassCoding)
                for m = 1:length(standardiseData)
                    grid = [grid; boxConstraint(i), kernelFunction(j), ...
                        NaN, multiclassCoding(l), standardiseData(m)];
                end
            end
        end
    end
end
%%
gridSize = size(grid, 1);
svm_hyperparameters = struct();

fprintf('Starting Grid Search\n');
parfor idx = 1:gridSize
    row = grid(idx, :);
    boxConstraint = double(row(1));
    kernelFunction = string(row(2));
    polynomialOrder = double(row(3));
    multiclassCoding = string(row(4));
    standardiseData = string(row(5));

    if isnan(polynomialOrder)
        polynomialOrder = [];
    end

    t = templateSVM('BoxConstraint', boxConstraint, ...
        'KernelFunction', kernelFunction, ...
        'PolynomialOrder', polynomialOrder, ...
        'Standardize', standardiseData, ...
        'KernelScale', 'auto');

    model = fitcecoc(X, Y, 'Learners', t, ...
        'Coding', multiclassCoding, 'ClassNames', classNames);

    predictedTargets = predict(model, tX);
    cm = confusionmat(tY, predictedTargets);
    accuracy = sum(diag(cm))/sum(cm(:));

    svm_hyperparameters(idx).BoxConstraint = boxConstraint;
    svm_hyperparameters(idx).KernelFunction = kernelFunction;
    svm_hyperparameters(idx).PolynomialOrder = polynomialOrder;
    svm_hyperparameters(idx).MulticlassCoding = multiclassCoding;
    svm_hyperparameters(idx).StandardiseData = standardiseData;
    svm_hyperparameters(idx).Model = model;
    svm_hyperparameters(idx).Confusion = cm;
    svm_hyperparameters(idx).Accuracy = accuracy;
end

save("combined_svm_segment_hyperparameters.mat", "svm_hyperparameters");
%%
load combined_svm_segment_hyperparameters.mat

% Get the best hyperparameters
accuracies = [svm_hyperparameters.Accuracy];
[~, idx] = max(accuracies);
bestModel = svm_hyperparameters(idx).Model;
bestBoxConstraint = svm_hyperparameters(idx).BoxConstraint;
bestKernelFunction = svm_hyperparameters(idx).KernelFunction;
bestPolynomialOrder = svm_hyperparameters(idx).PolynomialOrder;
bestMulticlassCoding = svm_hyperparameters(idx).MulticlassCoding;

fprintf('Best Hyperparameters\n');
fprintf('Box Constraint: %.4f\n', bestBoxConstraint);
fprintf('Kernel Function: %s\n', bestKernelFunction);
fprintf('Polynomial Order: %d\n', bestPolynomialOrder);
fprintf('Multiclass Coding: %s\n', bestMulticlassCoding);
fprintf('Accuracy: %.4f\n', accuracies(idx));

% Test the best SVM
predictedTargets = predict(bestModel, tX);
cm = confusionmat(tY, predictedTargets);
accuracy = sum(diag(cm))/sum(cm(:));
fprintf('Retrained SVM Accuracy: %.4f\n', accuracy);
fprintf('Retrained SVM Classification Error: %.4f\n', 1 - accuracy);
figure
cm = confusionchart(cm, classNames');
cm.Title = "SVM Confusion Matrix for Shank and Pelvis";
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
xlabel('Predicted Activity');
ylabel('True Activity');
