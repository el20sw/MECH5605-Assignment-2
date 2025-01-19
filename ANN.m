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
allFeatures.Subject = grp2idx(allFeatures.Subject);
subject = randsample(uniqueSubjects, 1);
testIdx = allFeatures.Subject == subject;
trainIdx = ~testIdx;
classNames = unique(allFeatures.Activity);

% extract the targets and remove from the features
allTargets = allFeatures.Activity;
allFeatures.Activity = [];

% get the heading names, sans 'Subject' and 'Activity'
headingNames = allFeatures.Properties.VariableNames;
headingNames = headingNames(~ismember(headingNames, {'Subject', 'Activity'}));

% convert allFeatures into a matrix - for an ANN, rows are features and columns are samples
allFeatures = table2array(allFeatures);
allFeatures(:, end) = [];

trainFeatures = allFeatures(trainIdx, 1:end)';
trainTargets = allTargets(trainIdx, :);
trainTargets = onehotencode(categorical(trainTargets), 2)';

testFeatures = allFeatures(testIdx, 1:end)';
testTargets = allTargets(testIdx, :);
testTargets = onehotencode(categorical(testTargets), 2)';
%%
% Baseline model - sanity check
net = patternnet(10);
net.divideParam.trainRatio = 85/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 0/100;
net.trainParam.showWindow = 0;
[net, tr] = train(net, trainFeatures, trainTargets);

% test the neural network
predictedTargets = net(testFeatures);
view(net);
figure
[c, cm] = confusion(testTargets, predictedTargets);
accuracy = sum(diag(cm))/sum(cm(:));
plotconfusion(testTargets, predictedTargets);
figure
plotperform(tr);
disp(accuracy);
%%
% Define hyperparameters to search over
L1 = 1:5:50;    % hidden layer 1 size
L2 = 1:5:50;    % hidden layer 2 size
L3 = 1:5:50;    % hidden layer 3 size

gridSize = length(L1) + length(L1)*length(L2) + ...
    length(L1)*length(L2)*length(L3);

hyperparameters = [];

% grid search for hyperparameters
disp('Starting grid search for hyperparameters');
disp(['Grid size: ', num2str(length(L1))]);
hyperparametersResults = zeros(length(L1), 5);
parfor idx = 1:length(L1)
    layers = [L1(idx)];
    [net, tr] = trainNet(trainFeatures, trainTargets, layers);
    accuracy = testNet(net, testFeatures, testTargets);
    hyperparametersResults(idx, :) = [L1(idx), NaN, NaN, 1, accuracy];
end
hyperparameters = [hyperparameters; hyperparametersResults];
disp('Finished 1 layer');

disp('Starting 2 layer');
combosTwo = length(L1)*length(L2);
disp(['Grid size: ', num2str(combosTwo)]);
hyperparametersResults = zeros(combosTwo, 5);
parfor idx = 1:combosTwo
    [i, j] = ind2sub([length(L1),length(L2)], idx);
    layers = [L1(i),L2(j)];
    [net, tr] = trainNet(trainFeatures, trainTargets, layers);
    accuracy = testNet(net, testFeatures, testTargets);
    hyperparametersResults(idx, :) = [L1(i), L2(j), NaN, 2, accuracy];
end
hyperparameters = [hyperparameters; hyperparametersResults];
disp('Finished 2 layer');

disp('Starting 3 layer');
combosThree = length(L1)*length(L2)*length(L3);
disp(['Grid size: ', num2str(combosThree)]);
hyperparametersResults = zeros(combosThree, 5);
parfor idx = 1:combosThree
    [i, j, k] = ind2sub([length(L1), ...
        length(L2),length(L3)], idx);
    layers = [L1(i),L2(j),L3(k)];
    [net, tr] = trainNet(trainFeatures, trainTargets, layers);
    accuracy = testNet(net, testFeatures, testTargets);
    hyperparametersResults(idx, :) = [L1(i), L2(j), L3(k), 3, accuracy];
end
hyperparameters = [hyperparameters; hyperparametersResults];
disp('Finished 3 layer');

function [net, tr] = trainNet(trainFeatures, trainTargets, layers)
    net = patternnet(layers);
    net.divideParam.trainRatio = 85/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 0/100;

    net.trainParam.showWindow = 0;
    [net, tr] = train(net, trainFeatures, trainTargets);
end

function accuracy = testNet(net, testFeatures, testTargets)
    predictedTargets = net(testFeatures);
    testPerformance = perform(net, testTargets, predictedTargets);
    [c, cm] = confusion(testTargets, predictedTargets);
    accuracy = sum(diag(cm))/sum(cm(:));
    fprintf('Accuracy: %f\n', accuracy);
end

save('ann_hyperparameters.mat', 'hyperparameters');
%%
% Find the best hyperparameters
load ann_hyperparameters.mat

accuracies = hyperparameters(:, end);
[~, idx] = max(accuracies);
bestHyperparameters = hyperparameters(idx, :);
L1 = bestHyperparameters(1);
L2 = bestHyperparameters(2);
L3 = bestHyperparameters(3);
LD = bestHyperparameters(4);
bestAccuracy = bestHyperparameters(5);

if isnan(L2)
    layers = L1;
elseif isnan(L3)
    layers = [L1, L2];
else
    layers = [L1, L2, L3];
end

net = patternnet(layers);
net.divideParam.trainRatio = 85/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 0/100;
net.trainParam.showWindow = 0;
[net, tr] = train(net, trainFeatures, trainTargets);

% test the neural network
predictedTargets = net(testFeatures);
[c, cm] = confusion(testTargets, predictedTargets);
accuracy = sum(diag(cm))/sum(cm(:));
plotconfusion(testTargets, predictedTargets);
figure
plotperform(tr);
disp(accuracy);
