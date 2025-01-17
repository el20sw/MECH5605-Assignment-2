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
% Baseline test model
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
hiddenLayer1Size = 1:50:500;
hiddenLayer2Size = 1:50:500;
hiddenLayerDepth = [1, 2];

hyperparameters = [];
bestAccuracy = 0;

gridSize = length(hiddenLayer1Size) + length(hiddenLayer1Size) * length(hiddenLayer2Size);
count = 0;

for i = 1:length(hiddenLayerDepth)
    depth = hiddenLayerDepth(i);
    for j = 1:length(hiddenLayer1Size)
        if depth == 2
            for k = 1:length(hiddenLayer2Size)
                count = count + 1;
                disp(['Training network ', num2str(count), ' of ', num2str(gridSize)]);
                disp(['Hidden layer 1 size: ', num2str(hiddenLayer1Size(j)), ', Hidden layer 2 size: ', num2str(hiddenLayer2Size(k))]);
                layers = [hiddenLayer1Size(j), hiddenLayer2Size(k)];
                [net, tr] = trainnet(trainFeatures, trainTargets, layers);

                % test the neural network
                accuracy = testNet(net, testFeatures, testTargets);
                if accuracy > bestAccuracy
                    bestAccuracy = accuracy;
                end
                hyperparameters = [hyperparameters; hiddenLayer1Size(j), hiddenLayer2Size(k), depth, accuracy];
            end
        else
            count = count + 1;
            disp(['Training network ', num2str(count), ' of ', num2str(gridSize)]);
            disp(['Hidden layer size: ', num2str(hiddenLayer1Size(j))]);
            layers = [hiddenLayer1Size(j)];
            [net, tr] = trainnet(trainFeatures, trainTargets, layers);

            % test the neural network
            accuracy = testNet(net, testFeatures, testTargets);
            if accuracy > bestAccuracy
                bestAccuracy = accuracy;
            end
            hyperparameters = [hyperparameters; hiddenLayer1Size(j), NaN, depth, accuracy];
        end
    end
end

function [net, tr] = trainnet(trainFeatures, trainTargets, layers)
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
% accuracies = str2double(accuracies);
[~, idx] = max(accuracies);

bestHyperparameters = hyperparameters(idx, :);
hiddenLayer1Size = bestHyperparameters(1);
hiddenLayer2Size = bestHyperparameters(2);
hiddenLayerDepth = bestHyperparameters(3);

layers = [hiddenLayer1Size, hiddenLayer2Size];
if isnan(hiddenLayer2Size)
    layers = hiddenLayer1Size;
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
