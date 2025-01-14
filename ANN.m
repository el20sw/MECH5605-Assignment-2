load preprocessed_with_features.mat

% ToDo:
% - get average accuracy from leaving out each subject
% - note that this requires modification for leaving out each subject
% - do some feature selection by randomly sampling from features and
% measuring accuracy?
% - find the 15 most useful features and report on these for the ANN

allFeatures = [];

% what do we need to do?
% for each feature row in each data file, it need the activity to be appended to the features and the person needs to be known
% then we can split the data into training and testing sets - we can use leave-one-subject-out cross-validation

% 1. get the features and labels
% 2. split the data into training and testing sets
% 3. define and train the neural network
% 4. test the neural network
% 5. confusion matrix

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
% using leave-one-subject-out cross-validation

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

% convert allFeatures into a matrix
allFeatures = table2array(allFeatures);

trainFeatures = allFeatures(trainIdx, 1:end-1)';
trainTargets = allTargets(trainIdx, :);
trainTargets = onehotencode(categorical(trainTargets), 2)';

testFeatures = allFeatures(testIdx, 1:end-1)';
testTargets = allTargets(testIdx, :);
testTargets = onehotencode(categorical(testTargets), 2)';
%%
% define and train the neural network
numClasses = length(unique(allTargets));
numFeatures = size(trainFeatures, 2);

net = patternnet(10);
net.divideParam.trainRatio = 85/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 0/100;

net.trainParam.showWindow = 0;
[net, tr] = train(net, trainFeatures, trainTargets);
%%
% test the neural network
predictedTargets = net(testFeatures);
testPerformance = perform(net, testTargets, predictedTargets);
view(net);
figure
[c, cm] = confusion(testTargets, predictedTargets);
accuracy = sum(diag(cm))/sum(cm(:));
plotconfusion(testTargets, predictedTargets); % may want to convert numeric to actual labels here
figure
plotperform(tr);
disp(accuracy);
%%
% Brute Force Feature selection - create a struct with the feature name and the accuracy
accuracies = struct('Feature', {}, 'Accuracy', {});
for i = 1:size(trainFeatures, 1)
    fprintf('Feature %d of %d\n', i, size(trainFeatures, 1));

    targets = trainTargets;
    inputs = trainFeatures(i, :);

    net = patternnet(10);
    net.divideParam.trainRatio = 85/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 0/100;

    net.trainParam.showWindow = 0;
    [net, tr] = train(net, inputs, targets);

    predictedTargets = net(testFeatures(i, :));
    [c, cm] = confusion(testTargets, predictedTargets);

    accuracy = sum(diag(cm))/sum(cm(:));
    accuracies(i).Feature = headingNames{i};
    accuracies(i).Accuracy = accuracy;
end
%%
% Try with standardised features
% define and train the neural network
trainFeatures = zscore(trainFeatures);
testFeatures = zscore(testFeatures);

net = patternnet(10);
net.divideParam.trainRatio = 85/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 0/100;

net.trainParam.showWindow = 0;
[net, tr] = train(net, trainFeatures, trainTargets);

% test the neural network
predictedTargets = net(testFeatures);
testPerformance = perform(net, testTargets, predictedTargets);
view(net);
figure
[c, cm] = confusion(testTargets, predictedTargets);
accuracy = sum(diag(cm))/sum(cm(:));
plotconfusion(testTargets, predictedTargets); % may want to convert numeric to actual labels here
figure
plotperform(tr);
disp(accuracy);