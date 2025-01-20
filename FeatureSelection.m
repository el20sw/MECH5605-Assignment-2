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

allFeatures = array2table(allFeatures);
allFeatures.Properties.VariableNames = headingNames;
allFeatures.Activity = allTargets;
allFeatures.Properties.VariableNames = [headingNames, {'Activity'}];
%% MRMR Feature Selection
[idx, scores] = fscmrmr(allFeatures, 'Activity');
bar(scores(idx));
title('MRMR Predictor Importance Estimates');
xlabel('Predictor rank');
ylabel('Predictor importance score');
% get the top 15 features
fscmrmr_significant_features = headingNames(idx(1:15));
fprintf('MRMR Feature Selection\n');
for i = 1:length(fscmrmr_significant_features)
    fprintf('%s\t%f\n', fscmrmr_significant_features{i}, scores(idx(i)));
end

save('mrmr_significant_features.mat', 'idx', 'scores');
%% Predictor Importance of Random Forest
numFeatures = size(allFeatures, 2) - 1;
t = templateTree(MaxNumSplits=1);
rf = fitcensemble(allFeatures,'Activity', ...
    Method='AdaBoostM2', ...
    Learners=t, ...
    NumLearningCycles=numFeatures, ...
    ClassNames=classNames);
importance = predictorImportance(rf);
bar(importance);
xlabel('Predictor rank');
ylabel('Predictor importance score');
% get the top 15 features
[~, idx] = sort(importance, 'descend');
rf_significant_features = allFeatures.Properties.VariableNames(idx(1:15));
fprintf('Random Forest Feature Selection\n');
for i = 1:length(rf_significant_features)
    fprintf('-  %s: %f\n', rf_significant_features{i}, importance(idx(i)));
end
%% Out-of-Bag Predictor Importance of Random Forest
numFeatures = size(allFeatures, 2) - 1;
t = templateTree('Reproducible',true);
model = fitcensemble(allFeatures, 'Activity', ...
    Method='Bag', ...
    NumLearningCycles=numFeatures, ...
    Learners=t);
options = statset('UseParallel',true);
importance = oobPermutedPredictorImportance(model, Options=options);
figure;
bar(importance);
title('Out-of-Bag Permuted Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = model.PredictorNames;
h.XTickLabelRotation = 90;
h.TickLabelInterpreter = 'none';
[~, idx] = sort(importance, 'descend');
oob_significant_features = allFeatures.Properties.VariableNames(idx(1:15));
fprintf('Out-of-Bag Random Forest Feature Selection\n');
for i = 1:length(oob_significant_features)
    fprintf('-  %s: %f\n', oob_significant_features{i}, importance(idx(i)));
end
%% PCA Feature Selection
% Not really useful, is more focused on linear relationships
pca_features = allFeatures{:, 1:end-1};
pca_features = zscore(pca_features);
[coeff, score, latent, ~, explained] = pca(pca_features);
bar(explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');

feature_contribution = abs(coeff) .* repmat(explained', size(coeff, 1), 1);
total_contribution = sum(feature_contribution, 2);

[sorted_contrib, feature_idx] = sort(total_contribution, 'descend');
top_15_idx = feature_idx(1:15);

top_15_features = headingNames(top_15_idx);

fprintf('\nTop 15 features by PCA contribution:\n');
for i = 1:15
    fprintf('-  %s: %.2f%%\n', headingNames{feature_idx(i)}, sorted_contrib(i));
end
%% Brute Force Single Feature Selection
% Train each ANN on a single feature, record the accuracy of the features
layer_sizes = 10:10:100;

w = [size(trainFeatures, 1), length(layer_sizes)];
feature_accuracy = zeros(w);

for sz = 1:length(layer_sizes)
    layers = layer_sizes(sz);
    parfor i = 1:size(trainFeatures, 1)
        features = trainFeatures(i, :);
        targets = trainTargets;

        net = patternnet(layers);
        net.divideParam.trainRatio = 85/100;
        net.divideParam.valRatio = 15/100;
        net.divideParam.testRatio = 0/100;
        net.trainParam.showWindow = 0;
        [net, tr] = train(net, features, targets);

        % test the neural network
        test = testFeatures(i, :);
        predictedTargets = net(test);
        [c, cm] = confusion(testTargets, predictedTargets);
        accuracy = sum(diag(cm))/sum(cm(:));

        feature_accuracy(i) = accuracy;
        feature = headingNames{i};
        fprintf('Feature %s: %f\n', feature, accuracy);
    end
end

% get the top 15 features as an average of each row
avg_feature_accuracy = mean(feature_accuracy, 2);
[sorted_accuracy, idx] = sort(avg_feature_accuracy, 'descend');
bf_single_significant_features = headingNames(idx(1:15));
fprintf('Brute Force Single Feature Selection\n');
for i = 1:length(bf_single_significant_features)
    fprintf('-  %s: %f\n', bf_single_significant_features{i}, sorted_accuracy(i));
end

%% Brute Force Combination Feature Selection - 15 Features
% We train ANNs with random groups of 15 features and
% then select the top 15 features appearing in the best performing ANNs

% set the number of features and iterations
numClasses = 5;
numFeatures = 15;
numIterations = 2500;

% ideal number of iterations given there are 441 features
% numIterations = nchoosek(441, 15);

% preallocate the array to store the top 15 features and their scores
bf_significant_features = cell(numFeatures, 1);
bf_significant_scores = zeros(numFeatures, 1);

% define the random groups of 15 features
featureGroups = cell(numIterations, 1);
for i = 1:numIterations
    featureGroups{i} = randperm(size(trainFeatures, 1), numFeatures);
end

% train the ANNs with the random groups of 15 features
parfor i = 1:numIterations
    features = trainFeatures(featureGroups{i}, :);
    targets = trainTargets;

    net = patternnet([numFeatures, numClasses*2]);
    net.divideParam.trainRatio = 85/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 0/100;
    net.trainParam.showWindow = 0;
    [net, tr] = train(net, features, targets);

    % test the neural network
    test = testFeatures(featureGroups{i}, :);
    predictedTargets = net(test);
    [c, cm] = confusion(testTargets, predictedTargets);
    accuracy = sum(diag(cm))/sum(cm(:));

    bf_significant_scores(i) = accuracy;
    fprintf('Iteration %d: %f\n', i, accuracy);
end

save('brute_force_significant_features.mat', ...
    'featureGroups', 'bf_significant_scores');
%% Analysis of Brute Force
% Get the 1000 top performing feature groups
load brute_force_significant_features.mat

[sortedScores, idx] = sort(bf_significant_scores, 'descend');
bf_significant_features = featureGroups(idx(1:1000));
all_features = cell2mat(bf_significant_features);

feature_counts = zeros(size(trainFeatures, 1), 1);
for i = 1:length(feature_counts)
    feature_counts(i) = sum(all_features(:) == i);
end

% Get the 15 most common features
[~, most_common_features] = maxk(feature_counts, 15);

% Display results
fprintf('\nMost Common Features in Top Groups:\n');
for i = 1:length(most_common_features)
    feature_name = headingNames{most_common_features(i)};
    fprintf('Feature %s appeared %d times\n', ...
        feature_name, feature_counts(most_common_features(i)));
end

%% Comparison of Feature Selection Methods
% Display the top 15 features from each useful feature selection method
fprintf('\nMRMR Feature Selection\n');
for i = 1:length(fscmrmr_significant_features)
    fprintf('%s: %f\n', fscmrmr_significant_features{i}, scores(idx(i)));
end

fprintf('\nRandom Forest Feature Selection\n');
for i = 1:length(rf_significant_features)
    fprintf('%s: %f\n', rf_significant_features{i}, importance(idx(i)));
end

fprintf('\nOut-of-Bag Random Forest Feature Selection\n');
for i = 1:length(oob_significant_features)
    fprintf('%s: %f\n', oob_significant_features{i}, importance(idx(i)));
end

fprintf('\ANN Single Feature Selection\n');
for i = 1:length(bf_single_significant_features)
    fprintf('%s: %f\n', bf_single_significant_features{i}, sorted_accuracy(i));
end
