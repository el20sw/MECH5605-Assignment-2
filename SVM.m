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

testFeatures = allFeatures(testIdx, 1:end);
testTargets = allTargets(testIdx, :);
testTargets = categorical(testTargets);

%%
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
hyperparameters = zeros(gridSize, 6);

fprintf('Starting grid search\n');
parfor idx = 1:gridSize
    standardiseData = grid(idx, 1);
    kernelFunction = grid(idx, 2);
    order = str2double(grid(idx, 3));
    coding = grid(idx, 4);
    box = str2double(grid(idx, 5));

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
    params = [box, kernelFunction, order, coding, standardiseData, accuracy];
    hyperparameters(idx, :) = params;

    disp(['Training network ', num2str(idx), ' of ', num2str(gridSize), ...
          ' | Accuracy=', num2str(accuracy)]);
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
%%
% Find the best hyperparameters
load svm_hyperparameters.mat

accuracies = hyperparameters(:, end);
accuracies = str2double(accuracies);
[~, idx] = max(accuracies);

bestHyperparameters = hyperparameters(idx, :);
box = str2double(bestHyperparameters(1));
kernelFunction = bestHyperparameters(2);
% order may be NaN, so we need to check
if isnan(double(bestHyperparameters(3)))
    order = [];
else
    order = str2double(bestHyperparameters(3));
end
coding = bestHyperparameters(4);
standardiseData = bestHyperparameters(5);

model = createTemplateSVM(box, kernelFunction, order, standardiseData);
svm = fitcecoc(trainFeatures, 'Activity', ...
    'Learners', model, ...
    'Coding', coding, ...
    'ClassNames', classNames);
predictions = predict(svm, testFeatures);
predictions = categorical(predictions);
accuracy = sum(predictions == testTargets)/length(testTargets);
fprintf("Accuracy: %.2f\n", accuracy);
confusionchart(testTargets, predictions);
disp(svm.CodingMatrix)
