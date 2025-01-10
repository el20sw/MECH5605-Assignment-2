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

trainFeatures = allFeatures(trainIdx, 1:end-1);
trainTargets = allTargets(trainIdx, :);
trainTargets = categorical(trainTargets);

testFeatures = allFeatures(testIdx, 1:end-1);
testTargets = allTargets(testIdx, :);
testTargets = categorical(testTargets);
%%
% Define and train the SVM - when using an rbf ensure "KernelScale" is set
% to "auto".

model = templateSVM("BoxConstraint", 1e6, "KernelFunction", "rbf", "KernelScale", "auto");
svm = fitcecoc(trainFeatures, trainTargets, 'Learners', model);
%%
% Test the SVM
predictions = predict(svm, testFeatures);
accuracy = sum(predictions == testTargets)/length(testTargets);
confusionchart(testTargets, predictions);

disp(accuracy);
disp(svm.CodingMatrix)
%%
% Trying with only the top 15 features selected in FeatureSelection.m
load pca_data.mat

% Split the data into training and testing sets
trainFeatures = X_pca(trainIdx, :);
trainTargets = Y(trainIdx);
testFeatures = X_pca(testIdx, :);
testTargets = Y(testIdx);
%%
% Define and train the SVM - when using an rbf ensure "KernelScale" is set
% to "auto".

model = templateSVM("BoxConstraint", 1e6, "KernelFunction", "rbf", "KernelScale", "auto");
svm = fitcecoc(trainFeatures, trainTargets, 'Learners', model);
%%
% Test the SVM
predictions = predict(svm, testFeatures);
accuracy = sum(predictions == testTargets)/length(testTargets);
confusionchart(testTargets, predictions);

disp(accuracy);
disp(svm.CodingMatrix)

%%
% Grid search for SVM hyperparameter tuning - we care about the box constraint,
% kernel function (and then polynomial order)

% Define the hyperparameters to search over
boxConstraint = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6];
kernelFunction = {"linear", "polynomial", "rbf"};
polynomialOrder = [2, 3, 4];

% make note of the hyperparameters and the accuaracy they produce
hyperparameters = [];
bestAccuracy = 0;

for kFunc = kernelFunction
    func = string(kFunc);
    if strcmp(func, 'polynomial')
        for order = polynomialOrder
            for box = boxConstraint
                model = templateSVM("BoxConstraint", box, "KernelFunction", func, "PolynomialOrder", order);
                svm = fitcecoc(trainFeatures, trainTargets, 'Learners', model);
                predictions = predict(svm, testFeatures);
                accuracy = sum(predictions == testTargets)/length(testTargets);
                fprintf('Box constraint: %.2f\n', box);
                fprintf('Kernel function: %s\n', func);
                fprintf('Polynomial order: %d\n', order);
                fprintf('Accuracy: %.2f\n', accuracy);

                if accuracy > bestAccuracy
                    hyperparameters = [hyperparameters; [box, func, order, accuracy]];
                end

            end
        end
    else
        for box = boxConstraint
            model = templateSVM("BoxConstraint", box, "KernelFunction", func, "KernelScale", "auto");
            svm = fitcecoc(trainFeatures, trainTargets, 'Learners', model);
            predictions = predict(svm, testFeatures);
            accuracy = sum(predictions == testTargets)/length(testTargets);
            fprintf('Box constraint: %.2f\n', box);
            fprintf('Kernel function: %s\n', func);
            fprintf('Accuracy: %.2f\n', accuracy);

            if accuracy > bestAccuracy
                hyperparameters = [hyperparameters; [box, func, NaN, accuracy]];
            end
        end
    end
end
%%
hyperparameters_copy = hyperparameters;
hyperparameters_copy = array2table(hyperparameters_copy);
hyperparameters_copy.Properties.VariableNames = ["BoxConstraint", "KernelFunction", "PolynomialOrder", "Accuracy"];
hyperparameters_copy = sortrows(hyperparameters_copy, 'Accuracy', 'descend');
head(hyperparameters_copy);
