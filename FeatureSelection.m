load preprocessed_with_features.mat

allFeatures = [];

for i = 1:length(dataStruct)
    dataStruct(i).Features.Activity = repmat({dataStruct(i).Activity}, size(dataStruct(i).Features, 1), 1);
    dataStruct(i).Features.Subject = repmat({dataStruct(i).Subject}, size(dataStruct(i).Features, 1), 1);
end

% Concatenate all the features into a single table
for i = 1:length(dataStruct)
    allFeatures = [allFeatures; dataStruct(i).Features];
end

% convert the subjects into a numerical representation
allFeatures.Subject = grp2idx(allFeatures.Subject);

% extract the targets and remove from the features
activityLabels = allFeatures.Activity;
allFeatures.Activity = [];

% get the heading names, sans 'Subject' and 'Activity'
headingNames = allFeatures.Properties.VariableNames;
headingNames = headingNames(~ismember(headingNames, {'Subject', 'Activity'}));

% convert allFeatures into a matrix - for an SVM, columns are features and rows are samples
allFeatures = table2array(allFeatures);

X = allFeatures(:, 1:end-1);
Y = activityLabels(:, :);
Y = categorical(Y);

[n_samples, n_features] = size(X);

% Standardise the features
X_std = zscore(X);

% Do PCA
[coeff, score, latent] = pca(X_std);

% plot the explained variance
figure;
plot(cumsum(latent) / sum(latent));
title('PCA Explained Variance');
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance');

% Get the 15 most significant components and their names
n_components = 15;
[~, idx] = sort(abs(coeff(:, 1)), 'descend');
significant_features = headingNames(idx(1:n_components));

% Extract the 15 most significant components
X_pca = score(:, 1:n_components);

% Save the data as a mat file
save('pca_data.mat', 'X_pca', 'Y', 'significant_features');
