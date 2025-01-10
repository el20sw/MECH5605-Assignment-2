%%
% From ANN.m, this is the generated neural network configuration
% that has been scrapped in favour of the one used in the examples

% layers = [
%     featureInputLayer(numFeatures, 'Normalization', 'zscore')
%     fullyConnectedLayer(64)
%     reluLayer
%     fullyConnectedLayer(64)
%     reluLayer
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer
% ];

% layers = [
%     featureInputLayer(numFeatures)
%     fullyConnectedLayer(16)
%     reluLayer
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer
% ];

% options = trainingOptions('adam', ...
%     'MaxEpochs', 100, ...
%     'MiniBatchSize', 64, ...
%     'Plots', 'training-progress');

% net = trainNetwork(trainFeatures, categorical(trainLabels), layers, options);

% predictedLabels = classify(net, testFeatures');
% accuracy = sum(predictedLabels == categorical(testLabels)) / numel(testLabels);
% fprintf('Accuracy: %.2f\n', accuracy);

tInd = tr.testInd;
predictedTargets = net(inputs(:, tInd));
[c, cm] = confusion(targets(:, tInd), predictedTargets);
