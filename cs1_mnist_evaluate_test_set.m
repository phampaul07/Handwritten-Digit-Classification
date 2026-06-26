%% Case Study 1: Classifier Evaluation
% This script evaluates the performance of the trained k-means classifier.

clear;
%close all;

% Load the test data

test = readmatrix('mnist_test_200_woutliers.csv');
correctlabels = test(:, 785);
test = test(:, 1:784); % Extract features from the test data


% Load the trained centroids, their labels, and the outlier thresholds
load(['classifierdata_final.mat']);

% Get k from the loaded centroids
k = size(centroids, 1);



%% --- Classify the Test Set ---
predictions = zeros(size(test, 1), 1); % Required deliverable
for i = 1:size(test, 1)
    [idx, ~] = assign_vector_to_centroid(test(i, 1:784), centroids);
    predicted_label = centroid_labels(idx);
    predictions(i) = predicted_label;
end

% Calculate and display the final accuracy
num_correct = sum(predictions == correctlabels);
accuracy = (num_correct / size(test, 1)) * 100;
fprintf('Classification Accuracy on the Test Set: %.2f%%\n', accuracy);

fprintf('Saving results to classifier_results.mat...\n');
save('classifier_results.mat', 'predictions', 'correctlabels');

%% --- Detect Outliers in the Test Set ---
outliers = zeros(size(test, 1), 1); % Required deliverable
for i = 1:size(test, 1)
    [idx, dist] = assign_vector_to_centroid(test(i, 1:784), centroids);
    if dist > distance_thresholds(idx)
         outliers(i) = 1; % Flag as an outlier
    end
end

% Display outlier results
num_outliers_found = sum(outliers);
fprintf('Identified %d potential outliers in the test set.\n', num_outliers_found);

%% --- Generate Required Submission Plots ---

% Plot 1: Predictions vs. Correct Labels
figure;
plot(1:length(correctlabels), correctlabels, 'o', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(1:length(predictions), predictions, 'x', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
title('Classification Predictions vs. Correct Labels');
xlabel('Test Set Index');
ylabel('Label');
legend('Correct Labels', 'Predicted Labels');
grid on;
saveas(gcf, 'predictions_plot.png');

% Plot 2: Outlier Flags
figure;
stem(1:length(outliers), outliers, 'filled');
title(sprintf('Detected Outliers in Test Set (%d found)', num_outliers_found));
xlabel('Test Set Index');
ylabel('Outlier Flag');
ylim([-0.1, 1.1]);
grid on
saveas(gcf, 'outliers_plot.png');