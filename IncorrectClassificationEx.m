% --- Code to generate misclassified_examples.png ---

% Load the test data and the results of your classifier
test = readmatrix('mnist_test_200_woutliers.csv');
test_images = test(:, 1:784);
correctlabels = test(:, 785);

load('classifier_results.mat', 'predictions'); 


%% --- Find a '4' that was predicted as a '9' ---
% Find indices where the actual label is 4 AND the predicted is 9
idx_4_as_9 = find(correctlabels == 4 & predictions == 9);
% Take the first one we find
image1_idx = idx_4_as_9(1); 
image1_vector = test_images(image1_idx, :);
image1_matrix = reshape(image1_vector, [28 28])';


% --- Find a '7' that was predicted as a '1' ---
% Find indices where the actual label is 7 AND the predicted is 1
idx_7_as_1 = find(correctlabels == 7 & predictions == 1);
% Take the first one we find
image2_idx = idx_7_as_1(1);
image2_vector = test_images(image2_idx, :);
image2_matrix = reshape(image2_vector, [28 28])';


%% --- Create and save the side-by-side plot ---
figure;
% Plot the first misclassified image on the left
subplot(1, 2, 1);
imagesc(image1_matrix);
colormap(gray);
axis square;
title('Actual: 4, Predicted: 9');

% Plot the second misclassified image on the right
subplot(1, 2, 2);
imagesc(image2_matrix);
colormap(gray);
axis square;
title('Actual: 7, Predicted: 1');

% Save the entire figure
saveas(gcf, 'misclassified_examples.png');

fprintf('Saved misclassified_examples.png\n');