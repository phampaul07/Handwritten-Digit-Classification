%%Case Study 1: K-Means Clustering
% Name: Derek Nester

clear;
%close all;

%% --- Initialize Data Set ---
fprintf('Loading training data...\n');
train = csvread('mnist_train_1500.csv');
trainsetlabels = train(:,785);
train = train(:,1:784);
train(:,785) = zeros(1500,1);

%% --- K-Means Algorithm Setup ---
k = 18; % SET THIS VALUE from the result of your BestK.m script
max_iter = 50;
fprintf('Starting k-means clustering with k = %d...\n', k);

% Initialize the centroids
centroids = initialize_centroids(train, k);
cost_iteration = zeros(max_iter, 1);

%% --- Run K-Means Clustering ---
for iter = 1:max_iter
    current_cost = 0;
    % Assignment Step
    for i = 1:size(train, 1)
        [idx, dist] = assign_vector_to_centroid(train(i, 1:784), centroids);
        train(i, 785) = idx;
        current_cost = current_cost + dist^2;
    end
    
    cost_iteration(iter) = current_cost;
    fprintf('Iteration %d, Cost: %f\n', iter, current_cost);
    
    % Update Step
    centroids = update_Centroids(train, k);
end
fprintf('Clustering complete.\n');

%% --- MNIST Example Image ---

% Get the pixel data for the first image (row 1, columns 2-end)
first_image_vector = train(10, 1:784);

% Reshape the 784-vector into a 28x28 matrix and transpose it
image_matrix = reshape(first_image_vector, [28 28])'; 

figure;
imagesc(image_matrix);
colormap(gray);
axis square;
title('Example Digit from MNIST');

% Save the figure
saveas(gcf, 'mnist_example.png');

fprintf('Saved mnist_example.png\n');
%% --- Calculate Final Model Parameters and Save ---
fprintf('Calculating final model parameters...\n');

% Calculate the label for each centroid based on the training data
centroid_labels = zeros(k, 1);
for i = 1:k
    indices_in_cluster = find(train(:, 785) == i);
    if ~isempty(indices_in_cluster)
        labels_in_cluster = trainsetlabels(indices_in_cluster);
        centroid_labels(i) = mode(labels_in_cluster);
    else
        centroid_labels(i) = -1; % Handle empty clusters
    end
end

% Calculate distance thresholds for outlier detection from the training data
distance_thresholds = zeros(k, 1);
for i = 1:k
    indices_in_cluster = find(train(:, 785) == i);
    if ~isempty(indices_in_cluster)
        distances = zeros(length(indices_in_cluster), 1);
        for j = 1:length(indices_in_cluster)
            point_idx = indices_in_cluster(j);
            distances(j) = norm(train(point_idx, 1:784) - centroids(i, 1:784));
        end
        distance_thresholds(i) = mean(distances) + 2 * std(distances);
    else
        distance_thresholds(i) = inf; % No threshold for empty clusters
    end
end

% Save the final centroids, their labels, and the outlier thresholds
save('classifierdata.mat', 'centroids', 'centroid_labels', 'distance_thresholds');
fprintf('Classifier data saved to classifierdata.mat\n');

%% --- Generate Required Plots for Clustering Script ---

% Plot 1: K-Means Cost vs. Iteration
figure;
plot(1:max_iter, cost_iteration, '-o', 'LineWidth', 2);
title('K-Means Cost Function vs. Iteration Number');
xlabel('Iteration Number');
ylabel('Cost (Sum of Squared Distances)');
grid on;
saveas(gcf, 'cost_vs_iteration.png');

% Plot 2: Final Centroids
figure;
colormap('gray');
plotsize = ceil(sqrt(k));
for ind = 1:k
    centr = centroids(ind, 1:784);
    subplot(plotsize, plotsize, ind);
    imagesc(reshape(centr, [28 28])');

    title(centroid_labels(ind));
end
sgtitle('Final Cluster Centroids');
saveas(gcf, 'final_centroids.png');